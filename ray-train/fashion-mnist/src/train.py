import argparse
import logging
import os
import tempfile
import uuid
from dataclasses import dataclass, field

import ray
import ray.train
import ray.train.torch
from omegaconf import OmegaConf
from rich.logging import RichHandler

logger = logging.getLogger(__name__)

import torch
from torch.distributed.fsdp import (
    fully_shard,
    FSDPModule,
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
)
from torch.distributed.checkpoint.state_dict import (
    get_state_dict,
    set_state_dict,
    get_model_state_dict,
    StateDictOptions,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.checkpoint.stateful import Stateful
import torch.distributed.checkpoint as dcp

from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from torchvision.models import VisionTransformer

from data import load_dataset

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _flatten(d, parent_key="", sep="."):
    items = {}
    for k, v in d.items():
        key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten(v, key, sep))
        else:
            items[key] = v
    return items


def log_overrides(base_cfg, overrides):
    flat_base = _flatten(OmegaConf.to_container(base_cfg, resolve=True))
    flat_overrides = _flatten(OmegaConf.to_container(overrides, resolve=True))
    for key, new_val in sorted(flat_overrides.items()):
        old_val = flat_base.get(key, "N/A")
        logger.info("%s:%s --> %s:%s", key, old_val, key, new_val)


# ---------------------------------------------------------------------------
# Structured configuration
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    hidden_dim: int = 3840
    mlp_dim: int = 768
    image_size: int = 28
    patch_size: int = 7
    num_layers: int = 12
    num_heads: int = 8
    num_classes: int = 10


@dataclass
class FSDPConfig:
    skip_model_shard: bool = True
    skip_cpu_offload: bool = True


@dataclass
class TrainingConfig:
    strategy: str = "fsdp"
    epochs: int = 1
    learning_rate: float = 0.001
    batch_size: int = 128
    use_float16: bool = False
    data_dir: str = ""
    log_interval: int = 10
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)
    deepspeed_config: str = ""


@dataclass
class ScalingConfig:
    num_workers: int = 2
    use_gpu: bool = True


@dataclass
class FailureConfig:
    max_failures: int = 2


@dataclass
class RunConfig:
    name_prefix: str = "fsdp_mnist"
    storage_path: str = ""
    failure: FailureConfig = field(default_factory=FailureConfig)


@dataclass
class RayConfig:
    scaling: ScalingConfig = field(default_factory=ScalingConfig)
    run: RunConfig = field(default_factory=RunConfig)


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    ray: RayConfig = field(default_factory=RayConfig)


def init_model(model_cfg: dict) -> torch.nn.Module:
    model = VisionTransformer(
        image_size=model_cfg["image_size"],
        patch_size=model_cfg["patch_size"],
        num_layers=model_cfg["num_layers"],
        num_heads=model_cfg["num_heads"],
        hidden_dim=model_cfg["hidden_dim"],
        mlp_dim=model_cfg["mlp_dim"],
        num_classes=model_cfg["num_classes"],
    )

    model.conv_proj = torch.nn.Conv2d(
        in_channels=1,
        out_channels=model_cfg["hidden_dim"],
        kernel_size=model_cfg["patch_size"],
        stride=model_cfg["patch_size"],
    )

    return model


# ---------------------------------------------------------------------------
# FSDP strategy
# ---------------------------------------------------------------------------


def prepare_model_fsdp(
    model: torch.nn.Module,
    fsdp_cfg: dict,
    use_float16: bool = False,
):
    world_size = ray.train.get_context().get_world_size()
    mesh = init_device_mesh(
        device_type="cuda", mesh_shape=(world_size,), mesh_dim_names=("data_parallel",)
    )

    offload_policy = None
    if not fsdp_cfg["skip_cpu_offload"]:
        offload_policy = CPUOffloadPolicy()

    if use_float16:
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
        )
    else:
        mp_policy = MixedPrecisionPolicy()

    for encoder_block in model.encoder.layers.children():
        fully_shard(
            encoder_block,
            mesh=mesh,
            reshard_after_forward=not fsdp_cfg["skip_model_shard"],
            offload_policy=offload_policy,
            mp_policy=mp_policy,
        )

    fully_shard(
        model,
        mesh=mesh,
        reshard_after_forward=not fsdp_cfg["skip_model_shard"],
        offload_policy=offload_policy,
        mp_policy=mp_policy,
    )


class AppState(Stateful):
    def __init__(self, model, optimizer, epoch: int):
        self.model = model
        self.optimizer = optimizer
        self.epoch = epoch

    def state_dict(self) -> dict:
        model_state_dict, optimizer_state_dict = get_state_dict(
            self.model, self.optimizer
        )
        return {
            "model": model_state_dict,
            "optim": optimizer_state_dict,
            "epoch": self.epoch,
        }

    def load_state_dict(self, state_dict):
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"],
        )
        self.epoch = state_dict["epoch"]


def load_fsdp_checkpoint(model, optimizer, ckpt):
    with ckpt.as_directory() as checkpoint_dir:
        app_state = AppState(model, optimizer, 0)
        dcp.load(
            state_dict={"app": app_state},
            checkpoint_id=checkpoint_dir,
        )
        return app_state.epoch


def save_fsdp_checkpoint(model, optimizer, metrics, epoch):
    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
        dcp.save(
            state_dict={"app": AppState(model, optimizer, epoch)},
            checkpoint_id=temp_checkpoint_dir,
        )
        checkpoint = ray.train.Checkpoint.from_directory(temp_checkpoint_dir)
        ray.train.report(metrics, checkpoint=checkpoint)


def save_fsdp_model_for_inference(model, world_rank):
    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
        save_file = os.path.join(temp_checkpoint_dir, "full-model.pt")

        model_state_dict = get_model_state_dict(
            model=model,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
            ),
        )

        checkpoint = None
        if world_rank == 0:
            torch.save(model_state_dict, save_file)
            checkpoint = ray.train.Checkpoint.from_directory(temp_checkpoint_dir)

        ray.train.report(
            metrics={},
            checkpoint=checkpoint,
            checkpoint_dir_name="full-model",
        )


# ---------------------------------------------------------------------------
# DeepSpeed strategy
# ---------------------------------------------------------------------------


def prepare_model_deepspeed(model, ds_config_path):
    import deepspeed
    import json

    with open(ds_config_path) as f:
        ds_config = json.load(f)

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config,
    )
    return model_engine, optimizer


def load_deepspeed_checkpoint(model_engine, ckpt):
    with ckpt.as_directory() as checkpoint_dir:
        _, client_state = model_engine.load_checkpoint(checkpoint_dir)
        return client_state.get("epoch", 0)


def save_deepspeed_checkpoint(model_engine, metrics, epoch):
    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
        model_engine.save_checkpoint(
            temp_checkpoint_dir,
            client_state={"epoch": epoch},
        )
        checkpoint = ray.train.Checkpoint.from_directory(temp_checkpoint_dir)
        ray.train.report(metrics, checkpoint=checkpoint)


def save_deepspeed_model_for_inference(model_engine, world_rank):
    with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
        save_file = os.path.join(temp_checkpoint_dir, "full-model.pt")

        checkpoint = None
        if world_rank == 0:
            state_dict = model_engine.module.state_dict()
            torch.save(state_dict, save_file)
            checkpoint = ray.train.Checkpoint.from_directory(temp_checkpoint_dir)

        ray.train.report(
            metrics={},
            checkpoint=checkpoint,
            checkpoint_dir_name="full-model",
        )


def train_func(config):
    model_cfg = config["model"]
    train_cfg = config["training"]
    strategy = train_cfg["strategy"]

    world_rank = ray.train.get_context().get_world_rank()
    world_size = ray.train.get_context().get_world_size()

    logger.info(
        "Worker rank=%d starting: strategy=%s, world_size=%d",
        world_rank,
        strategy,
        world_size,
    )

    model = init_model(model_cfg)
    logger.info(
        "Model initialized: hidden_dim=%d, num_layers=%d, num_heads=%d",
        model_cfg["hidden_dim"],
        model_cfg["num_layers"],
        model_cfg["num_heads"],
    )

    device = ray.train.torch.get_device()
    torch.cuda.set_device(device)
    model.to(device)

    if strategy == "fsdp":
        prepare_model_fsdp(
            model,
            fsdp_cfg=train_cfg["fsdp"],
            use_float16=train_cfg["use_float16"],
        )
        criterion = CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=train_cfg["learning_rate"])
        train_model = model
        logger.info("FSDP model prepared on device=%s", device)
    elif strategy == "deepspeed":
        train_model, optimizer = prepare_model_deepspeed(
            model,
            train_cfg["deepspeed_config"],
        )
        criterion = CrossEntropyLoss()
        logger.info(
            "DeepSpeed model prepared from config=%s on device=%s",
            train_cfg["deepspeed_config"],
            device,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    start_epoch = 0
    loaded_checkpoint = ray.train.get_checkpoint()
    if loaded_checkpoint:
        logger.info("Resuming from checkpoint")
        if strategy == "fsdp":
            start_epoch = load_fsdp_checkpoint(
                train_model,
                optimizer,
                loaded_checkpoint,
            )
        else:
            start_epoch = load_deepspeed_checkpoint(
                train_model,
                loaded_checkpoint,
            )
        logger.info("Resumed at epoch=%d", start_epoch)

    data_dir = train_cfg["data_dir"]
    if not data_dir:
        raise ValueError("training.data_dir must be set -- data should be pre-staged")
    train_data = load_dataset(data_dir, train=True)
    train_loader = DataLoader(
        train_data,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=2,
    )
    train_loader = ray.train.torch.prepare_data_loader(train_loader)
    logger.info(
        "DataLoader ready: data_dir=%s, batch_size=%d, samples=%d",
        data_dir,
        train_cfg["batch_size"],
        len(train_data),
    )

    epochs = train_cfg["epochs"]
    logger.info("Training epochs %d -> %d", start_epoch, epochs)

    log_interval = train_cfg.get("log_interval", 10)
    total_batches = len(train_loader)

    for epoch in range(start_epoch, epochs):
        running_loss = 0.0
        num_batches = 0

        if world_size > 1:
            train_loader.sampler.set_epoch(epoch)

        for batch_idx, (images, labels) in enumerate(train_loader):
            outputs = train_model(images)
            loss = criterion(outputs, labels)

            if strategy == "deepspeed":
                train_model.backward(loss)
                train_model.step()
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            num_batches += 1

            if world_rank == 0 and (batch_idx + 1) % log_interval == 0:
                logger.info(
                    "Epoch %d/%d [%d/%d] loss=%.6f",
                    epoch + 1,
                    epochs,
                    batch_idx + 1,
                    total_batches,
                    running_loss / num_batches,
                )

        avg_loss = running_loss / num_batches
        metrics = {"loss": avg_loss, "epoch": epoch + 1}

        if strategy == "fsdp":
            save_fsdp_checkpoint(train_model, optimizer, metrics, epoch + 1)
        else:
            save_deepspeed_checkpoint(train_model, metrics, epoch + 1)

        logger.info(
            "Epoch %d/%d complete: loss=%.6f",
            epoch + 1,
            epochs,
            avg_loss,
        )

    logger.info("Training complete, saving model for inference")
    if strategy == "fsdp":
        save_fsdp_model_for_inference(train_model, world_rank)
    else:
        save_deepspeed_model_for_inference(train_model, world_rank)
    logger.info("Inference model saved")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[RichHandler(rich_tracebacks=True)],
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs",
            "config.yaml",
        ),
        help="Path to the YAML config file",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Config overrides in dotlist format (e.g. --override=training.epochs=5)",
    )
    args = parser.parse_args()

    schema = OmegaConf.structured(Config)
    file_cfg = OmegaConf.load(args.config)
    overrides = OmegaConf.from_dotlist(args.override)

    base_cfg = OmegaConf.merge(schema, file_cfg)
    log_overrides(base_cfg, overrides)
    cfg = OmegaConf.merge(base_cfg, overrides)

    train_loop_config = OmegaConf.to_container(cfg, resolve=True)

    scaling_config = ray.train.ScalingConfig(
        num_workers=cfg.ray.scaling.num_workers,
        use_gpu=cfg.ray.scaling.use_gpu,
    )

    training_name = cfg.ray.run.name_prefix + "_" + str(uuid.uuid4())[:8]

    trainer = ray.train.torch.TorchTrainer(
        train_func,
        scaling_config=scaling_config,
        train_loop_config=train_loop_config,
        run_config=ray.train.RunConfig(
            storage_path=cfg.ray.run.storage_path,
            name=training_name,
            failure_config=ray.train.FailureConfig(
                max_failures=cfg.ray.run.failure.max_failures,
            ),
        ),
    )

    logger.info("Starting TorchTrainer: name=%s", training_name)
    result = trainer.fit()
    logger.info("Training result: %s", result)
