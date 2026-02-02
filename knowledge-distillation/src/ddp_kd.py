"""
Multi-GPU Knowledge Distillation with Distributed Data Parallel (DDP).

This module implements distributed training where each GPU has:
- A replicated (frozen) teacher model
- A DDP student model shard processing a batch shard

Architecture:
+------------------+    +------------------+    +------------------+
|   GPU 0 (Rank 0) |    |   GPU 1 (Rank 1) |    |   GPU N (Rank N) |
+------------------+    +------------------+    +------------------+
| Teacher (Frozen) |    | Teacher (Frozen) |    | Teacher (Frozen) |
|    Replicated    |    |    Replicated    |    |    Replicated    |
+------------------+    +------------------+    +------------------+
        |                       |                       |
        v                       v                       v
+------------------+    +------------------+    +------------------+
|  Student (DDP)   |    |  Student (DDP)   |    |  Student (DDP)   |
|  Batch Shard 0   |    |  Batch Shard 1   |    |  Batch Shard N   |
+------------------+    +------------------+    +------------------+
        |                       |                       |
        +----------+------------+-----------+-----------+
                   |                        |
                   v                        v
          +------------------+    +------------------+
          | Gradient AllReduce|    | Optimizer Update |
          |  (Synchronize)   |--->|   (All Ranks)    |
          +------------------+    +------------------+

Data Flow:
1. Each rank receives a different batch shard (via DistributedSampler)
2. Each rank's teacher computes logits for its local shard (no communication)
3. Each rank's student computes forward pass and loss
4. Gradients are synchronized across ranks via AllReduce (NCCL)
5. All ranks update their model parameters identically

Why Replicate Teacher?
- Teacher inference is local (no cross-GPU communication)
- Minimizes latency compared to centralized teacher
- Teacher is frozen, so no gradient sync overhead for it
- Memory trade-off: uses GPU memory on each rank

Configuration:
- TeacherGroupConfig: num_replicas=N (one per DDP worker, replicated)
- StudentGroupConfig: num_workers=N, training_mode="ddp"
- placement_strategy controls worker distribution across nodes
"""

from __future__ import annotations

import os
import socket
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

import ray
from ray.actor import ActorHandle
from ray.util.placement_group import placement_group, PlacementGroup, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from common import (
    BaseKDTrainer,
    DistillationConfig,
    KDTrainerConfig,
    StudentGroupConfig,
    TeacherGroupConfig,
    TrainingConfig,
    STUDENT_BUILDERS,
    build_teacher,
    count_parameters,
    distillation_loss,
    download_mnist,
    get_dataloaders,
)


# =============================================================================
# Utilities
# =============================================================================


def find_free_port() -> int:
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 0))
        s.listen(1)
        return s.getsockname()[1]


# =============================================================================
# DDP Worker Actor
# =============================================================================


@ray.remote
class DDPWorker:
    """
    DDP Worker for distributed knowledge distillation.

    Each worker:
    - Holds a replicated (frozen) teacher model
    - Holds a DDP-wrapped student model
    - Processes a shard of each batch
    - Participates in gradient AllReduce
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        master_addr: str,
        master_port: int,
        config: KDTrainerConfig,
    ):
        self.rank = rank
        self.world_size = world_size
        self.master_addr = master_addr
        self.master_port = master_port
        self.config = config

        self.device: torch.device = None  # type: ignore
        self.teacher: nn.Module = None  # type: ignore
        self.student: nn.Module = None  # type: ignore
        self.student_ddp: DDP = None  # type: ignore
        self.optimizer: torch.optim.Optimizer = None  # type: ignore
        self.scheduler: torch.optim.lr_scheduler.LRScheduler = None  # type: ignore
        self.train_loader = None
        self.test_loader = None

        self._is_initialized = False

    def initialize(self) -> Dict[str, Any]:
        """
        Initialize the worker: setup DDP, models, and data.

        Returns:
            Dictionary with initialization info.
        """
        print(f"[Rank {self.rank}] DEBUG: Starting initialization...")
        
        # Set environment variables for DDP
        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = str(self.master_port)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["RANK"] = str(self.rank)
        os.environ["LOCAL_RANK"] = str(0)  # One GPU per worker
        
        # NCCL debugging - uncomment for verbose output
        # os.environ["NCCL_DEBUG"] = "INFO"
        # os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"
        
        # Network configuration for cross-node communication
        # os.environ["NCCL_SOCKET_IFNAME"] = "eth0"  # Adjust to your network interface
        # os.environ["NCCL_IB_DISABLE"] = "1"  # Disable InfiniBand if not available
        
        print(f"[Rank {self.rank}] DEBUG: Environment set - MASTER={self.master_addr}:{self.master_port}")

        # Set device - use self.device consistently
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(0)  # Local rank 0 (one GPU per worker)

        print(f"[Rank {self.rank}] DEBUG: Device set to {self.device}")
        if torch.cuda.is_available():
            print(f"[Rank {self.rank}] DEBUG: GPU: {torch.cuda.get_device_name(0)}")

        # Initialize process group
        print(f"[Rank {self.rank}] DEBUG: Calling dist.init_process_group()...")
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{self.master_addr}:{self.master_port}",
            world_size=self.world_size,
            rank=self.rank,
        )
        print(f"[Rank {self.rank}] DEBUG: dist.init_process_group() COMPLETE")

        # Load data (each rank gets different shard via DistributedSampler)
        print(f"[Rank {self.rank}] DEBUG: Loading data...")
        download_mnist(self.config.training.data_dir)
        self.train_loader, self.test_loader = get_dataloaders(
            batch_size=self.config.training.batch_size,
            data_dir=self.config.training.data_dir,
            distributed=True,
            rank=self.rank,
            world_size=self.world_size,
        )
        print(f"[Rank {self.rank}] DEBUG: Data loaded - {len(self.train_loader)} train batches")

        # Teacher model (replicated, frozen)
        print(f"[Rank {self.rank}] DEBUG: Building teacher model...")
        self.teacher = self.config.teacher_group.model_builder().to(self.device)

        if self.config.teacher_group.checkpoint_path:
            checkpoint = torch.load(
                self.config.teacher_group.checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.teacher.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.teacher.load_state_dict(checkpoint)
            print(f"[Rank {self.rank}] DEBUG: Loaded teacher from checkpoint")

        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # Count total params (not just trainable) for teacher
        teacher_total_params = sum(p.numel() for p in self.teacher.parameters())
        print(f"[Rank {self.rank}] DEBUG: Teacher ready - {teacher_total_params:,} total params (frozen)")

        # Student model (DDP wrapped)
        arch = self.config.student_group.architecture
        if arch not in STUDENT_BUILDERS:
            raise ValueError(f"Unknown architecture: {arch}")

        print(f"[Rank {self.rank}] DEBUG: Building student model ({arch})...")
        self.student = STUDENT_BUILDERS[arch]().to(self.device)

        # Wrap in DDP
        print(f"[Rank {self.rank}] DEBUG: Wrapping student in DDP...")
        self.student_ddp = DDP(
            self.student,
            device_ids=[0],  # Local GPU index
            output_device=0,
        )
        print(f"[Rank {self.rank}] DEBUG: DDP wrapper created")

        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.student_ddp.parameters(),
            lr=self.config.student_group.learning_rate,
            weight_decay=self.config.student_group.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.training.max_epochs,
        )

        self._is_initialized = True
        print(f"[Rank {self.rank}] DEBUG: Initialization COMPLETE")

        return {
            "rank": self.rank,
            "device": str(self.device),
            "teacher_params": teacher_total_params,
            "student_params": count_parameters(self.student),
        }

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary with training metrics
        """
        print(f"[Rank {self.rank}] DEBUG: train_epoch({epoch}) starting...")
        self.student_ddp.train()

        # Set epoch for DistributedSampler (ensures different shuffle each epoch)
        if hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(epoch)

        total_loss = 0.0
        total_kd_loss = 0.0
        total_ce_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            if batch_idx == 0:
                print(f"[Rank {self.rank}] DEBUG: First batch - data shape: {data.shape}")
            
            data = data.to(self.device)
            target = target.to(self.device)

            # Teacher forward (local, no communication)
            with torch.no_grad():
                teacher_logits = self.teacher(data)

            # Student forward
            self.optimizer.zero_grad(set_to_none=True)
            
            if batch_idx == 0:
                print(f"[Rank {self.rank}] DEBUG: Before student forward pass...")
            
            student_logits = self.student_ddp(data)
            
            if batch_idx == 0:
                print(f"[Rank {self.rank}] DEBUG: After student forward pass")

            # Distillation loss
            loss, kd_loss, ce_loss = distillation_loss(
                student_logits,
                teacher_logits,
                target,
                temperature=self.config.distillation.temperature,
                alpha=self.config.distillation.alpha,
            )

            if batch_idx == 0:
                print(f"[Rank {self.rank}] DEBUG: Before loss.backward() (AllReduce happens here)...")
            
            # Backward (DDP automatically syncs gradients via AllReduce)
            loss.backward()
            
            if batch_idx == 0:
                print(f"[Rank {self.rank}] DEBUG: After loss.backward() - AllReduce complete")

            # Optimizer step (all ranks update identically)
            self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            total_kd_loss += kd_loss.item()
            total_ce_loss += ce_loss.item()

            pred = torch.argmax(student_logits, dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
            
            if batch_idx == 0:
                print(f"[Rank {self.rank}] DEBUG: First batch complete, continuing...")

        # Update scheduler
        self.scheduler.step()
        print(f"[Rank {self.rank}] DEBUG: train_epoch({epoch}) COMPLETE - {total} samples")

        num_batches = len(self.train_loader)
        return {
            "loss": total_loss / num_batches,
            "kd_loss": total_kd_loss / num_batches,
            "ce_loss": total_ce_loss / num_batches,
            "accuracy": correct / total,
            "samples": total,
        }

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the student model.

        Returns:
            Dictionary with evaluation metrics
        """
        print(f"[Rank {self.rank}] DEBUG: evaluate() starting...")
        self.student_ddp.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_loader):
                data = data.to(self.device)
                target = target.to(self.device)

                logits = self.student_ddp(data)
                pred = torch.argmax(logits, dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)

        self.student_ddp.train()
        print(f"[Rank {self.rank}] DEBUG: Local eval complete - {correct}/{total}")

        # Aggregate across all ranks
        correct_tensor = torch.tensor([correct], device=self.device)
        total_tensor = torch.tensor([total], device=self.device)

        print(f"[Rank {self.rank}] DEBUG: Before dist.all_reduce()...")
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        print(f"[Rank {self.rank}] DEBUG: After dist.all_reduce() - global {correct_tensor.item()}/{total_tensor.item()}")

        return {
            "accuracy": correct_tensor.item() / total_tensor.item(),
            "local_correct": correct,
            "local_total": total,
        }

    def evaluate_teacher(self) -> Dict[str, float]:
        """Evaluate teacher model."""
        print(f"[Rank {self.rank}] DEBUG: evaluate_teacher() starting...")
        self.teacher.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_loader):
                data = data.to(self.device)
                target = target.to(self.device)

                logits = self.teacher(data)
                pred = torch.argmax(logits, dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)

        print(f"[Rank {self.rank}] DEBUG: Teacher local eval complete - {correct}/{total}")

        # Aggregate across all ranks
        correct_tensor = torch.tensor([correct], device=self.device)
        total_tensor = torch.tensor([total], device=self.device)

        print(f"[Rank {self.rank}] DEBUG: Before dist.all_reduce() for teacher...")
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        print(f"[Rank {self.rank}] DEBUG: After dist.all_reduce() for teacher")

        return {
            "accuracy": correct_tensor.item() / total_tensor.item(),
        }

    def get_state_dict(self) -> Dict[str, Any]:
        """Get model state dict (from underlying model, not DDP wrapper)."""
        return {
            "model_state_dict": self.student.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "architecture": self.config.student_group.architecture,
        }

    def cleanup(self) -> None:
        """Cleanup distributed process group."""
        if dist.is_initialized():
            dist.destroy_process_group()
        print(f"[Rank {self.rank}] Cleanup complete")


# =============================================================================
# DDP KD Trainer (Orchestrator)
# =============================================================================


class DDPKDTrainer(BaseKDTrainer):
    """
    Distributed Data Parallel Knowledge Distillation Trainer.

    This trainer orchestrates multiple DDPWorker actors to perform
    distributed training with gradient synchronization.

    Architecture:
    - Each worker has a replicated teacher (frozen) and DDP student
    - Workers process different batch shards
    - Gradients are synchronized via NCCL AllReduce
    - All workers update parameters identically

    Configuration:
    - TeacherGroupConfig: num_replicas should match student num_workers
      (each student worker gets a teacher replica)
    - StudentGroupConfig: num_workers >= 2, training_mode="ddp"
    - placement_strategy controls worker distribution
    """

    def __init__(self, config: KDTrainerConfig):
        """
        Initialize the DDP KD Trainer.

        Args:
            config: Complete trainer configuration.
        """
        # Validate config for DDP
        if config.student_group.num_workers < 1:
            raise ValueError(
                f"DDPKDTrainer requires student_group.num_workers >= 1, "
                f"got {config.student_group.num_workers}"
            )
        if config.student_group.training_mode != "ddp":
            raise ValueError(
                f"DDPKDTrainer requires student_group.training_mode='ddp', "
                f"got {config.student_group.training_mode}"
            )

        super().__init__(config)

        self.workers: List[ActorHandle] = []
        self._master_addr: Optional[str] = None
        self._master_port: Optional[int] = None


    def _setup_placement_groups(self):
        """
        Create placement group for student workers.

        For DDP with replicated teacher, we only need one placement group
        for all workers. Each worker gets teacher + student on its GPU.

        The placement_strategy from StudentGroupConfig controls distribution:
        - PACK: All workers on same node (fast AllReduce via NVLink)
        - SPREAD: Workers across nodes (better fault tolerance)
        - STRICT_PACK: Must be on same node
        """

        print(f"Creating placement group: {self.student_group_config.num_workers} workers, strategy={self.student_group_config.placement_strategy}")

        self.student_placement_group = placement_group(
            bundles=[self.student_group_config.resources_per_worker for _ in range(self.student_group_config.num_workers)],
            strategy=self.student_group_config.placement_strategy,
        )
        ray.get(self.student_placement_group.ready())
        print("Placement group ready")

        # Teacher uses same placement group (colocated with students)
        self.teacher_placement_group = self.student_placement_group

    def _create_workers(self):
        """
        Create DDP workers within the placement group.

        Each worker is scheduled on a specific bundle index.
        """
        num_workers = self.student_group_config.num_workers

        # Get master address and port for DDP
        self._master_addr = ray.util.get_node_ip_address()
        self._master_port = find_free_port()
        print(f"DDP Master: {self._master_addr}:{self._master_port}")

        # Create workers
        print(f"Creating {num_workers} DDP workers...")
        
        for rank in range(num_workers):
            # Extract resources from config
            resources = self.student_group_config.resources_per_worker
            num_gpus = resources.get("GPU", 1)
            num_cpus = resources.get("CPU", 2)
            
            worker = DDPWorker.options(
                num_gpus=num_gpus,
                num_cpus=num_cpus,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=self.student_placement_group,
                    placement_group_bundle_index=rank,
                )
            ).remote(
                rank=rank,
                world_size=num_workers,
                master_addr=self._master_addr,
                master_port=self._master_port,
                config=self.config,
            )
            self.workers.append(worker)

        # Initialize all workers
        print("Initializing workers...")
        init_results = ray.get([w.initialize.remote() for w in self.workers])

        for result in init_results:
            print(
                f"  Rank {result['rank']}: device={result['device']}, "
                f"teacher={result['teacher_params']:,} params, "
                f"student={result['student_params']:,} params"
            )

    def _setup(self):
        """Set up placement groups, workers, and data loaders."""
        print(f"\n{'='*60}")
        print("DDP KNOWLEDGE DISTILLATION")
        print(f"{'='*60}")

        # Print group configs
        print(f"\n--- Teacher Group Config ---")
        print(f"Replicas: {self.teacher_group_config.num_replicas} (one per worker)")
        print(f"Checkpoint: {self.teacher_group_config.checkpoint_path}")

        print(f"\n--- Student Group Config ---")
        print(f"Workers: {self.student_group_config.num_workers}")
        print(f"Training mode: {self.student_group_config.training_mode}")
        print(f"Architecture: {self.student_group_config.architecture}")
        print(f"Resources: {self.student_group_config.resources_per_worker}")
        print(f"Placement: {self.student_group_config.placement_strategy}")

        print(f"\n--- Distillation Config ---")
        print(f"Temperature: {self.distillation_config.temperature}")
        print(f"Alpha: {self.distillation_config.alpha}")

        print(f"\n--- Training Config ---")
        print(f"Batch size (per GPU): {self.training_config.batch_size}")
        print(f"Epochs: {self.training_config.max_epochs}")

        print(f"{'='*60}\n")

        # Initialize Ray if not already
        if not ray.is_initialized():
            ray.init()

        self._setup_placement_groups()
        self._create_workers()

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch across all workers."""
        print(f"DEBUG [Orchestrator]: Starting train_epoch({epoch})...")
        
        # Train all workers (they synchronize internally via AllReduce)
        print(f"DEBUG [Orchestrator]: Dispatching train_epoch to {len(self.workers)} workers...")
        train_futures = [w.train_epoch.remote(epoch) for w in self.workers]
        
        print(f"DEBUG [Orchestrator]: Waiting for train_epoch results (ray.get)...")
        train_results = ray.get(train_futures)
        print(f"DEBUG [Orchestrator]: train_epoch results received")

        # Aggregate training metrics (average across workers)
        avg_loss = sum(r["loss"] for r in train_results) / len(train_results)
        avg_kd_loss = sum(r["kd_loss"] for r in train_results) / len(train_results)
        avg_ce_loss = sum(r["ce_loss"] for r in train_results) / len(train_results)
        total_samples = sum(r["samples"] for r in train_results)

        return {
            "loss": avg_loss,
            "kd_loss": avg_kd_loss,
            "ce_loss": avg_ce_loss,
            "samples": total_samples,
        }

    def _evaluate(self) -> Dict[str, float]:
        """Evaluate the student model."""
        print(f"DEBUG [Orchestrator]: Starting _evaluate()...")
        
        # All workers must participate in AllReduce, so call all of them
        print(f"DEBUG [Orchestrator]: Dispatching evaluate to all workers...")
        eval_futures = [w.evaluate.remote() for w in self.workers]
        
        print(f"DEBUG [Orchestrator]: Waiting for evaluate results...")
        eval_results = ray.get(eval_futures)
        print(f"DEBUG [Orchestrator]: evaluate results received")
        
        # Return result from rank 0 (all should have same global accuracy after AllReduce)
        return eval_results[0]

    def _evaluate_teacher(self) -> Dict[str, float]:
        """Evaluate teacher model."""
        print(f"DEBUG [Orchestrator]: Starting _evaluate_teacher()...")
        
        # All workers must participate in AllReduce, so call all of them
        print(f"DEBUG [Orchestrator]: Dispatching evaluate_teacher to all workers...")
        eval_futures = [w.evaluate_teacher.remote() for w in self.workers]
        
        print(f"DEBUG [Orchestrator]: Waiting for evaluate_teacher results...")
        eval_results = ray.get(eval_futures)
        print(f"DEBUG [Orchestrator]: evaluate_teacher results received")
        
        return eval_results[0]

    def _cleanup(self):
        """Cleanup workers and resources."""
        print("\nCleaning up...")

        # Cleanup workers (destroy process groups)
        if self.workers:
            try:
                ray.get([w.cleanup.remote() for w in self.workers])
            except Exception as e:
                print(f"Warning during worker cleanup: {e}")

        # Remove placement groups
        self._remove_placement_groups()

        print("Cleanup complete")

    def fit(self) -> Dict[str, Any]:
        """
        Run distributed training.

        Returns:
            Dictionary with training results
        """
        start_time = time.time()

        try:
            self._setup()

            # Evaluate teacher baseline
            teacher_metrics = self._evaluate_teacher()
            print(f"Teacher baseline accuracy: {teacher_metrics['accuracy']:.4f}")

            # Training loop
            print("\nStarting distributed training...")
            best_accuracy = 0.0
            history = {
                "train_loss": [],
                "test_accuracy": [],
            }

            for epoch in range(self.training_config.max_epochs):
                epoch_start = time.time()

                # Train
                train_metrics = self._train_epoch(epoch)

                # Evaluate
                test_metrics = self._evaluate()
                test_accuracy = test_metrics["accuracy"]

                # Track history
                history["train_loss"].append(train_metrics["loss"])
                history["test_accuracy"].append(test_accuracy)

                # Update best
                if test_accuracy > best_accuracy:
                    best_accuracy = test_accuracy

                    # Save checkpoint from rank 0
                    state_dict = ray.get(self.workers[0].get_state_dict.remote())
                    checkpoint_path = self.training_config.output_dir / "student_ddp_best.pt"
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        **state_dict,
                        "epoch": epoch,
                        "accuracy": best_accuracy,
                    }, checkpoint_path)

                epoch_time = time.time() - epoch_start

                print(
                    f"Epoch {epoch:>2d}: "
                    f"loss={train_metrics['loss']:.4f} "
                    f"(KD={train_metrics['kd_loss']:.4f}, CE={train_metrics['ce_loss']:.4f}), "
                    f"test_acc={test_accuracy:.4f}, "
                    f"samples={train_metrics['samples']}, "
                    f"time={epoch_time:.1f}s"
                )

            # Final results
            total_time = time.time() - start_time

            print(f"\n{'='*60}")
            print("DDP TRAINING COMPLETE")
            print(f"{'='*60}")
            print(f"Total time: {total_time:.1f}s")
            print(f"Workers: {self.student_group_config.num_workers}")
            print(f"Placement: {self.student_group_config.placement_strategy}")
            print(f"Teacher accuracy: {teacher_metrics['accuracy']:.4f}")
            print(f"Student best accuracy: {best_accuracy:.4f}")
            print(f"Gap: {teacher_metrics['accuracy'] - best_accuracy:.4f}")
            print(f"{'='*60}")

            return {
                "teacher_accuracy": teacher_metrics["accuracy"],
                "student_best_accuracy": best_accuracy,
                "total_time": total_time,
                "num_workers": self.student_group_config.num_workers,
                "history": history,
            }

        finally:
            self._cleanup()


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Main entry point for DDP KD training."""
    import argparse

    parser = argparse.ArgumentParser(description="DDP Knowledge Distillation")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="Number of DDP workers (GPUs)")
    parser.add_argument("--teacher-checkpoint", type=str, default=None,
                        help="Path to teacher checkpoint")
    parser.add_argument("--student-arch", type=str, default="small",
                        choices=["small", "medium", "large"],
                        help="Student architecture")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--temperature", type=float, default=2.0,
                        help="Distillation temperature")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="KD loss weight")
    parser.add_argument("--placement", type=str, default="SPREAD",
                        choices=["SPREAD", "PACK", "STRICT_PACK"],
                        help="Placement strategy for workers")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Data directory")
    parser.add_argument("--output-dir", type=str, default="./outputs/ddp_kd",
                        help="Output directory")
    parser.add_argument("--train-teacher", action="store_true",
                        help="Train teacher first if no checkpoint")
    args = parser.parse_args()

    # Initialize Ray
    print("Initializing Ray...")
    ray.init()

    try:
        # Optionally train teacher first
        teacher_checkpoint = args.teacher_checkpoint
        if teacher_checkpoint is None and args.train_teacher:
            from common import train_teacher
            print("No teacher checkpoint. Training teacher first...")
            teacher_path = Path(args.output_dir) / "teacher.pt"
            train_teacher(
                data_dir=Path(args.data_dir),
                output_path=teacher_path,
                batch_size=args.batch_size,
                epochs=args.epochs,
            )
            teacher_checkpoint = str(teacher_path)

        # Create configuration using group configs
        config = KDTrainerConfig(
            teacher_group=TeacherGroupConfig(
                num_replicas=args.num_workers,  # One replica per DDP worker
                resources_per_worker={"GPU": 1, "CPU": 2},
                placement_strategy=args.placement,
                checkpoint_path=teacher_checkpoint,
            ),
            student_group=StudentGroupConfig(
                num_workers=args.num_workers,
                resources_per_worker={"GPU": 1, "CPU": 2},
                placement_strategy=args.placement,
                training_mode="ddp",
                architecture=args.student_arch,
                learning_rate=args.lr,
            ),
            distillation=DistillationConfig(
                temperature=args.temperature,
                alpha=args.alpha,
            ),
            training=TrainingConfig(
                batch_size=args.batch_size,
                max_epochs=args.epochs,
                data_dir=Path(args.data_dir),
                output_dir=Path(args.output_dir),
            ),
            colocate=False,  # DDP workers are distributed
        )

        # Run training
        trainer = DDPKDTrainer(config)
        results = trainer.fit()

        print("\nFinal Results:")
        print(f"  Teacher accuracy: {results['teacher_accuracy']:.4f}")
        print(f"  Student accuracy: {results['student_best_accuracy']:.4f}")
        print(f"  Training time: {results['total_time']:.1f}s")

    finally:
        print("\nShutting down Ray...")
        ray.shutdown()


if __name__ == "__main__":
    main()
