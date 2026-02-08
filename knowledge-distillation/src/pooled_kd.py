"""
Dedicated Teacher and Student GPU Pools Knowledge Distillation.

This module implements an advanced KD architecture with separate GPU pools
for teachers and students, with load balancing for teacher inference.

Architecture (Dedicated Teacher and Student GPU Pools):

    +--------------------------------------------------+
    |              Teacher GPU Pool                    |
    |  +----------+  +----------+  +----------+        |
    |  |  GPU 0   |  |  GPU 1   |  |  GPU K-1 |        |
    |  | Teacher  |  | Teacher  |  | Teacher  |        |
    |  | Replica 0|  | Replica 1|  | Replica K|        |
    |  | (Frozen) |  | (Frozen) |  | (Frozen) |        |
    |  +----------+  +----------+  +----------+        |
    +--------------------------------------------------+
                          |
                          v
              +------------------------+
              |    Load Balancer       |
              | (Round-robin/dynamic)  |
              +------------------------+
                          |
            +-------------+-------------+
            |      P2P transfer         |
            v             v             v
    +--------------------------------------------------+
    |              Student GPU Pool                    |
    |  +----------+  +----------+  +----------+        |
    |  | GPU M    |  | GPU M+1  |  | GPU N    |        |
    |  | Student 1|  | Student 2|  | Student N|        |
    |  |  (DDP)   |  |  (DDP)   |  |  (DDP)   |        |
    |  +----------+  +----------+  +----------+        |
    +--------------------------------------------------+
                          |
                          v
              +------------------------+
              |   Gradient AllReduce   |
              |  (Student GPUs only)   |
              +------------------------+

Key Features:
1. Multiple Teacher Replicas: Deploy K teacher replicas on GPUs 0 to K-1
2. Load Balancing: Distribute student batches across available teachers
3. Throughput Matching: Ratio K/(N-K) chosen to balance teacher/student speeds
4. Independent Optimization: Student GPUs synchronize gradients among themselves only
5. Efficient P2P: Teacher outputs transferred via NVLink/PCIe to student GPUs

Configuration Example (8 GPUs):
- GPUs 0-1: Teacher replicas (2)
- GPUs 2-7: Student replicas (6) with DDP
- Ratio balances teacher/student throughput
- Each student batch serviced by available teacher

Problems Solved:
- Memory efficiency: Teachers don't share GPU with students
- GPU utilization: Teachers can run at full capacity for inference
- Throughput: Load balancing prevents teacher bottleneck
- Scalability: Can adjust teacher/student ratio based on workload
"""

from __future__ import annotations

import os
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

import ray
from ray.actor import ActorHandle
from ray.util.actor_pool import ActorPool
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from common import (
    DistillationConfig,
    StudentGroupConfig,
    TeacherGroupConfig,
    TrainingConfig,
    STUDENT_BUILDERS,
    count_parameters,
    distillation_loss,
    download_mnist,
    get_dataloaders,
)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class PooledKDConfig:
    """
    Configuration for Pooled Teacher/Student KD training.

    This config explicitly separates teacher and student pools:
    - teacher_group: Config for teacher pool (num_replicas > 1 for load balancing)
    - student_group: Config for student pool (DDP training)
    """
    teacher_group: TeacherGroupConfig = field(default_factory=lambda: TeacherGroupConfig(
        num_replicas=2,
        placement_strategy="PACK",  # Pack teachers together for locality
    ))
    student_group: StudentGroupConfig = field(default_factory=lambda: StudentGroupConfig(
        num_workers=4,
        training_mode="ddp",
        placement_strategy="SPREAD",  # Spread students for fault tolerance
    ))
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self):
        # Validate
        if self.teacher_group.num_replicas < 1:
            raise ValueError("teacher_group.num_replicas must be >= 1")
        if self.student_group.num_workers < 1:
            raise ValueError("student_group.num_workers must be >= 1")
        if self.student_group.training_mode != "ddp":
            raise ValueError(
                f"PooledKDTrainer requires student_group.training_mode='ddp', "
                f"got '{self.student_group.training_mode}'"
            )


# =============================================================================
# Utilities
# =============================================================================


def find_free_port() -> int:
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# =============================================================================
# Teacher Replica Actor (for Teacher Pool)
# =============================================================================


@ray.remote
class TeacherReplica:
    """
    Teacher replica for the teacher GPU pool.

    Each replica:
    - Runs on its own dedicated GPU
    - Holds a frozen copy of the teacher model
    - Provides inference service to student workers
    - Is managed by a load balancer for request distribution
    """

    def __init__(
        self,
        replica_id: int,
        config: TeacherGroupConfig,
    ):
        print(f"[TeacherReplica {replica_id}] DEBUG: Initializing...")
        self.replica_id = replica_id
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[TeacherReplica {replica_id}] DEBUG: Device set to {self.device}")

        # Build teacher model
        print(f"[TeacherReplica {replica_id}] DEBUG: Building model...")
        self.model = config.model_builder().to(self.device)

        # Load checkpoint if provided
        if config.checkpoint_path:
            print(f"[TeacherReplica {replica_id}] DEBUG: Loading checkpoint...")
            checkpoint = torch.load(
                config.checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"[TeacherReplica {replica_id}] DEBUG: Checkpoint loaded")

        # Freeze teacher
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.num_params = count_parameters(self.model)
        self.inference_count = 0

        print(f"[TeacherReplica {replica_id}] DEBUG: Initialization COMPLETE")
        print(f"[TeacherReplica {replica_id}] Device: {self.device}, Parameters: {self.num_params:,} (frozen)")

    def get_logits(self, data: torch.Tensor) -> torch.Tensor:
        """
        Get teacher logits for a batch.

        Args:
            data: Input tensor [batch_size, ...]

        Returns:
            Logits tensor [batch_size, num_classes] on CPU
        """
        self.inference_count += 1
        with torch.no_grad():
            data = data.to(self.device)
            logits = self.model(data)
            return logits.cpu()  # Return to CPU for P2P transfer

    def get_info(self) -> Dict[str, Any]:
        """Get replica information."""
        return {
            "replica_id": self.replica_id,
            "device": str(self.device),
            "num_params": self.num_params,
            "inference_count": self.inference_count,
        }


# =============================================================================
# Student DDP Worker Actor (for Student Pool)
# =============================================================================


@ray.remote
class StudentDDPWorker:
    """
    Student DDP worker for the student GPU pool.

    Each worker:
    - Runs on its own dedicated GPU
    - Holds a DDP-wrapped student model
    - Receives teacher logits from the teacher pool
    - Participates in gradient AllReduce with other student workers
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        master_addr: str,
        master_port: int,
        config: StudentGroupConfig,
        distillation_config: DistillationConfig,
        training_config: TrainingConfig,
    ):
        self.rank = rank
        self.world_size = world_size
        self.master_addr = master_addr
        self.master_port = master_port
        self.config = config
        self.distillation_config = distillation_config
        self.training_config = training_config

        self.device: torch.device = None  # type: ignore
        self.model: nn.Module = None  # type: ignore
        self.model_ddp: DDP = None  # type: ignore
        self.optimizer: torch.optim.Optimizer = None  # type: ignore
        self.scheduler: torch.optim.lr_scheduler.LRScheduler = None  # type: ignore
        self.train_loader = None
        self.test_loader = None

        self._is_initialized = False

    def initialize(self) -> Dict[str, Any]:
        """Initialize the worker: setup DDP, model, and data."""
        print(f"[StudentDDP {self.rank}] DEBUG: Starting initialization...")
        
        # Set environment variables for DDP
        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = str(self.master_port)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["RANK"] = str(self.rank)
        os.environ["LOCAL_RANK"] = str(0)
        
        print(f"[StudentDDP {self.rank}] DEBUG: Environment set - MASTER={self.master_addr}:{self.master_port}")

        # Initialize process group (students only sync among themselves)
        print(f"[StudentDDP {self.rank}] DEBUG: Calling dist.init_process_group()...")
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{self.master_addr}:{self.master_port}",
            world_size=self.world_size,
            rank=self.rank,
        )
        print(f"[StudentDDP {self.rank}] DEBUG: dist.init_process_group() COMPLETE")

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        print(f"[StudentDDP {self.rank}] DEBUG: Device set to {self.device}")

        # Load data (sharded via DistributedSampler)
        print(f"[StudentDDP {self.rank}] DEBUG: Loading data...")
        download_mnist(self.training_config.data_dir)
        self.train_loader, self.test_loader = get_dataloaders(
            batch_size=self.training_config.batch_size,
            data_dir=self.training_config.data_dir,
            distributed=True,
            rank=self.rank,
            world_size=self.world_size,
        )
        print(f"[StudentDDP {self.rank}] DEBUG: Data loaded - {len(self.train_loader)} train batches")

        # Build student model
        arch = self.config.architecture
        if arch not in STUDENT_BUILDERS:
            raise ValueError(f"Unknown architecture: {arch}")

        print(f"[StudentDDP {self.rank}] DEBUG: Building model ({arch})...")
        self.model = STUDENT_BUILDERS[arch]().to(self.device)

        # Wrap in DDP (gradients sync among student workers only)
        print(f"[StudentDDP {self.rank}] DEBUG: Wrapping in DDP...")
        self.model_ddp = DDP(
            self.model,
            device_ids=[0],
            output_device=0,
        )
        print(f"[StudentDDP {self.rank}] DEBUG: DDP wrapper created")

        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model_ddp.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.training_config.max_epochs,
        )

        self._is_initialized = True
        num_params = count_parameters(self.model)
        print(f"[StudentDDP {self.rank}] DEBUG: Initialization COMPLETE")

        return {
            "rank": self.rank,
            "device": str(self.device),
            "architecture": arch,
            "num_params": num_params,
        }

    def train_step(
        self,
        data: torch.Tensor,
        target: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Train on a single batch with teacher logits.

        Note: This receives teacher logits from the teacher pool
        (P2P transfer from teacher GPUs to student GPUs).

        Args:
            data: Input data [batch_size, ...]
            target: Labels [batch_size]
            teacher_logits: Teacher logits from pool [batch_size, num_classes]

        Returns:
            Training metrics
        """
        data = data.to(self.device)
        target = target.to(self.device)
        teacher_logits = teacher_logits.to(self.device)

        self.optimizer.zero_grad(set_to_none=True)

        # Student forward
        student_logits = self.model_ddp(data)

        # Distillation loss
        loss, kd_loss, ce_loss = distillation_loss(
            student_logits,
            teacher_logits,
            target,
            temperature=self.distillation_config.temperature,
            alpha=self.distillation_config.alpha,
        )

        # Backward (DDP syncs gradients among student workers)
        loss.backward()
        self.optimizer.step()

        pred = torch.argmax(student_logits, dim=1)
        correct = (pred == target).sum().item()

        return {
            "loss": loss.item(),
            "kd_loss": kd_loss.item(),
            "ce_loss": ce_loss.item(),
            "correct": correct,
            "total": target.size(0),
        }

    def step_scheduler(self):
        """Step the learning rate scheduler."""
        self.scheduler.step()

    def evaluate(self) -> Dict[str, float]:
        """Evaluate student on test set."""
        print(f"[StudentDDP {self.rank}] DEBUG: evaluate() starting...")
        self.model_ddp.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data = data.to(self.device)
                target = target.to(self.device)

                logits = self.model_ddp(data)
                pred = torch.argmax(logits, dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)

        self.model_ddp.train()
        print(f"[StudentDDP {self.rank}] DEBUG: Local eval complete - {correct}/{total}")

        # Aggregate across all student workers
        correct_tensor = torch.tensor([correct], device=self.device)
        total_tensor = torch.tensor([total], device=self.device)

        print(f"[StudentDDP {self.rank}] DEBUG: Before dist.all_reduce()...")
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        print(f"[StudentDDP {self.rank}] DEBUG: After dist.all_reduce()")

        return {
            "accuracy": correct_tensor.item() / total_tensor.item(),
            "local_correct": correct,
            "local_total": total,
        }

    def get_state_dict(self) -> Dict[str, Any]:
        """Get model state dict."""
        return {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "architecture": self.config.architecture,
        }

    def cleanup(self) -> None:
        """Cleanup distributed process group."""
        if dist.is_initialized():
            dist.destroy_process_group()
        print(f"StudentDDPWorker {self.rank} cleanup complete")


# =============================================================================
# Pooled KD Trainer (Orchestrator)
# =============================================================================


class PooledKDTrainer:
    """
    Pooled Teacher and Student GPU Pools KD Trainer.

    This trainer manages:
    1. Teacher Pool: Multiple teacher replicas with load balancing
    2. Student Pool: DDP student workers with gradient sync
    3. Data Flow: Load-balanced teacher inference -> P2P transfer -> Student training

    Architecture:
    - Teacher replicas in their own placement group (PACK for locality)
    - Student workers in their own placement group (SPREAD for fault tolerance)
    - ActorPool for load balancing teacher inference requests
    - Gradient AllReduce only among student workers
    """

    def __init__(self, config: PooledKDConfig):
        self.config = config

        # Teacher pool
        self.teacher_replicas: List[ActorHandle] = []
        self.teacher_pool: Optional[ActorPool] = None
        self.teacher_placement_group = None

        # Student pool
        self.student_workers: List[ActorHandle] = []
        self.student_placement_group = None

        # DDP coordination
        self._master_addr: Optional[str] = None
        self._master_port: Optional[int] = None

    def _setup_teacher_pool(self):
        """Create teacher GPU pool with load balancing."""
        num_replicas = self.config.teacher_group.num_replicas
        strategy = self.config.teacher_group.placement_strategy
        teacher_resources = self.config.teacher_group.resources_per_worker

        print(f"DEBUG [TeacherPool]: Creating teacher pool...")
        print(f"DEBUG [TeacherPool]:   - Replicas: {num_replicas}")
        print(f"DEBUG [TeacherPool]:   - Strategy: {strategy}")
        print(f"DEBUG [TeacherPool]:   - Resources per replica: {teacher_resources}")

        # Create placement group for teachers
        print(f"DEBUG [TeacherPool]: Creating placement group...")
        self.teacher_placement_group = placement_group(
            bundles=[teacher_resources.copy() for _ in range(num_replicas)],
            strategy=strategy,
        )
        
        print(f"DEBUG [TeacherPool]: Waiting for placement group to be ready...")
        ray.get(self.teacher_placement_group.ready())
        print(f"DEBUG [TeacherPool]: Placement group ready")

        # Extract resource values for actor options
        teacher_num_gpus = teacher_resources.get("GPU", 1)
        teacher_num_cpus = teacher_resources.get("CPU", 2)

        # Create teacher replicas
        print(f"DEBUG [TeacherPool]: Creating {num_replicas} teacher replicas...")
        self.teacher_replicas = []
        for i in range(num_replicas):
            print(f"DEBUG [TeacherPool]: Creating TeacherReplica {i} with GPU={teacher_num_gpus}, CPU={teacher_num_cpus}")
            replica = TeacherReplica.options(
                num_gpus=teacher_num_gpus,
                num_cpus=teacher_num_cpus,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=self.teacher_placement_group,
                    placement_group_bundle_index=i,
                )
            ).remote(
                replica_id=i,
                config=self.config.teacher_group,
            )
            self.teacher_replicas.append(replica)

        # Wait for initialization
        print(f"DEBUG [TeacherPool]: Waiting for all replicas to initialize...")
        teacher_infos = ray.get([r.get_info.remote() for r in self.teacher_replicas])
        for info in teacher_infos:
            print(f"DEBUG [TeacherPool]: TeacherReplica {info['replica_id']}: {info['num_params']:,} params on {info['device']}")

        # Create ActorPool for load balancing
        self.teacher_pool = ActorPool(self.teacher_replicas)
        print(f"DEBUG [TeacherPool]: Teacher pool ready with {num_replicas} replicas (load-balanced)")

    def _setup_student_pool(self):
        """Create student GPU pool with DDP."""
        num_workers = self.config.student_group.num_workers
        strategy = self.config.student_group.placement_strategy
        student_resources = self.config.student_group.resources_per_worker

        print(f"DEBUG [StudentPool]: Creating student pool...")
        print(f"DEBUG [StudentPool]:   - Workers: {num_workers}")
        print(f"DEBUG [StudentPool]:   - Strategy: {strategy}")
        print(f"DEBUG [StudentPool]:   - Resources per worker: {student_resources}")

        # Create placement group for students
        print(f"DEBUG [StudentPool]: Creating placement group...")
        self.student_placement_group = placement_group(
            bundles=[student_resources.copy() for _ in range(num_workers)],
            strategy=strategy,
        )
        
        print(f"DEBUG [StudentPool]: Waiting for placement group to be ready...")
        ray.get(self.student_placement_group.ready())
        print(f"DEBUG [StudentPool]: Placement group ready")

        # Get master address and port for DDP
        self._master_addr = ray.util.get_node_ip_address()
        self._master_port = find_free_port()
        print(f"DEBUG [StudentPool]: DDP Master: {self._master_addr}:{self._master_port}")

        # Extract resource values for actor options
        student_num_gpus = student_resources.get("GPU", 1)
        student_num_cpus = student_resources.get("CPU", 2)

        # Create student workers
        print(f"DEBUG [StudentPool]: Creating {num_workers} student DDP workers...")
        self.student_workers = []
        for rank in range(num_workers):
            print(f"DEBUG [StudentPool]: Creating StudentDDPWorker rank={rank} with GPU={student_num_gpus}, CPU={student_num_cpus}")
            worker = StudentDDPWorker.options(
                num_gpus=student_num_gpus,
                num_cpus=student_num_cpus,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=self.student_placement_group,
                    placement_group_bundle_index=rank,
                )
            ).remote(
                rank=rank,
                world_size=num_workers,
                master_addr=self._master_addr,
                master_port=self._master_port,
                config=self.config.student_group,
                distillation_config=self.config.distillation,
                training_config=self.config.training,
            )
            self.student_workers.append(worker)

        # Initialize all workers
        print(f"DEBUG [StudentPool]: Initializing all workers...")
        init_results = ray.get([w.initialize.remote() for w in self.student_workers])
        for info in init_results:
            print(
                f"DEBUG [StudentPool]: StudentWorker {info['rank']}: {info['architecture']}, "
                f"{info['num_params']:,} params on {info['device']}"
            )

        print(f"DEBUG [StudentPool]: Student pool ready with {num_workers} DDP workers")

    def _setup(self):
        """Set up both teacher and student pools."""
        print(f"\n{'='*70}")
        print("POOLED TEACHER AND STUDENT GPU POOLS")
        print("Dedicated GPU Pools with Load Balancing")
        print(f"{'='*70}")

        # Print configuration
        print(f"\n--- Teacher Pool Config ---")
        print(f"Replicas: {self.config.teacher_group.num_replicas}")
        print(f"Placement: {self.config.teacher_group.placement_strategy}")
        print(f"Checkpoint: {self.config.teacher_group.checkpoint_path}")

        print(f"\n--- Student Pool Config ---")
        print(f"Workers: {self.config.student_group.num_workers}")
        print(f"Architecture: {self.config.student_group.architecture}")
        print(f"Training mode: {self.config.student_group.training_mode}")
        print(f"Placement: {self.config.student_group.placement_strategy}")

        print(f"\n--- Resource Allocation ---")
        total_gpus = (
            self.config.teacher_group.num_replicas +
            self.config.student_group.num_workers
        )
        print(f"Teacher GPUs: 0 to {self.config.teacher_group.num_replicas - 1}")
        print(f"Student GPUs: {self.config.teacher_group.num_replicas} to {total_gpus - 1}")
        print(f"Total GPUs: {total_gpus}")

        ratio = self.config.teacher_group.num_replicas / self.config.student_group.num_workers
        print(f"Teacher/Student ratio: {ratio:.2f}")

        print(f"\n--- Distillation Config ---")
        print(f"Temperature: {self.config.distillation.temperature}")
        print(f"Alpha: {self.config.distillation.alpha}")

        print(f"{'='*70}\n")

        # Initialize Ray if needed
        if not ray.is_initialized():
            ray.init()

        # Setup pools
        self._setup_teacher_pool()
        print()
        self._setup_student_pool()

        print(f"\n{'='*70}\n")

    def _get_teacher_logits_load_balanced(self, data: torch.Tensor) -> torch.Tensor:
        """
        Get teacher logits using load-balanced dispatch.

        The ActorPool automatically distributes requests across teacher replicas.

        Args:
            data: Input tensor

        Returns:
            Teacher logits
        """
        # Submit to pool (round-robin or least-loaded)
        self.teacher_pool.submit(
            lambda actor, d: actor.get_logits.remote(d),
            data
        )
        # Get result
        return self.teacher_pool.get_next()

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch using both pools."""
        print(f"DEBUG [Train]: Starting epoch {epoch}...")
        
        # Get data loader
        download_mnist(self.config.training.data_dir)
        train_loader, _ = get_dataloaders(
            batch_size=self.config.training.batch_size,
            data_dir=self.config.training.data_dir,
        )
        print(f"DEBUG [Train]: Data loader ready with {len(train_loader)} batches")

        total_loss = 0.0
        total_kd_loss = 0.0
        total_ce_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx == 0:
                print(f"DEBUG [Train]: First batch - data shape: {data.shape}")
            
            # Step 1: Get teacher logits (load balanced across teacher pool)
            if batch_idx == 0:
                print("DEBUG [Train]: Getting teacher logits (load-balanced)...")
            teacher_logits = self._get_teacher_logits_load_balanced(data)
            if batch_idx == 0:
                print(f"DEBUG [Train]: Teacher logits received, shape: {teacher_logits.shape}")

            # Step 2: Train all student workers in parallel
            # Each worker processes a shard of the batch (via DistributedSampler)
            # but we send the full batch for simplicity - workers will use their own data
            if batch_idx == 0:
                print("DEBUG [Train]: Dispatching train_step to all student workers...")
            train_futures = [
                worker.train_step.remote(data, target, teacher_logits)
                for worker in self.student_workers
            ]

            # Step 3: Collect results
            train_results = ray.get(train_futures)
            if batch_idx == 0:
                print("DEBUG [Train]: All student workers completed first batch")

            # Step 4: Aggregate metrics (average across workers)
            batch_loss = sum(r["loss"] for r in train_results) / len(train_results)
            batch_kd_loss = sum(r["kd_loss"] for r in train_results) / len(train_results)
            batch_ce_loss = sum(r["ce_loss"] for r in train_results) / len(train_results)
            batch_correct = sum(r["correct"] for r in train_results)
            batch_samples = sum(r["total"] for r in train_results)

            total_loss += batch_loss
            total_kd_loss += batch_kd_loss
            total_ce_loss += batch_ce_loss
            total_correct += batch_correct
            total_samples += batch_samples

            if batch_idx % 200 == 0:
                print(
                    f"  Epoch {epoch} [{batch_idx:>4d}/{len(train_loader)}] "
                    f"Loss: {batch_loss:.4f}"
                )

        # Step schedulers
        print(f"DEBUG [Train]: Stepping schedulers...")
        ray.get([w.step_scheduler.remote() for w in self.student_workers])
        print(f"DEBUG [Train]: Epoch {epoch} COMPLETE")

        num_batches = len(train_loader)
        return {
            "loss": total_loss / num_batches,
            "kd_loss": total_kd_loss / num_batches,
            "ce_loss": total_ce_loss / num_batches,
            "accuracy": total_correct / total_samples if total_samples > 0 else 0.0,
        }

    def _evaluate(self) -> Dict[str, float]:
        """Evaluate student model."""
        print("DEBUG [Orchestrator]: Starting _evaluate()...")
        
        # CRITICAL: All workers must participate in AllReduce, so call all of them
        print(f"DEBUG [Orchestrator]: Dispatching evaluate to all {len(self.student_workers)} workers...")
        eval_futures = [w.evaluate.remote() for w in self.student_workers]
        
        print("DEBUG [Orchestrator]: Waiting for evaluate results...")
        eval_results = ray.get(eval_futures)
        print("DEBUG [Orchestrator]: Evaluate complete")
        
        # Return result from rank 0 (all have same global accuracy after AllReduce)
        return eval_results[0]

    def _get_teacher_stats(self) -> Dict[str, Any]:
        """Get statistics from teacher pool."""
        infos = ray.get([r.get_info.remote() for r in self.teacher_replicas])
        total_inferences = sum(info["inference_count"] for info in infos)
        return {
            "total_inferences": total_inferences,
            "per_replica": {info["replica_id"]: info["inference_count"] for info in infos},
        }

    def _cleanup(self):
        """Cleanup all resources."""
        print("\nDEBUG [Cleanup]: Starting cleanup...")

        # Cleanup student workers (destroy DDP process groups)
        if self.student_workers:
            try:
                print("DEBUG [Cleanup]: Calling cleanup on student workers...")
                ray.get([w.cleanup.remote() for w in self.student_workers])
                print("DEBUG [Cleanup]: Student DDP cleanup complete")
            except Exception as e:
                print(f"DEBUG [Cleanup]: Warning during student cleanup: {e}")

        # Kill teacher replicas
        for i, replica in enumerate(self.teacher_replicas):
            try:
                print(f"DEBUG [Cleanup]: Killing TeacherReplica {i}...")
                ray.kill(replica)
            except Exception as e:
                print(f"DEBUG [Cleanup]: Warning killing replica {i}: {e}")

        # Remove placement groups
        if self.teacher_placement_group:
            try:
                print("DEBUG [Cleanup]: Removing teacher placement group...")
                remove_placement_group(self.teacher_placement_group)
            except Exception as e:
                print(f"DEBUG [Cleanup]: Warning removing teacher PG: {e}")

        if self.student_placement_group:
            try:
                print("DEBUG [Cleanup]: Removing student placement group...")
                remove_placement_group(self.student_placement_group)
            except Exception as e:
                print(f"DEBUG [Cleanup]: Warning removing student PG: {e}")

        print("DEBUG [Cleanup]: Cleanup complete")

    def fit(self) -> Dict[str, Any]:
        """Run training with pooled architecture."""
        start_time = time.time()

        try:
            self._setup()

            # Training loop
            print("Starting training...\n")
            best_accuracy = 0.0
            history = {"train_loss": [], "test_accuracy": []}

            for epoch in range(self.config.training.max_epochs):
                epoch_start = time.time()

                # Train
                train_metrics = self._train_epoch(epoch)

                # Evaluate
                eval_metrics = self._evaluate()
                test_accuracy = eval_metrics["accuracy"]

                # Track history
                history["train_loss"].append(train_metrics["loss"])
                history["test_accuracy"].append(test_accuracy)

                # Update best
                if test_accuracy > best_accuracy:
                    best_accuracy = test_accuracy

                    # Save checkpoint
                    state_dict = ray.get(self.student_workers[0].get_state_dict.remote())
                    checkpoint_path = self.config.training.output_dir / "student_pooled_best.pt"
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
                    f"time={epoch_time:.1f}s"
                )

            # Final results
            total_time = time.time() - start_time
            teacher_stats = self._get_teacher_stats()

            print(f"\n{'='*70}")
            print("POOLED KD TRAINING COMPLETE")
            print(f"{'='*70}")
            print(f"Total time: {total_time:.1f}s")
            print(f"Teacher replicas: {self.config.teacher_group.num_replicas}")
            print(f"Student workers: {self.config.student_group.num_workers}")
            print(f"Best accuracy: {best_accuracy:.4f}")

            print(f"\nTeacher Pool Statistics:")
            for replica_id, count in teacher_stats["per_replica"].items():
                print(f"  Replica {replica_id}: {count} inferences")
            print(f"  Total: {teacher_stats['total_inferences']} inferences")

            print(f"{'='*70}")

            return {
                "best_accuracy": best_accuracy,
                "total_time": total_time,
                "teacher_replicas": self.config.teacher_group.num_replicas,
                "student_workers": self.config.student_group.num_workers,
                "teacher_stats": teacher_stats,
                "history": history,
            }

        finally:
            self._cleanup()


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Main entry point for pooled KD training."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Pooled Teacher and Student GPU Pools KD"
    )
    parser.add_argument(
        "--teacher-replicas", type=int, default=2,
        help="Number of teacher replicas in the pool"
    )
    parser.add_argument(
        "--student-workers", type=int, default=4,
        help="Number of student DDP workers"
    )
    parser.add_argument(
        "--teacher-checkpoint", type=str, default=None,
        help="Path to teacher checkpoint"
    )
    parser.add_argument(
        "--student-arch", type=str, default="small",
        choices=["small", "medium", "large"],
        help="Student architecture"
    )
    parser.add_argument(
        "--teacher-placement", type=str, default="PACK",
        choices=["PACK", "SPREAD", "STRICT_PACK"],
        help="Teacher pool placement strategy"
    )
    parser.add_argument(
        "--student-placement", type=str, default="SPREAD",
        choices=["PACK", "SPREAD", "STRICT_PACK"],
        help="Student pool placement strategy"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size per GPU"
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--temperature", type=float, default=2.0,
        help="Distillation temperature"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="KD loss weight"
    )
    parser.add_argument(
        "--data-dir", type=str, default="./data",
        help="Data directory"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./outputs/pooled_kd",
        help="Output directory"
    )
    parser.add_argument(
        "--train-teacher", action="store_true",
        help="Train teacher first if no checkpoint"
    )
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

        # Create configuration
        config = PooledKDConfig(
            teacher_group=TeacherGroupConfig(
                num_replicas=args.teacher_replicas,
                placement_strategy=args.teacher_placement,
                checkpoint_path=teacher_checkpoint,
            ),
            student_group=StudentGroupConfig(
                num_workers=args.student_workers,
                training_mode="ddp",
                architecture=args.student_arch,
                placement_strategy=args.student_placement,
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
        )

        # Run training
        trainer = PooledKDTrainer(config)
        results = trainer.fit()

        print("\nFinal Results:")
        print(f"  Best accuracy: {results['best_accuracy']:.4f}")
        print(f"  Training time: {results['total_time']:.1f}s")
        print(f"  Teacher replicas: {results['teacher_replicas']}")
        print(f"  Student workers: {results['student_workers']}")

    finally:
        print("\nShutting down Ray...")
        ray.shutdown()


if __name__ == "__main__":
    main()
