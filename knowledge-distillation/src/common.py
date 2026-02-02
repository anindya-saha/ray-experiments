"""
Common utilities for Knowledge Distillation.

This module provides shared components:
- Model builders (teacher and student architectures)
- Data loading utilities
- Loss functions
- Base classes for worker groups and trainers
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
import torchvision
import torchvision.transforms as transforms

import ray
from ray.types import ObjectRef
from ray.util.placement_group import PlacementGroup, placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy


T = TypeVar("T")


# =============================================================================
# Model Definitions
# =============================================================================


def build_teacher() -> nn.Module:
    """
    Teacher model: 784 -> 512 -> 256 -> 10 (~530K params).

    For MNIST classification with larger capacity.
    Memory estimate (fp32): ~2MB parameters + activations during forward.
    """
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 10),
    )


def build_student_small() -> nn.Module:
    """
    Small student: 784 -> 64 -> 10 (~50K params).

    Lightweight model for edge deployment.
    """
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )


def build_student_medium() -> nn.Module:
    """
    Medium student: 784 -> 128 -> 10 (~101K params).

    Balanced model for moderate resource environments.
    """
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )


def build_student_large() -> nn.Module:
    """
    Large student: 784 -> 256 -> 128 -> 10 (~235K params).

    Higher capacity student for better accuracy.
    """
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )


# Registry of student architectures
STUDENT_BUILDERS: Dict[str, Callable[[], nn.Module]] = {
    "small": build_student_small,
    "medium": build_student_medium,
    "large": build_student_large,
}


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_model_memory_mb(model: nn.Module, dtype: torch.dtype = torch.float32) -> float:
    """Estimate model memory in MB (parameters only, not activations)."""
    bytes_per_param = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
    }
    num_params = sum(p.numel() for p in model.parameters())
    return num_params * bytes_per_param.get(dtype, 4) / (1024 * 1024)


# =============================================================================
# Data Loading
# =============================================================================


def download_mnist(data_dir: Path) -> None:
    """Download MNIST dataset if not present."""
    data_dir.mkdir(parents=True, exist_ok=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    # Download train and test sets
    torchvision.datasets.MNIST(root=str(data_dir), train=True, download=True, transform=transform)
    torchvision.datasets.MNIST(root=str(data_dir), train=False, download=True, transform=transform)
    print(f"MNIST data ready at {data_dir}")


def get_dataloaders(
    batch_size: int,
    data_dir: Path,
    num_workers: int = 2,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test dataloaders.

    Args:
        batch_size: Batch size per GPU
        data_dir: Directory containing MNIST data
        num_workers: Number of data loading workers
        distributed: Whether to use DistributedSampler for DDP
        rank: Current process rank (for distributed)
        world_size: Total number of processes (for distributed)

    Returns:
        Tuple of (train_loader, test_loader)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = torchvision.datasets.MNIST(
        root=str(data_dir), train=True, download=False, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root=str(data_dir), train=False, download=False, transform=transform
    )

    if distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        test_sampler = DistributedSampler(
            test_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )
        shuffle = False  # Sampler handles shuffling
    else:
        train_sampler = None
        test_sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


# =============================================================================
# Loss Functions
# =============================================================================


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 2.0,
    alpha: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute combined distillation loss.

    Loss = alpha * KD_loss + (1 - alpha) * CE_loss

    Where KD_loss is the KL divergence between softened teacher and student
    distributions, and CE_loss is the standard cross-entropy with hard labels.

    Args:
        student_logits: Student model outputs [batch_size, num_classes]
        teacher_logits: Teacher model outputs [batch_size, num_classes]
        labels: Ground truth labels [batch_size]
        temperature: Softening temperature (higher = softer distributions)
        alpha: Weight for KD loss (1-alpha for CE loss)

    Returns:
        Tuple of (total_loss, kd_loss, ce_loss)
    """
    # Knowledge distillation loss (soft targets)
    kd_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature ** 2)

    # Cross-entropy loss (hard targets)
    ce_loss = F.cross_entropy(student_logits, labels)

    # Combined loss
    total_loss = alpha * kd_loss + (1.0 - alpha) * ce_loss

    return total_loss, kd_loss, ce_loss


# =============================================================================
# Configuration Classes
# =============================================================================


@dataclass
class TeacherGroupConfig:
    """
    Configuration for teacher worker group.

    Defines how teacher workers are created and placed:
    - num_replicas: Number of teacher replicas (1 for single, >1 for pooled)
    - resources_per_worker: GPU/CPU resources per replica
    - placement_strategy: How to place replicas (PACK, SPREAD, STRICT_PACK)
    - checkpoint_path: Path to pre-trained teacher weights
    - model_builder: Function to create the teacher model

    Placement Strategies:
    - PACK: Place replicas on same node if possible (maximize locality)
    - SPREAD: Spread replicas across nodes (maximize fault tolerance)
    - STRICT_PACK: Must place on same node (fail if not possible)
    """
    num_replicas: int = 1
    resources_per_worker: Dict[str, float] = field(
        default_factory=lambda: {"GPU": 1, "CPU": 2}
    )
    placement_strategy: str = "PACK"
    checkpoint_path: Optional[str] = None
    model_builder: Callable[[], nn.Module] = field(default=build_teacher)

    def __post_init__(self):
        valid_strategies = {"PACK", "SPREAD", "STRICT_PACK", "STRICT_SPREAD"}
        if self.placement_strategy not in valid_strategies:
            raise ValueError(
                f"Invalid placement_strategy: {self.placement_strategy}. "
                f"Must be one of {valid_strategies}"
            )


@dataclass
class StudentGroupConfig:
    """
    Configuration for student worker group.

    Defines how student workers are created and placed:
    - num_workers: Number of workers (1 for single GPU, >1 for DDP/FSDP)
    - resources_per_worker: GPU/CPU resources per worker
    - placement_strategy: How to place workers
    - training_mode: Training parallelism mode (single, ddp, fsdp)
    - architecture: Student model architecture name
    - learning_rate: Optimizer learning rate
    - weight_decay: Optimizer weight decay

    Training Modes:
    - single: Single GPU, no distribution
    - ddp: Distributed Data Parallel (gradient AllReduce)
    - fsdp: Fully Sharded Data Parallel (model + gradient sharding)
    """
    num_workers: int = 1
    resources_per_worker: Dict[str, float] = field(
        default_factory=lambda: {"GPU": 1, "CPU": 2}
    )
    placement_strategy: str = "SPREAD"
    training_mode: str = "single"  # "single", "ddp", "fsdp"
    architecture: str = "small"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    def __post_init__(self):
        valid_strategies = {"PACK", "SPREAD", "STRICT_PACK", "STRICT_SPREAD"}
        if self.placement_strategy not in valid_strategies:
            raise ValueError(
                f"Invalid placement_strategy: {self.placement_strategy}. "
                f"Must be one of {valid_strategies}"
            )

        valid_modes = {"single", "ddp", "fsdp"}
        if self.training_mode not in valid_modes:
            raise ValueError(
                f"Invalid training_mode: {self.training_mode}. "
                f"Must be one of {valid_modes}"
            )

        if self.architecture not in STUDENT_BUILDERS:
            raise ValueError(
                f"Invalid architecture: {self.architecture}. "
                f"Must be one of {list(STUDENT_BUILDERS.keys())}"
            )

        # Validate num_workers vs training_mode
        if self.training_mode == "single" and self.num_workers > 1:
            raise ValueError(
                "training_mode='single' requires num_workers=1. "
                "Use 'ddp' or 'fsdp' for multi-worker training."
            )


@dataclass
class DistillationConfig:
    """Configuration for distillation hyperparameters."""
    temperature: float = 2.0
    alpha: float = 0.5  # Weight for KD loss


@dataclass
class TrainingConfig:
    """Configuration for training loop."""
    batch_size: int = 64
    max_epochs: int = 10
    data_dir: Path = field(default_factory=lambda: Path("./data").resolve())
    output_dir: Path = field(default_factory=lambda: Path("./outputs").resolve())

    def __post_init__(self):
        self.data_dir = Path(self.data_dir).resolve()
        self.output_dir = Path(self.output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class KDTrainerConfig:
    """
    Complete configuration for a KD trainer.

    This is the top-level config that combines:
    - Teacher group configuration
    - Student group configuration
    - Distillation hyperparameters
    - Training loop parameters
    """
    teacher_group: TeacherGroupConfig = field(default_factory=TeacherGroupConfig)
    student_group: StudentGroupConfig = field(default_factory=StudentGroupConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Whether to colocate teacher and student in same placement group
    # Only applicable when teacher has 1 replica and student has 1 worker
    colocate: bool = False

    def __post_init__(self):
        if self.colocate:
            if self.teacher_group.num_replicas != 1:
                raise ValueError(
                    "colocate=True requires teacher_group.num_replicas=1"
                )
            if self.student_group.num_workers != 1:
                raise ValueError(
                    "colocate=True requires student_group.num_workers=1"
                )


# =============================================================================
# Training Utilities
# =============================================================================


def train_teacher(
    data_dir: Path,
    output_path: Path,
    batch_size: int = 64,
    epochs: int = 10,
    learning_rate: float = 1e-3,
) -> nn.Module:
    """
    Train a teacher model from scratch.

    Args:
        data_dir: Directory containing MNIST data
        output_path: Path to save the trained model
        batch_size: Training batch size
        epochs: Number of training epochs
        learning_rate: Learning rate

    Returns:
        Trained teacher model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training teacher on {device}")

    # Data
    download_mnist(data_dir)
    train_loader, test_loader = get_dataloaders(batch_size, data_dir)

    # Model
    model = build_teacher().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"Teacher parameters: {count_parameters(model):,}")
    print(f"Estimated memory: {estimate_model_memory_mb(model):.2f} MB")

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(data)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = torch.argmax(logits, dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

            if batch_idx % 200 == 0:
                print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")

        scheduler.step()

        # Evaluate
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                logits = model(data)
                pred = torch.argmax(logits, dim=1)
                test_correct += (pred == target).sum().item()
                test_total += target.size(0)

        train_acc = correct / total
        test_acc = test_correct / test_total
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}: loss={avg_loss:.4f}, train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")

    # Save model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "epochs": epochs,
        "test_accuracy": test_acc,
    }, output_path)
    print(f"Teacher saved to {output_path}")

    return model


# =============================================================================
# Abstract Base Classes for KD Trainers
# =============================================================================


class BaseKDTrainer(ABC):
    """
    Abstract base class for all KD trainers.

    Subclasses implement specific topologies:
    - Single GPU (colocated teacher + student)
    - Multi-GPU DDP (replicated teacher + DDP student)
    - Pooled teachers + multiple students

    Each trainer:
    1. Creates placement groups based on TeacherGroupConfig and StudentGroupConfig
    2. Spawns worker actors within those placement groups
    3. Manages data transfer between teacher and student workers
    4. Coordinates training loop and checkpointing
    """

    def __init__(self, config: KDTrainerConfig):
        """
        Initialize the trainer with configuration.

        Args:
            config: Complete trainer configuration including teacher group,
                    student group, distillation, and training configs.
        """
        self.config = config

        # Shortcuts for convenience
        self.teacher_group_config = config.teacher_group
        self.student_group_config = config.student_group
        self.distillation_config = config.distillation
        self.training_config = config.training

        # Placement groups (created in _setup_placement_groups)
        self.teacher_placement_group: Optional[PlacementGroup] = None
        self.student_placement_group: Optional[PlacementGroup] = None

    @abstractmethod
    def _setup_placement_groups(self):
        """
        Create placement groups for teacher and student workers.

        This is where placement strategies are applied:
        - PACK: Workers on same node when possible
        - SPREAD: Workers across different nodes
        - STRICT_PACK: Must be on same node
        - Colocate: Teacher and student share a placement group
        """
        pass

    @abstractmethod
    def _create_workers(self):
        """
        Create teacher and student workers within placement groups.

        Workers are Ray actors scheduled on specific bundles within
        the placement groups.
        """
        pass

    @abstractmethod
    def _setup(self):
        """
        Full setup: placement groups, workers, data loaders.

        Typically calls:
        1. _setup_placement_groups()
        2. _create_workers()
        3. Initialize data loaders
        """
        pass

    @abstractmethod
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        pass

    @abstractmethod
    def _evaluate(self) -> Dict[str, float]:
        """Evaluate the student model."""
        pass

    @abstractmethod
    def _cleanup(self):
        """
        Clean up resources.

        Should:
        1. Destroy distributed process groups (if any)
        2. Kill worker actors
        3. Remove placement groups
        """
        pass

    @abstractmethod
    def fit(self) -> Dict[str, Any]:
        """
        Run the full training loop.

        Returns:
            Dictionary containing training results and metrics.
        """
        pass

    def _remove_placement_groups(self):
        """Helper to remove placement groups safely."""
        if self.teacher_placement_group is not None:
            try:
                remove_placement_group(self.teacher_placement_group)
            except Exception:
                pass  # Already removed or invalid
            self.teacher_placement_group = None

        if self.student_placement_group is not None:
            try:
                remove_placement_group(self.student_placement_group)
            except Exception:
                pass
            self.student_placement_group = None
