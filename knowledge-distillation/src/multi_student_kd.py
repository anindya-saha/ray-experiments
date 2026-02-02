"""
Single Teacher, Multiple Student Architectures Knowledge Distillation.

This module implements training multiple heterogeneous student models
from a single teacher simultaneously.

Architecture (from diagram):
                     Shared Input Batch
                           |
                           v
                  +------------------+
                  |   GPU 0          |
                  |   Teacher Model  |
                  |   (e.g., DINOv3) |
                  |   (Frozen)       |
                  +------------------+
                           |
                    features & logits
                           |
         +---------+-------+-------+---------+
         |         |               |         |
         v         v               v         v
   +---------+ +---------+   +---------+ +---------+
   |  GPU 1  | |  GPU 2  |   |  GPU 3  | |  GPU 4  |
   | Student1| | Student2|   | Student3| | Student4|
   |MobileNet| |MobileNet|   | ViT-Tiny| |Efficient|
   | Small   | | Large   |   |         | | Net-B0  |
   | (~5M)   | | (~10M)  |   | (~5M)   | | (~5M)   |
   +---------+ +---------+   +---------+ +---------+
        |          |              |          |
        v          v              v          v
   Optimizer1 Optimizer2    Optimizer3 Optimizer4
     (AdamW)    (AdamW)       (AdamW)    (AdamW)

Key Characteristics:
1. Single Teacher Instance: One teacher provides knowledge to all students
2. Heterogeneous Students: Each student has a DIFFERENT architecture
3. Independent Training: Each student has its own optimizer
4. Shared Input: All models process the same input batch
5. No Gradient Synchronization: Students don't communicate with each other
6. Parallel Execution: All students train simultaneously

Use Cases:
- Train a family of models for different deployment targets
- Compare architectures fairly (same teacher, same data)
- Neural architecture search with knowledge distillation
- Deploy the best performing student for your use case

Teacher and students have separate placement groups:
+-----------------------------------+     +-----------------------------------+
|   TEACHER Placement Group         |     |   STUDENT Placement Group         |
|   - 1 bundle                      |     |   - N bundles (one per student)   |
|   - Strategy from TeacherGroupCfg |     |   - Strategy from StudentGroupCfg |
|   - (PACK/SPREAD has no effect    |     |   - SPREAD: across nodes          |
|      for single bundle)           |     |   - PACK: on same node            |
+-----------------------------------+     +-----------------------------------+
         |                                          |
         v                                          v
   +-----------+                    +-------+ +-------+ +-------+
   |  Teacher  |                    |Student| |Student| |Student|
   |  Actor    |                    |   0   | |   1   | |   2   |
   +-----------+                    +-------+ +-------+ +-------+
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

import ray
from ray.actor import ActorHandle
from ray.util.placement_group import placement_group, remove_placement_group
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
    estimate_model_memory_mb,
    get_dataloaders,
)


# =============================================================================
# Multi-Student Configuration
# =============================================================================


@dataclass
class MultiStudentKDConfig:
    """
    Configuration for multi-student KD training.

    Unlike KDTrainerConfig which has a single student_group,
    this config supports multiple heterogeneous student groups.
    """
    teacher_group: TeacherGroupConfig = field(default_factory=TeacherGroupConfig)
    student_groups: List[StudentGroupConfig] = field(default_factory=list)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self):
        if not self.student_groups:
            # Default: train small, medium, large students
            self.student_groups = [
                StudentGroupConfig(
                    num_workers=1,
                    training_mode="single",
                    architecture="small",
                ),
                StudentGroupConfig(
                    num_workers=1,
                    training_mode="single",
                    architecture="medium",
                ),
                StudentGroupConfig(
                    num_workers=1,
                    training_mode="single",
                    architecture="large",
                ),
            ]

        # Validate all student groups are single-worker
        for i, sg in enumerate(self.student_groups):
            if sg.num_workers != 1:
                raise ValueError(
                    f"MultiStudentKDTrainer requires each student_group to have "
                    f"num_workers=1, but student_groups[{i}] has {sg.num_workers}"
                )
            if sg.training_mode != "single":
                raise ValueError(
                    f"MultiStudentKDTrainer requires training_mode='single', "
                    f"but student_groups[{i}] has '{sg.training_mode}'"
                )


# =============================================================================
# Teacher Actor
# =============================================================================


@ray.remote
class TeacherActor:
    """
    Teacher model actor that runs on a dedicated GPU.

    Provides inference service to all student actors.
    Teacher is frozen and only performs forward passes.
    """

    def __init__(
        self,
        config: TeacherGroupConfig,
    ):
        print("[Teacher] DEBUG: Initializing TeacherActor...")
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Teacher] DEBUG: Device set to {self.device}")

        # Build teacher model
        print("[Teacher] DEBUG: Building teacher model...")
        self.model = config.model_builder().to(self.device)

        # Load checkpoint if provided
        if config.checkpoint_path:
            print(f"[Teacher] DEBUG: Loading checkpoint from {config.checkpoint_path}")
            checkpoint = torch.load(
                config.checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"[Teacher] DEBUG: Checkpoint loaded successfully")

        # Freeze teacher
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.num_params = count_parameters(self.model)
        print(f"[Teacher] DEBUG: Initialization COMPLETE")
        print(f"[Teacher] Device: {self.device}, Parameters: {self.num_params:,} (frozen)")

    def get_logits(self, data: torch.Tensor) -> torch.Tensor:
        """
        Get teacher logits for a batch of data.

        Args:
            data: Input tensor [batch_size, ...]

        Returns:
            Teacher logits [batch_size, num_classes]
        """
        with torch.no_grad():
            data = data.to(self.device)
            logits = self.model(data)
            return logits.cpu()  # Return to CPU for transfer

    def ping(self) -> str:
        """Health check method."""
        return f"Teacher alive on {self.device}"

    def get_info(self) -> Dict[str, Any]:
        """Get teacher information."""
        return {
            "device": str(self.device),
            "num_params": self.num_params,
            "checkpoint": self.config.checkpoint_path,
        }


# =============================================================================
# Student Actor
# =============================================================================


@ray.remote
class StudentActor:
    """
    Student model actor that trains on its own GPU.

    Each student:
    - Has its own architecture
    - Has its own optimizer
    - Receives teacher logits and trains independently
    - Does NOT synchronize gradients with other students
    """

    def __init__(
        self,
        student_id: int,
        config: StudentGroupConfig,
        distillation_config: DistillationConfig,
        training_config: TrainingConfig,
    ):
        print(f"[Student {student_id}] DEBUG: Initializing StudentActor...")
        self.student_id = student_id
        self.config = config
        self.distillation_config = distillation_config
        self.training_config = training_config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Student {student_id}] DEBUG: Device set to {self.device}")

        # Build student model
        if config.architecture not in STUDENT_BUILDERS:
            raise ValueError(f"Unknown architecture: {config.architecture}")

        print(f"[Student {student_id}] DEBUG: Building model ({config.architecture})...")
        self.model = STUDENT_BUILDERS[config.architecture]().to(self.device)
        self.model.train()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=training_config.max_epochs,
        )

        # Data loaders (each student loads full dataset, no sharding)
        print(f"[Student {student_id}] DEBUG: Loading data...")
        download_mnist(training_config.data_dir)
        self.train_loader, self.test_loader = get_dataloaders(
            batch_size=training_config.batch_size,
            data_dir=training_config.data_dir,
            distributed=False,
        )

        self.num_params = count_parameters(self.model)
        print(f"[Student {student_id}] DEBUG: Initialization COMPLETE")
        print(
            f"[Student {student_id}] Architecture: {config.architecture}, "
            f"Device: {self.device}, Parameters: {self.num_params:,}"
        )

    def ping(self) -> str:
        """Health check method."""
        return f"Student {self.student_id} ({self.config.architecture}) alive on {self.device}"

    def train_batch(
        self,
        data: torch.Tensor,
        target: torch.Tensor,
        teacher_logits: torch.Tensor,
        _batch_idx: int = -1,  # Optional for debugging
    ) -> Dict[str, float]:
        """
        Train on a single batch with teacher logits.

        Args:
            data: Input data [batch_size, ...]
            target: Labels [batch_size]
            teacher_logits: Teacher's logits [batch_size, num_classes]
            _batch_idx: Optional batch index for debugging

        Returns:
            dict: Training metrics (loss, kd_loss, ce_loss)
        """
        data = data.to(self.device)
        target = target.to(self.device)
        teacher_logits = teacher_logits.to(self.device)

        self.optimizer.zero_grad(set_to_none=True)

        # Forward pass
        student_logits = self.model(data)

        # Compute distillation loss
        loss, kd_loss, ce_loss = distillation_loss(
            student_logits,
            teacher_logits,
            target,
            temperature=self.distillation_config.temperature,
            alpha=self.distillation_config.alpha,
        )

        # Backward pass
        loss.backward()
        self.optimizer.step()

        # Compute accuracy
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
        """
        Evaluate student on test set.

        Returns:
            dict: Evaluation metrics (accuracy, loss)
        """
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0

        with torch.no_grad():
            for data, target in self.test_loader:
                data = data.to(self.device)
                target = target.to(self.device)

                logits = self.model(data)
                loss = torch.nn.functional.cross_entropy(logits, target)

                total_loss += loss.item()
                pred = torch.argmax(logits, dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)

        self.model.train()

        return {
            "accuracy": correct / total if total > 0 else 0.0,
            "loss": total_loss / len(self.test_loader),
            "correct": correct,
            "total": total,
        }

    def get_state_dict(self) -> Dict[str, Any]:
        """Get model state dict for checkpointing."""
        return {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "student_id": self.student_id,
            "architecture": self.config.architecture,
            "num_params": self.num_params,
        }

    def get_info(self) -> Dict[str, Any]:
        """Get student information."""
        return {
            "student_id": self.student_id,
            "architecture": self.config.architecture,
            "num_params": self.num_params,
            "device": str(self.device),
            "learning_rate": self.config.learning_rate,
        }


# =============================================================================
# Multi-Student KD Trainer
# =============================================================================


class MultiStudentKDTrainer:
    """
    Multi-Student Knowledge Distillation Trainer.

    Trains multiple heterogeneous student models from a single teacher.

    Architecture:
    - One teacher actor on dedicated GPU (separate placement group)
    - Multiple student actors, each on its own GPU (separate placement group)
    - Teacher computes logits once per batch
    - Logits are broadcast to all students
    - Students train independently (no gradient sync)

    This is NOT DDP - students have different architectures and
    don't synchronize with each other.
    """

    def __init__(self, config: MultiStudentKDConfig):
        self.config = config

        self.teacher: Optional[ActorHandle] = None
        self.students: List[ActorHandle] = []
        
        # Separate placement groups for teacher and students
        self.teacher_placement_group = None
        self.student_placement_group = None

        # Track best accuracy per student
        self.best_accuracies: Dict[str, float] = {}

    def _setup(self):
        """Set up placement group, teacher, and student actors."""
        print(f"\n{'='*70}")
        print("MULTI-STUDENT KNOWLEDGE DISTILLATION")
        print("Single Teacher, Multiple Heterogeneous Students")
        print(f"{'='*70}")

        # Print configuration
        print(f"\n--- Teacher Group Config ---")
        print(f"Replicas: {self.config.teacher_group.num_replicas}")
        print(f"Checkpoint: {self.config.teacher_group.checkpoint_path}")

        print(f"\n--- Student Groups ({len(self.config.student_groups)} students) ---")
        for i, sg in enumerate(self.config.student_groups):
            print(f"  Student {i}: {sg.architecture}, lr={sg.learning_rate}")

        print(f"\n--- Distillation Config ---")
        print(f"Temperature: {self.config.distillation.temperature}")
        print(f"Alpha: {self.config.distillation.alpha}")

        print(f"\n--- Training Config ---")
        print(f"Batch size: {self.config.training.batch_size}")
        print(f"Epochs: {self.config.training.max_epochs}")
        print(f"{'='*70}\n")

        # Initialize Ray if not already
        if not ray.is_initialized():
            ray.init()

        num_students = len(self.config.student_groups)
        
        # =====================================================================
        # Create SEPARATE placement groups for teacher and students
        # =====================================================================
        
        # --- Teacher Placement Group (1 bundle) ---
        teacher_resources = self.config.teacher_group.resources_per_worker
        teacher_strategy = self.config.teacher_group.placement_strategy
        
        print(f"DEBUG [Setup]: Creating TEACHER placement group:")
        print(f"DEBUG [Setup]:   - Bundles: 1")
        print(f"DEBUG [Setup]:   - Resources: {teacher_resources}")
        print(f"DEBUG [Setup]:   - Strategy: {teacher_strategy} (no effect for single bundle)")
        
        self.teacher_placement_group = placement_group(
            bundles=[teacher_resources.copy()],
            strategy=teacher_strategy,  # <--- PACK, SPREAD, or STRICT_PACK (Doesn't apply for single bundle)
        )
        
        # --- Student Placement Group (N bundles) ---
        # All students share the same placement group but may have different resources
        # Use first student's placement strategy (they should be consistent)
        student_strategy = self.config.student_groups[0].placement_strategy
        student_bundles = [sg.resources_per_worker.copy() for sg in self.config.student_groups]
        
        print(f"DEBUG [Setup]: Creating STUDENT placement group:")
        print(f"DEBUG [Setup]:   - Bundles: {len(student_bundles)}")
        print(f"DEBUG [Setup]:   - Resources: {student_bundles}")
        print(f"DEBUG [Setup]:   - Strategy: {student_strategy}")
        
        self.student_placement_group = placement_group(
            bundles=student_bundles,
            strategy=student_strategy,  # <--- PACK, SPREAD, or STRICT_PACK
        )
        
        # Wait for both placement groups to be ready
        print("DEBUG [Setup]: Waiting for placement groups to be ready...")
        ray.get([
            self.teacher_placement_group.ready(),
            self.student_placement_group.ready(),
        ])
        print("DEBUG [Setup]: Both placement groups ready")

        # =====================================================================
        # Create Teacher Actor (in teacher placement group, bundle 0)
        # =====================================================================
        print("DEBUG [Setup]: Creating teacher actor...")
        teacher_num_gpus = teacher_resources.get("GPU", 1)
        teacher_num_cpus = teacher_resources.get("CPU", 2)
        
        self.teacher = TeacherActor.options(
            num_gpus=teacher_num_gpus,
            num_cpus=teacher_num_cpus,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=self.teacher_placement_group,
                placement_group_bundle_index=0,
            )
        ).remote(config=self.config.teacher_group)

        # Wait for teacher to initialize
        print("DEBUG [Setup]: Waiting for teacher initialization...")
        teacher_info = ray.get(self.teacher.get_info.remote())
        print(f"DEBUG [Setup]: Teacher ready: {teacher_info['num_params']:,} params on {teacher_info['device']}")

        # =====================================================================
        # Create Student Actors (in student placement group, bundles 0..N-1)
        # =====================================================================
        print(f"DEBUG [Setup]: Creating {num_students} student actors...")
        self.students = []
        for i, student_config in enumerate(self.config.student_groups):
            student_resources = student_config.resources_per_worker
            student_num_gpus = student_resources.get("GPU", 1)
            student_num_cpus = student_resources.get("CPU", 2)
            
            print(f"DEBUG [Setup]: Creating student {i} ({student_config.architecture}) "
                  f"with GPU={student_num_gpus}, CPU={student_num_cpus}, bundle_idx={i}")
            
            student = StudentActor.options(
                num_gpus=student_num_gpus,
                num_cpus=student_num_cpus,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=self.student_placement_group,
                    placement_group_bundle_index=i,  # Students use bundles 0..N-1 in their own PG
                )
            ).remote(
                student_id=i,
                config=student_config,
                distillation_config=self.config.distillation,
                training_config=self.config.training,
            )
            self.students.append(student)
            self.best_accuracies[student_config.architecture] = 0.0

        # Wait for all students to initialize
        print("DEBUG [Setup]: Waiting for all students to initialize...")
        student_infos = ray.get([s.get_info.remote() for s in self.students])
        for info in student_infos:
            print(
                f"DEBUG [Setup]: Student {info['student_id']} ({info['architecture']}): "
                f"{info['num_params']:,} params on {info['device']}"
            )

        print(f"\n{'='*70}\n")

    def _train_epoch(self, epoch: int) -> Dict[str, Dict[str, float]]:
        """
        Train all students for one epoch.

        Returns:
            Dict mapping architecture name to metrics
        """
        print(f"DEBUG [Train]: Starting epoch {epoch}...")
        
        # Get data loader from first student (all have same data)
        # We'll iterate through batches and broadcast to all

        # For simplicity, we get batches on the driver and broadcast
        # In production, you might want to use Ray Data for streaming

        download_mnist(self.config.training.data_dir)
        train_loader, _ = get_dataloaders(
            batch_size=self.config.training.batch_size,
            data_dir=self.config.training.data_dir,
        )
        print(f"DEBUG [Train]: Data loader ready with {len(train_loader)} batches")

        # Metrics per student
        epoch_metrics: Dict[str, Dict[str, float]] = {
            sg.architecture: {"loss": 0.0, "kd_loss": 0.0, "ce_loss": 0.0, "correct": 0, "total": 0}
            for sg in self.config.student_groups
        }

        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx == 0:
                print(f"DEBUG [Train]: First batch - data shape: {data.shape}")
            
            # Step 1: Get teacher logits (single forward pass)
            if batch_idx == 0:
                print("DEBUG [Train]: Getting teacher logits for first batch...")
            teacher_logits = ray.get(self.teacher.get_logits.remote(data))
            if batch_idx == 0:
                print(f"DEBUG [Train]: Teacher logits received, shape: {teacher_logits.shape}")

            # Step 2: Train all students in parallel on the same batch
            if batch_idx == 0:
                print("DEBUG [Train]: Dispatching train_batch to all students...")
            train_futures = [
                student.train_batch.remote(data, target, teacher_logits)
                for student in self.students
            ]

            # Step 3: Collect results
            train_results = ray.get(train_futures)
            if batch_idx == 0:
                print("DEBUG [Train]: All students completed first batch")

            # Step 4: Aggregate metrics
            for sg, result in zip(self.config.student_groups, train_results):
                arch = sg.architecture
                epoch_metrics[arch]["loss"] += result["loss"]
                epoch_metrics[arch]["kd_loss"] += result["kd_loss"]
                epoch_metrics[arch]["ce_loss"] += result["ce_loss"]
                epoch_metrics[arch]["correct"] += result["correct"]
                epoch_metrics[arch]["total"] += result["total"]

            # Progress logging
            if batch_idx % 200 == 0:
                losses_str = ", ".join(
                    f"{sg.architecture}={epoch_metrics[sg.architecture]['loss'] / (batch_idx + 1):.4f}"
                    for sg in self.config.student_groups
                )
                print(f"  Epoch {epoch} [{batch_idx:>4d}/{len(train_loader)}] Loss: {losses_str}")

        # Step schedulers
        print(f"DEBUG [Train]: Stepping schedulers...")
        ray.get([s.step_scheduler.remote() for s in self.students])

        # Compute averages
        num_batches = len(train_loader)
        for arch in epoch_metrics:
            epoch_metrics[arch]["loss"] /= num_batches
            epoch_metrics[arch]["kd_loss"] /= num_batches
            epoch_metrics[arch]["ce_loss"] /= num_batches
            if epoch_metrics[arch]["total"] > 0:
                epoch_metrics[arch]["accuracy"] = (
                    epoch_metrics[arch]["correct"] / epoch_metrics[arch]["total"]
                )
            else:
                epoch_metrics[arch]["accuracy"] = 0.0

        print(f"DEBUG [Train]: Epoch {epoch} COMPLETE")
        return epoch_metrics

    def _evaluate(self) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all students.

        Returns:
            Dict mapping architecture name to evaluation metrics
        """
        print("DEBUG [Eval]: Starting evaluation of all students...")
        eval_futures = [s.evaluate.remote() for s in self.students]
        eval_results = ray.get(eval_futures)
        print("DEBUG [Eval]: Evaluation complete")

        return {
            sg.architecture: result
            for sg, result in zip(self.config.student_groups, eval_results)
        }

    def _cleanup(self):
        """Clean up resources."""
        print("\nDEBUG [Cleanup]: Starting cleanup...")

        # Kill actors
        if self.teacher:
            print("DEBUG [Cleanup]: Killing teacher actor...")
            ray.kill(self.teacher)
        
        for i, student in enumerate(self.students):
            print(f"DEBUG [Cleanup]: Killing student {i}...")
            ray.kill(student)

        # Remove placement groups (separate for teacher and students)
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
        """
        Run the full training loop.

        Returns:
            Dictionary with results for each student
        """
        start_time = time.time()

        try:
            self._setup()

            # Training loop
            print("Starting training...\n")

            for epoch in range(self.config.training.max_epochs):
                epoch_start = time.time()

                # Train
                train_metrics = self._train_epoch(epoch)

                # Evaluate
                eval_metrics = self._evaluate()

                # Update best accuracies and save checkpoints
                for sg in self.config.student_groups:
                    arch = sg.architecture
                    accuracy = eval_metrics[arch]["accuracy"]

                    if accuracy > self.best_accuracies[arch]:
                        self.best_accuracies[arch] = accuracy

                        # Save checkpoint
                        student_idx = self.config.student_groups.index(sg)
                        state_dict = ray.get(self.students[student_idx].get_state_dict.remote())

                        checkpoint_path = (
                            self.config.training.output_dir / f"student_{arch}_best.pt"
                        )
                        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save({
                            **state_dict,
                            "epoch": epoch,
                            "accuracy": accuracy,
                        }, checkpoint_path)

                epoch_time = time.time() - epoch_start

                # Print epoch summary
                print(f"\nEpoch {epoch} Summary (time={epoch_time:.1f}s):")
                print(f"{'Architecture':<12} {'Train Loss':<12} {'Test Acc':<12} {'Best Acc':<12}")
                print("-" * 50)
                for sg in self.config.student_groups:
                    arch = sg.architecture
                    train_loss = train_metrics[arch]["loss"]
                    test_acc = eval_metrics[arch]["accuracy"]
                    best_acc = self.best_accuracies[arch]
                    print(f"{arch:<12} {train_loss:<12.4f} {test_acc:<12.4f} {best_acc:<12.4f}")
                print()

            # Final results
            total_time = time.time() - start_time

            print(f"\n{'='*70}")
            print("TRAINING COMPLETE")
            print(f"{'='*70}")
            print(f"Total time: {total_time:.1f}s")
            print(f"\nFinal Results:")
            print(f"{'Architecture':<12} {'Best Accuracy':<15} {'Parameters':<15}")
            print("-" * 45)

            results = {}
            for sg in self.config.student_groups:
                arch = sg.architecture
                # Get param count
                student_idx = self.config.student_groups.index(sg)
                info = ray.get(self.students[student_idx].get_info.remote())

                print(f"{arch:<12} {self.best_accuracies[arch]:<15.4f} {info['num_params']:,}")

                results[arch] = {
                    "best_accuracy": self.best_accuracies[arch],
                    "num_params": info["num_params"],
                    "checkpoint": str(
                        self.config.training.output_dir / f"student_{arch}_best.pt"
                    ),
                }

            # Find best student
            best_arch = max(self.best_accuracies, key=lambda x: self.best_accuracies[x])
            print(f"\nBest performing student: {best_arch} ({self.best_accuracies[best_arch]:.4f})")
            print(f"{'='*70}")

            return {
                "students": results,
                "best_architecture": best_arch,
                "best_accuracy": self.best_accuracies[best_arch],
                "total_time": total_time,
            }

        finally:
            self._cleanup()


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Main entry point for multi-student KD training."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-Student Knowledge Distillation"
    )
    parser.add_argument(
        "--teacher-checkpoint", type=str, default="./outputs/multi_student/teacher.pt",
        help="Path to teacher checkpoint"
    )
    parser.add_argument(
        "--student-archs", type=str, nargs="+",
        default=["small", "medium", "large"],
        help="Student architectures to train"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Training batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate (same for all students)"
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
        "--output-dir", type=str, default="./outputs/multi_student",
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

        # Create student group configs for each architecture
        student_groups = [
            StudentGroupConfig(
                num_workers=1,
                training_mode="single",
                architecture=arch,
                learning_rate=args.lr,
            )
            for arch in args.student_archs
        ]

        # Create configuration
        config = MultiStudentKDConfig(
            teacher_group=TeacherGroupConfig(
                num_replicas=1,
                checkpoint_path=teacher_checkpoint,
            ),
            student_groups=student_groups,
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
        trainer = MultiStudentKDTrainer(config)
        results = trainer.fit()

        print("\nFinal Results:")
        for arch, data in results["students"].items():
            print(f"  {arch}: accuracy={data['best_accuracy']:.4f}, params={data['num_params']:,}")
        print(f"\nBest: {results['best_architecture']} ({results['best_accuracy']:.4f})")

    finally:
        print("\nShutting down Ray...")
        ray.shutdown()


if __name__ == "__main__":
    main()
