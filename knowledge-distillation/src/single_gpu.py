"""
Single GPU Knowledge Distillation.

This module implements colocated teacher + student training on a single GPU.

Architecture (from diagram):
+----------------------------------+
|         GPU 0 (e.g., A100)       |
+----------------------------------+
|                                  |
|  +----------------------------+  |
|  | Teacher Model (Frozen)     |  |
|  | - Forward only, no grads   |  |
|  | - Uses ~8-12 GB HBM        |  |
|  +----------------------------+  |
|              |                   |
|              v                   |
|  +----------------------------+  |
|  | Student Model (Trainable)  |  |
|  | - Forward + Backward       |  |
|  | - Uses ~2-4 GB HBM         |  |
|  +----------------------------+  |
|              |                   |
|              v                   |
|  +----------------------------+  |
|  | Activations & Gradients    |  |
|  | - Uses ~4-8 GB HBM         |  |
|  +----------------------------+  |
|              |                   |
|              v                   |
|  +----------------------------+  |
|  | Optimizer States (AdamW)   |  |
|  | - Uses ~2-4 GB HBM         |  |
|  +----------------------------+  |
|                                  |
+----------------------------------+

Use Cases:
- Quick experimentation and prototyping
- When both models fit on a single GPU
- Minimal infrastructure overhead
- Direct tensor transfer (no serialization)

Memory Budget (A100 40GB/80GB):
- Teacher (frozen):     8-12 GB
- Student (trainable):  2-4 GB
- Activations:          4-8 GB
- Optimizer states:     2-4 GB
- Total:                16-28 GB
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ray.util.placement_group import PlacementGroup

from common import (
    BaseKDTrainer,
    DistillationConfig,
    KDTrainerConfig,
    StudentGroupConfig,
    TeacherGroupConfig,
    TrainingConfig,
    STUDENT_BUILDERS,
    count_parameters,
    distillation_loss,
    download_mnist,
    estimate_model_memory_mb,
    get_dataloaders,
)


class SingleGPUKDTrainer(BaseKDTrainer):
    """
    Single GPU Knowledge Distillation Trainer.

    Both teacher and student reside on the same GPU.
    This is the simplest and most efficient setup when models fit in memory.

    Configuration:
    - TeacherGroupConfig: num_replicas=1, placement_strategy ignored (local)
    - StudentGroupConfig: num_workers=1, training_mode="single"
    - colocate=True (implicit for single GPU)

    Advantages:
    - No inter-GPU communication overhead
    - Direct tensor operations (no serialization)
    - Simple debugging and profiling
    - Lowest latency for teacher inference

    Memory Optimization Tips:
    - Use torch.cuda.amp for mixed precision
    - Enable gradient checkpointing for student if needed
    - Teacher is frozen, so no gradient memory for it
    """

    def __init__(self, config: KDTrainerConfig):
        """
        Initialize the Single GPU KD Trainer.

        Args:
            config: Complete trainer configuration.
                    For single GPU, teacher and student groups should have
                    num_replicas=1 and num_workers=1 respectively.
        """
        # Validate config for single GPU
        if config.teacher_group.num_replicas != 1:
            raise ValueError(
                f"SingleGPUKDTrainer requires teacher_group.num_replicas=1, "
                f"got {config.teacher_group.num_replicas}"
            )
        if config.student_group.num_workers != 1:
            raise ValueError(
                f"SingleGPUKDTrainer requires student_group.num_workers=1, "
                f"got {config.student_group.num_workers}"
            )
        if config.student_group.training_mode != "single":
            raise ValueError(
                f"SingleGPUKDTrainer requires student_group.training_mode='single', "
                f"got {config.student_group.training_mode}"
            )

        super().__init__(config)

        self.device: torch.device = None  # type: ignore
        self.teacher: nn.Module = None  # type: ignore
        self.student: nn.Module = None  # type: ignore
        self.optimizer: torch.optim.Optimizer = None  # type: ignore
        self.scheduler: torch.optim.lr_scheduler.LRScheduler = None  # type: ignore
        self.train_loader = None
        self.test_loader = None

        # Metrics tracking
        self.history: Dict[str, list] = {
            "train_loss": [],
            "train_kd_loss": [],
            "train_ce_loss": [],
            "train_accuracy": [],
            "test_accuracy": [],
        }

    def _setup_placement_groups(self):
        """
        For single GPU, no placement groups are needed.

        Both teacher and student run on the local GPU.
        """
        # No placement groups for single GPU - everything is local
        pass

    def _create_workers(self):
        """
        For single GPU, no Ray workers are created.

        Models are created directly on the local device.
        """
        # No Ray workers for single GPU - models created in _setup()
        pass

    def _setup(self):
        """Set up models, optimizers, and data loaders."""

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"\n{'='*60}")
        print("SINGLE GPU KNOWLEDGE DISTILLATION")
        print(f"{'='*60}")
        print(f"Device: {self.device}")

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"GPU: {gpu_name}")
            print(f"GPU Memory: {gpu_memory:.1f} GB")

        # Print group configs
        print(f"\n--- Teacher Group Config ---")
        print(f"Replicas: {self.teacher_group_config.num_replicas}")
        print(f"Resources: {self.teacher_group_config.resources_per_worker}")
        print(f"Placement: {self.teacher_group_config.placement_strategy} (ignored for single GPU)")

        print(f"\n--- Student Group Config ---")
        print(f"Workers: {self.student_group_config.num_workers}")
        print(f"Training mode: {self.student_group_config.training_mode}")
        print(f"Resources: {self.student_group_config.resources_per_worker}")
        print(f"Placement: {self.student_group_config.placement_strategy} (ignored for single GPU)")

        # Data
        print(f"\n--- Data ---")
        print(f"Loading data from {self.training_config.data_dir}")
        download_mnist(self.training_config.data_dir)
        self.train_loader, self.test_loader = get_dataloaders(
            batch_size=self.training_config.batch_size,
            data_dir=self.training_config.data_dir,
            distributed=False, # no distributed data loading for single GPU
        )
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Test batches: {len(self.test_loader)}")

        # Teacher model (frozen)
        print("\n--- Teacher Model ---")
        self.teacher = self.teacher_group_config.model_builder().to(self.device)

        if self.teacher_group_config.checkpoint_path:
            checkpoint = torch.load(
                self.teacher_group_config.checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.teacher.load_state_dict(checkpoint["model_state_dict"])
                print(f"Loaded teacher from: {self.teacher_group_config.checkpoint_path}")
                if "test_accuracy" in checkpoint:
                    print(f"Teacher accuracy: {checkpoint['test_accuracy']:.4f}")
            else:
                self.teacher.load_state_dict(checkpoint)

        # Freeze teacher
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        teacher_params = count_parameters(self.teacher)
        teacher_memory = estimate_model_memory_mb(self.teacher)
        print(f"Parameters: {teacher_params:,} (frozen)")
        print(f"Memory estimate: {teacher_memory:.2f} MB")

        # Student model (trainable)
        print("\n--- Student Model ---")
        arch = self.student_group_config.architecture
        self.student = STUDENT_BUILDERS[arch]().to(self.device)
        self.student.train()

        student_params = count_parameters(self.student)
        student_memory = estimate_model_memory_mb(self.student)
        print(f"Architecture: {arch}")
        print(f"Parameters: {student_params:,} (trainable)")
        print(f"Memory estimate: {student_memory:.2f} MB")
        print(f"Compression ratio: {teacher_params / student_params:.1f}x")

        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=self.student_group_config.learning_rate,
            weight_decay=self.student_group_config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.training_config.max_epochs,
        )

        # Distillation config
        print("\n--- Distillation Config ---")
        print(f"Temperature: {self.distillation_config.temperature}")
        print(f"Alpha (KD weight): {self.distillation_config.alpha}")
        print(f"{'='*60}\n")

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.student.train()

        total_loss = 0.0
        total_kd_loss = 0.0
        total_ce_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to(self.device)
            target = target.to(self.device)

            # Teacher forward (no gradients)
            with torch.no_grad():
                teacher_logits = self.teacher(data)

            # Student forward + backward
            self.optimizer.zero_grad(set_to_none=True)

            student_logits = self.student(data)

            # Distillation loss
            loss, kd_loss, ce_loss = distillation_loss(
                student_logits,
                teacher_logits,
                target,
                temperature=self.distillation_config.temperature,
                alpha=self.distillation_config.alpha,
            )

            loss.backward()
            self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            total_kd_loss += kd_loss.item()
            total_ce_loss += ce_loss.item()

            pred = torch.argmax(student_logits, dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

            # Progress logging
            if batch_idx % 200 == 0:
                print(
                    f"  Epoch {epoch} [{batch_idx:>4d}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.4f} (KD: {kd_loss.item():.4f}, CE: {ce_loss.item():.4f})"
                )

        # Update scheduler
        self.scheduler.step()

        # Epoch metrics
        num_batches = len(self.train_loader)
        metrics = {
            "loss": total_loss / num_batches,
            "kd_loss": total_kd_loss / num_batches,
            "ce_loss": total_ce_loss / num_batches,
            "accuracy": correct / total,
        }

        return metrics

    def _evaluate(self) -> Dict[str, float]:
        """Evaluate the student model."""
        self.student.eval()

        correct = 0
        total = 0
        total_loss = 0.0

        with torch.no_grad():
            for data, target in self.test_loader:
                data = data.to(self.device)
                target = target.to(self.device)

                logits = self.student(data)
                loss = torch.nn.functional.cross_entropy(logits, target)

                total_loss += loss.item()
                pred = torch.argmax(logits, dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)

        self.student.train()

        return {
            "accuracy": correct / total,
            "loss": total_loss / len(self.test_loader),
        }

    def _evaluate_teacher(self) -> Dict[str, float]:
        """Evaluate the teacher model (for comparison)."""
        self.teacher.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data = data.to(self.device)
                target = target.to(self.device)

                logits = self.teacher(data)
                pred = torch.argmax(logits, dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)

        return {
            "accuracy": correct / total,
        }

    def _cleanup(self):
        """Clean up resources."""
        # Clear GPU memory
        if self.teacher is not None:
            del self.teacher
        if self.student is not None:
            del self.student
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Remove placement groups (no-op for single GPU)
        self._remove_placement_groups()

    def fit(self) -> Dict[str, Any]:
        """
        Run the full training loop.

        Returns:
            Dictionary containing training results and metrics.
        """
        start_time = time.time()

        try:
            self._setup()

            # Evaluate teacher baseline
            teacher_metrics = self._evaluate_teacher()
            print(f"Teacher baseline accuracy: {teacher_metrics['accuracy']:.4f}")

            # Training loop
            print("\nStarting training...")
            best_accuracy = 0.0

            for epoch in range(self.training_config.max_epochs):
                epoch_start = time.time()

                # Train
                train_metrics = self._train_epoch(epoch)

                # Evaluate
                test_metrics = self._evaluate()

                # Track history
                self.history["train_loss"].append(train_metrics["loss"])
                self.history["train_kd_loss"].append(train_metrics["kd_loss"])
                self.history["train_ce_loss"].append(train_metrics["ce_loss"])
                self.history["train_accuracy"].append(train_metrics["accuracy"])
                self.history["test_accuracy"].append(test_metrics["accuracy"])

                # Update best
                if test_metrics["accuracy"] > best_accuracy:
                    best_accuracy = test_metrics["accuracy"]

                    # Save best checkpoint
                    checkpoint_path = self.training_config.output_dir / "student_best.pt"
                    torch.save({
                        "model_state_dict": self.student.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "epoch": epoch,
                        "accuracy": best_accuracy,
                        "architecture": self.student_group_config.architecture,
                    }, checkpoint_path)

                epoch_time = time.time() - epoch_start
                current_lr = self.optimizer.param_groups[0]["lr"]

                print(
                    f"Epoch {epoch:>2d}: "
                    f"train_loss={train_metrics['loss']:.4f}, "
                    f"train_acc={train_metrics['accuracy']:.4f}, "
                    f"test_acc={test_metrics['accuracy']:.4f}, "
                    f"lr={current_lr:.6f}, "
                    f"time={epoch_time:.1f}s"
                )

            # Final results
            total_time = time.time() - start_time

            print(f"\n{'='*60}")
            print("TRAINING COMPLETE")
            print(f"{'='*60}")
            print(f"Total time: {total_time:.1f}s")
            print(f"Teacher accuracy: {teacher_metrics['accuracy']:.4f}")
            print(f"Student best accuracy: {best_accuracy:.4f}")
            print(f"Gap: {teacher_metrics['accuracy'] - best_accuracy:.4f}")
            print(f"Checkpoint saved: {self.training_config.output_dir / 'student_best.pt'}")
            print(f"{'='*60}")

            return {
                "teacher_accuracy": teacher_metrics["accuracy"],
                "student_best_accuracy": best_accuracy,
                "student_final_accuracy": test_metrics["accuracy"],
                "total_time": total_time,
                "history": self.history,
                "checkpoint_path": str(self.training_config.output_dir / "student_best.pt"),
            }

        finally:
            self._cleanup()


def main():
    """Main entry point for single GPU KD training."""
    import argparse

    parser = argparse.ArgumentParser(description="Single GPU Knowledge Distillation")
    parser.add_argument("--teacher-checkpoint", type=str, default=None,
                        help="Path to teacher checkpoint")
    parser.add_argument("--student-arch", type=str, default="small",
                        choices=["small", "medium", "large"],
                        help="Student architecture")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--temperature", type=float, default=2.0,
                        help="Distillation temperature")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="KD loss weight (1-alpha for CE)")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Data directory")
    parser.add_argument("--output-dir", type=str, default="./outputs/single_gpu",
                        help="Output directory")
    parser.add_argument("--train-teacher", action="store_true",
                        help="Train teacher first if no checkpoint provided")
    args = parser.parse_args()

    # Optionally train teacher first
    teacher_checkpoint = args.teacher_checkpoint
    if teacher_checkpoint is None and args.train_teacher:
        from common import train_teacher
        print("No teacher checkpoint provided. Training teacher first...")
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
            num_replicas=1,
            resources_per_worker={"GPU": 1, "CPU": 2},
            placement_strategy="PACK",  # Ignored for single GPU
            checkpoint_path=teacher_checkpoint,
        ),
        student_group=StudentGroupConfig(
            num_workers=1,
            resources_per_worker={"GPU": 1, "CPU": 2},
            placement_strategy="PACK",  # Ignored for single GPU
            training_mode="single",
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
        colocate=True,  # Single GPU is always colocated
    )

    # Run training
    trainer = SingleGPUKDTrainer(config)
    results = trainer.fit()

    print("\nFinal Results:")
    print(f"  Teacher accuracy: {results['teacher_accuracy']:.4f}")
    print(f"  Student accuracy: {results['student_best_accuracy']:.4f}")


if __name__ == "__main__":
    main()
