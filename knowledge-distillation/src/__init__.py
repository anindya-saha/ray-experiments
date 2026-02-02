"""
Knowledge Distillation Framework with Ray.

This package provides flexible KD trainers for different deployment scenarios:

Modules:
    common: Shared utilities (models, data, loss functions, configs, base classes)
    single_gpu: Single GPU training (colocated teacher + student)
    ddp_kd: Multi-GPU DDP training (replicated teacher + DDP student)

Configuration:
    TeacherGroupConfig: Configures teacher worker group (replicas, placement)
    StudentGroupConfig: Configures student worker group (workers, placement, training mode)
    DistillationConfig: Distillation hyperparameters (temperature, alpha)
    TrainingConfig: Training loop parameters (batch size, epochs, paths)
    KDTrainerConfig: Complete configuration combining all of the above

Trainers:
    SingleGPUKDTrainer: Colocated teacher + student on one GPU
    DDPKDTrainer: Distributed training with replicated teacher + DDP student

Usage Examples:
--------------

Single GPU Training:
    from src import KDTrainerConfig, TeacherGroupConfig, StudentGroupConfig
    from src import SingleGPUKDTrainer

    config = KDTrainerConfig(
        teacher_group=TeacherGroupConfig(num_replicas=1, checkpoint_path="teacher.pt"),
        student_group=StudentGroupConfig(num_workers=1, training_mode="single", architecture="small"),
        colocate=True,
    )
    trainer = SingleGPUKDTrainer(config)
    results = trainer.fit()

DDP Training (4 GPUs):
    from src import KDTrainerConfig, TeacherGroupConfig, StudentGroupConfig
    from src import DDPKDTrainer

    config = KDTrainerConfig(
        teacher_group=TeacherGroupConfig(num_replicas=4, placement_strategy="SPREAD"),
        student_group=StudentGroupConfig(num_workers=4, training_mode="ddp", placement_strategy="SPREAD"),
    )
    trainer = DDPKDTrainer(config)
    results = trainer.fit()

Command Line:
    python -m src.single_gpu --train-teacher --epochs 10
    python -m src.ddp_kd --num-workers 4 --train-teacher --epochs 10
"""

from .common import (
    # Model builders
    build_teacher,
    build_student_small,
    build_student_medium,
    build_student_large,
    STUDENT_BUILDERS,
    # Utilities
    count_parameters,
    estimate_model_memory_mb,
    # Data
    download_mnist,
    get_dataloaders,
    # Loss
    distillation_loss,
    # Group Configs
    TeacherGroupConfig,
    StudentGroupConfig,
    # Other Configs
    DistillationConfig,
    TrainingConfig,
    KDTrainerConfig,
    # Training utilities
    train_teacher,
    # Base classes
    BaseKDTrainer,
)

from .single_gpu import SingleGPUKDTrainer

from .ddp_kd import (
    DDPWorker,
    DDPKDTrainer,
)

from .multi_student_kd import (
    MultiStudentKDConfig,
    TeacherActor,
    StudentActor,
    MultiStudentKDTrainer,
)

from .pooled_kd import (
    PooledKDConfig,
    TeacherReplica,
    StudentDDPWorker,
    PooledKDTrainer,
)

__all__ = [
    # Models
    "build_teacher",
    "build_student_small",
    "build_student_medium",
    "build_student_large",
    "STUDENT_BUILDERS",
    # Utilities
    "count_parameters",
    "estimate_model_memory_mb",
    # Data
    "download_mnist",
    "get_dataloaders",
    # Loss
    "distillation_loss",
    # Group Configs
    "TeacherGroupConfig",
    "StudentGroupConfig",
    # Other Configs
    "DistillationConfig",
    "TrainingConfig",
    "KDTrainerConfig",
    # Trainers
    "train_teacher",
    "BaseKDTrainer",
    "SingleGPUKDTrainer",
    "DDPWorker",
    "DDPKDTrainer",
    # Multi-Student
    "MultiStudentKDConfig",
    "TeacherActor",
    "StudentActor",
    "MultiStudentKDTrainer",
    # Pooled Teacher/Student
    "PooledKDConfig",
    "TeacherReplica",
    "StudentDDPWorker",
    "PooledKDTrainer",
]
