# Knowledge Distillation Framework with Ray

A flexible, scalable knowledge distillation framework built on Ray, designed to support
multiple training topologies from single-GPU experimentation to production-scale distributed training.


## Design Choices Summary

| Design Choice | Benefit |
|---------------|---------|
| **TeacherGroupConfig / StudentGroupConfig** | Clean separation of model and deployment config |
| **Ray Placement strategies** | Explicit control over worker locality and fault tolerance |
| **Ray ActorPool for teachers** | Automatic load balancing, throughput matching |
| **ray.get() for data transfer** | Explicit, debuggable data flow |
| **Separate placement groups** | Resource isolation between teacher and student pools |
| **Configurable training_mode for student** | Same config works for single/ddp/fsdp |


## Table of Contents

- [Overview](#overview)
- [Design Philosophy](#design-philosophy)
- [Core Abstractions](#core-abstractions)
- [Supported Architectures](#supported-architectures)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration Reference](#configuration-reference)
- [Architecture Deep Dive](#architecture-deep-dive)
- [When to Use Which Trainer](#when-to-use-which-trainer)

---

## Overview

This framework provides a unified approach to knowledge distillation that scales from
quick prototyping on a single GPU to production deployments across multiple nodes.

**Key Features:**
- Flexible teacher/student placement via Ray placement groups
- Load-balanced teacher inference with ActorPool
- DDP and multi-architecture student training
- Clean separation between configuration and execution
- Explicit resource management and GPU allocation

---



## Design Philosophy

### Why Group Configs?

The framework is built around two core configuration abstractions:

```python
TeacherGroupConfig  ->  Defines HOW and WHERE teacher workers are created
StudentGroupConfig  ->  Defines HOW and WHERE student workers are created
```

**Rationale:**

1. **Separation of Concerns**: Model configuration (architecture, hyperparameters) is
   separate from deployment configuration (placement, resources).

2. **Explicit Resource Management**: Instead of implicit GPU allocation, you explicitly
   specify resources per worker and placement strategies.

3. **Composability**: The same `StudentGroupConfig` can be used with different trainers
   (single GPU, DDP, multi-student) without modification.

4. **Topology Flexibility**: By configuring placement strategies, you control whether
   workers are colocated (low latency) or distributed (fault tolerance).

### Why Placement Strategies?

Ray's placement groups give us fine-grained control over worker placement:

```
+------------------+     +------------------+     +------------------+
|   STRICT_PACK    |     |      PACK        |     |     SPREAD       |
+------------------+     +------------------+     +------------------+
| All workers MUST |     | Workers PREFER   |     | Workers spread   |
| be on same node  |     | same node, but   |     | across different |
|                  |     | can spread       |     | nodes            |
+------------------+     +------------------+     +------------------+
| Use case:        |     | Use case:        |     | Use case:        |
| - NVLink transfer|     | - Balance latency|     | - Fault tolerance|
| - Shared memory  |     |   and flexibility|     | - Max throughput |
+------------------+     +------------------+     +------------------+
```

**Benefits:**

- **STRICT_PACK for colocated training**: Teacher and student on same node enables
  fast tensor transfer via shared memory or NVLink.

- **PACK for teacher pools**: Keep teacher replicas together for efficient load balancing.

- **SPREAD for student DDP**: Distribute students for fault tolerance and to utilize
  full cluster bandwidth for gradient AllReduce.

### Why ActorPool for Load Balancing?

When you have multiple teacher replicas, you need to distribute inference requests:

```
                    +------------------------+
                    |      ActorPool         |
                    | (Round-robin dispatch) |
                    +------------------------+
                              |
            +-----------------+-----------------+
            |                 |                 |
            v                 v                 v
     +------------+    +------------+    +------------+
     | Teacher 0  |    | Teacher 1  |    | Teacher 2  |
     +------------+    +------------+    +------------+
```

**Benefits:**

- **Automatic load balancing**: Requests distributed evenly across replicas
- **Throughput matching**: Add more teacher replicas to match student training speed
- **No single point of failure**: If one replica is slow, others handle the load
- **Simple API**: `pool.submit()` and `pool.get_next()` handle all coordination

### Why `ray.get` for Data Transfer?

We use `ray.get()` for explicit data transfer between actors:

```python
# Teacher computes logits
teacher_logits_ref = teacher.get_logits.remote(data)

# Explicit transfer to driver
teacher_logits = ray.get(teacher_logits_ref)

# Pass to students
student.train_step.remote(data, target, teacher_logits)
```

**Benefits:**

- **Explicit control**: You know exactly when data moves between GPUs
- **Object store locality**: Ray optimizes transfer based on worker locations
- **Debugging**: Easy to inspect intermediate values
- **Flexibility**: Can broadcast same logits to multiple students

---

## Core Abstractions

### TeacherGroupConfig

```python
@dataclass
class TeacherGroupConfig:
    num_replicas: int = 1                    # Number of teacher copies
    resources_per_worker: Dict[str, float]   # {"GPU": 1, "CPU": 2}
    placement_strategy: str = "PACK"         # PACK, SPREAD, STRICT_PACK
    checkpoint_path: Optional[str] = None    # Pre-trained weights
    model_builder: Callable[[], nn.Module]   # Factory function
```

### StudentGroupConfig

```python
@dataclass
class StudentGroupConfig:
    num_workers: int = 1                     # Workers per student group
    resources_per_worker: Dict[str, float]   # {"GPU": 1, "CPU": 2}
    placement_strategy: str = "SPREAD"       # Placement strategy
    training_mode: str = "single"            # single, ddp, fsdp
    architecture: str = "small"              # Model architecture
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
```

### KDTrainerConfig

```python
@dataclass
class KDTrainerConfig:
    teacher_group: TeacherGroupConfig
    student_group: StudentGroupConfig
    distillation: DistillationConfig
    training: TrainingConfig
    colocate: bool = False  # Share placement group?
```

---

## Implemented Architectures

### 1. Single GPU Training (`single_gpu.py`)

Teacher and student colocated on the same GPU.

```
+----------------------------------+
|         GPU 0 (e.g., A100)       |
+----------------------------------+
|  +----------------------------+  |
|  | Teacher Model (Frozen)     |  |
|  | - Forward only, no grads   |  |
|  +----------------------------+  |
|              |                   |
|              v                   |
|  +----------------------------+  |
|  | Student Model (Trainable)  |  |
|  | - Forward + Backward       |  |
|  +----------------------------+  |
|              |                   |
|              v                   |
|  +----------------------------+  |
|  | Optimizer (AdamW)          |  |
|  +----------------------------+  |
+----------------------------------+
```

**Use case**: Quick experimentation, prototyping, single-GPU machines.

### 2. Multi-GPU DDP Training (`ddp_kd.py`)

Replicated teacher + DDP student with gradient AllReduce.

```
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
        +-----------> Gradient AllReduce <--------------+
                              |
                              v
                    Optimizer Update (All Ranks)
```

**Use case**: Scale training throughput, train larger students faster.

### 3. Multi-Student Architectures (`multi_student_kd.py`)

Single teacher, multiple heterogeneous students.

```
                     Shared Input Batch
                           |
                           v
                  +------------------+
                  |   GPU 0          |
                  |   Teacher Model  |
                  |   (Frozen)       |
                  +------------------+
                           |
                    features & logits (broadcast)
                           |
         +---------+-------+-------+---------+
         |         |               |         |
         v         v               v         v
   +---------+ +---------+   +---------+ +---------+
   |  GPU 1  | |  GPU 2  |   |  GPU 3  | |  GPU 4  |
   | Student | | Student |   | Student | | Student |
   |  small  | | medium  |   |  large  | |  wide   |
   +---------+ +---------+   +---------+ +---------+
        |          |              |          |
        v          v              v          v
   Optimizer  Optimizer     Optimizer  Optimizer
   (indep.)   (indep.)      (indep.)   (indep.)


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
```

**Use case**: Compare architectures, neural architecture search, deploy best student.

### [WIP] 4. Pooled Teacher and Student Pools (`pooled_kd.py`)

Dedicated GPU pools with load balancing.

```
    +--------------------------------------------------+
    |              Teacher GPU Pool                    |
    |  +----------+  +----------+  +----------+        |
    |  |  GPU 0   |  |  GPU 1   |     ...     |        |
    |  | Teacher  |  | Teacher  |  | Teacher  |        |
    |  | Replica 0|  | Replica 1|  | Replica K|        |
    |  | (Frozen) |  | (Frozen) |  | (Frozen) |        |
    |  +----------+  +----------+  +----------+        |
    +--------------------------------------------------+
                          |
                          v
              +------------------------+
              |    Load Balancer       |
              |     (ActorPool)        |
              +------------------------+
                          |
            +-------------+-------------+
            |      P2P transfer         |
            v             v             v
    +--------------------------------------------------+
    |              Student GPU Pool                    |
    |  +----------+  +----------+  +----------+        |
    |  | GPU M    |  | GPU M+1  |     ...     |        |
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
```

**Use case**: Production-scale training, throughput matching, resource isolation.


---

## Sample Usage

### Single GPU Training

```bash
# Train teacher first, then student
python single_gpu.py \
    --train-teacher \
    --epochs 10 \
    --student-arch small

# With existing teacher
python single_gpu.py \
    --teacher-checkpoint outputs/single_gpu/teacher.pt \
    --epochs 10
```

### Multi-GPU DDP Training

```bash
# 4 GPU DDP training
python ddp_kd.py \
    --num-workers 2 \
    --train-teacher \
    --epochs 10

# With existing teacher
python ddp_kd.py \
    --num-workers 2 \
    --teacher-checkpoint outputs/ddp_kd//teacher.pt \
    --epochs 10

# With SPREAD placement
python ddp_kd.py \
    --num-workers 2 \
    --placement SPREAD
```

### Multi-Student Training

```bash
# Train 3 different architectures
python multi_student_kd.py \
    --student-archs small medium \
    --train-teacher \
    --epochs 10

# Train 3 different architectures # With existing teacher
python multi_student_kd.py \
    --student-archs small medium \
    --teacher-checkpoint outputs/multi_student/teacher.pt \
    --epochs 10
```

### Pooled Training

```bash
# 2 teacher replicas + 6 student DDP workers
python pooled_kd.py \
    --teacher-replicas 2 \
    --student-workers 2 \
    --teacher-placement PACK \
    --student-placement SPREAD \
    --train-teacher

# With existing teacher
python pooled_kd.py \
    --teacher-replicas 2 \
    --student-workers 2 \
    --teacher-placement PACK \
    --student-placement SPREAD \
    --teacher-checkpoint outputs/pooled_kd//teacher.pt
```

---

## Sample Programmatic Usage

### Single GPU Colocated

```python
from src import (
    KDTrainerConfig,
    TeacherGroupConfig,
    StudentGroupConfig,
    DistillationConfig,
    TrainingConfig,
    SingleGPUKDTrainer,
    DDPKDTrainer,
)

# Single GPU
config = KDTrainerConfig(
    teacher_group=TeacherGroupConfig(
        num_replicas=1,
        checkpoint_path="teacher.pt",
    ),
    student_group=StudentGroupConfig(
        num_workers=1,
        training_mode="single",
        architecture="small",
    ),
    distillation=DistillationConfig(temperature=2.0, alpha=0.5),
    training=TrainingConfig(batch_size=64, max_epochs=10),
    colocate=True,
)
trainer = SingleGPUKDTrainer(config)
results = trainer.fit()
```
### 1 teacher, Strudent DDP

```
config = KDTrainerConfig(
    teacher_group=TeacherGroupConfig(
        num_replicas=4,
        placement_strategy="SPREAD",
    ),
    student_group=StudentGroupConfig(
        num_workers=4,
        training_mode="ddp",
        placement_strategy="SPREAD",
    ),
)
trainer = DDPKDTrainer(config)
results = trainer.fit()
```

### Multi-Student Configuration

```python
from src import MultiStudentKDConfig, MultiStudentKDTrainer

config = MultiStudentKDConfig(
    teacher_group=TeacherGroupConfig(num_replicas=1),
    student_groups=[
        StudentGroupConfig(architecture="small", learning_rate=1e-3),
        StudentGroupConfig(architecture="medium", learning_rate=1e-3),
        StudentGroupConfig(architecture="large", learning_rate=5e-4),
    ],
)
trainer = MultiStudentKDTrainer(config)
results = trainer.fit()
```

### Pooled Configuration

```python
from src import PooledKDConfig, PooledKDTrainer

config = PooledKDConfig(
    teacher_group=TeacherGroupConfig(
        num_replicas=2,
        placement_strategy="PACK",
    ),
    student_group=StudentGroupConfig(
        num_workers=6,
        training_mode="ddp",
        placement_strategy="SPREAD",
    ),
)
trainer = PooledKDTrainer(config)
results = trainer.fit()
```

---

## Architecture Deep Dive

### Data Flow Patterns

| Trainer | Teacher -> Student Data Flow |
|---------|------------------------------|
| SingleGPU | Direct tensor (same GPU, zero-copy) |
| DDP | Local replica (each worker has teacher copy) |
| MultiStudent | Broadcast (one teacher, multiple students via ray.get) |
| Pooled | Load-balanced + P2P (ActorPool dispatch, cross-GPU transfer) |

### Gradient Synchronization

| Trainer | Gradient Sync |
|---------|---------------|
| SingleGPU | None (single worker) |
| DDP | AllReduce across all student workers |
| MultiStudent | None (students train independently) |
| Pooled | AllReduce within student pool only |

### Placement Group Strategy

| Trainer | Teacher Placement | Student Placement |
|---------|-------------------|-------------------|
| SingleGPU | N/A (local) | N/A (local) |
| DDP | Same as student (colocated) | Configurable (PACK/SPREAD) |
| MultiStudent | Dedicated GPU | SPREAD (separate GPUs) |
| Pooled | PACK (teacher pool) | SPREAD (student pool) |

---

## When to Use Which Trainer

| Scenario | Recommended Trainer | Why |
|----------|---------------------|-----|
| Quick prototyping | `SingleGPUKDTrainer` | Simple, fast iteration |
| Scale training throughput | `DDPKDTrainer` | Parallel batch processing |
| Compare architectures | `MultiStudentKDTrainer` | Fair comparison, same teacher |
| Production with large teacher | `PooledKDTrainer` | Separate pools, load balancing |
| Memory-constrained | `PooledKDTrainer` | Teacher not on student GPUs |
| Fault tolerance needed | `DDPKDTrainer` + SPREAD | Workers on different nodes |
