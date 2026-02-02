# LLM Evaluation at Scale with MLflow, Ray, and Kubernetes

A production-ready LLM evaluation framework that positions MLflow as a unified evaluation platform,
with massively parallel execution on Ray deployed to Amazon EKS.

---

## Data + AI Summit 2026 Proposal

### Title

**LLM Evaluation at Scale with Ray on Kubernetes and MLflow**

### Abstract

Evaluating LLMs at scale remains fragmented - teams rely on ad-hoc scripts, inconsistent metrics,
and disconnected evaluation platforms. This lightning talk demonstrates **MLflow** as a **unified LLM
evaluation system**, not just an experiment tracker.

We present a production-ready architecture running on Ray deployed on Kubernetes, enabling parallel LLM evaluation 
across thousands of prompts. The pipeline supports multiple judge configurations:
OpenAI GPT-4 as an external judge, or a self-hosted judge model served via vLLM on the same Ray cluster.

Using MLflow's `mlflow.genai.evaluate()` API with built-in and custom `Scorer` objects, we score
quality, safety, and task performance - logging all inputs, judge outputs, scores, and metadata
to MLflow for side-by-side comparison. Attendees will see how MLflow provides a **single pane of
glass** for LLM evaluation, replacing fragmented tooling with reproducible, auditable, and scalable benchmarking.

---

## Core Message

> "MLflow is not just an experiment tracker - it's a complete LLM evaluation platform"

---

## MLflow as the Comet ML Opik Equivalent

**Opik** is Comet's specialized LLM evaluation/observability platform. MLflow competes directly
with its new **GenAI evaluation system** (see [docs](https://mlflow.org/docs/latest/genai/eval-monitor/)):

| Opik Feature | MLflow GenAI Equivalent |
|--------------|-------------------------|
| LLM Tracing | MLflow Tracing - captures prompts, completions, latency per step |
| Evaluation Datasets | **Evaluation Datasets** - centralized test case repository |
| LLM-as-a-Judge Metrics | `mlflow.genai.evaluate()` with `Scorer` objects |
| Prompt Versioning | MLflow Prompt Management |
| Side-by-side Comparison | MLflow Evaluation UI - compare runs across experiments |
| Custom Scorers | `@scorer` decorator for custom evaluation logic |
| Human Feedback | Annotation and feedback attached to traces |

**The "sell" for MLflow**: Unlike fragmented tools (Opik, LangSmith, custom scripts), MLflow provides
a **unified platform** for the entire ML lifecycle - experiment tracking, model registry, AND LLM
evaluation in one place.

### Evaluation-Driven Development

MLflow promotes **Evaluation-Driven Development** - an emerging practice to tackle the challenge
of building high-quality LLM/Agentic applications. The three pillars are:

1. **Dataset Management** - Centralized repository for test cases and ground truth
2. **Human Feedback** - Collect and track annotations from users and domain experts
3. **Automated Evaluation** - Scale quality assessment with LLM-as-a-Judge scorers

---

## Architecture Overview

```
+------------------+     +---------------------------+     +------------------+
|   Eval Dataset   | --> |  Ray on EKS (Parallel     | --> |    MLflow        |
|  (prompts.jsonl) |     |  LLM-as-Judge Execution)  |     |  (Eval Tracker)  |
+------------------+     +---------------------------+     +------------------+
                                    |
                    +---------------+---------------+
                    |               |               |
              +-----------+   +-----------+   +-----------+
              | OpenAI    |   | Self-Host |   | lm-eval   |
              | GPT-4     |   | Judge LLM |   | harness   |
              | (Judge)   |   | (vLLM)    |   | (metrics) |
              +-----------+   +-----------+   +-----------+
```

### Component Breakdown

```
+-------------------------------------------------------------------+
|                           EKS Cluster                             |
+-------------------------------------------------------------------+
|                                                                   |
|  +-------------------------------------------------------------+  |
|  |                     Ray Cluster (KubeRay)                   |  |
|  +-------------------------------------------------------------+  |
|  |                                                             |  |
|  |  +------------------+    +-------------------------------+  |  |
|  |  |   Ray Head Node  |    |       Ray Worker Pool         |  |  |
|  |  |   - Job submit   |    |  +----------+ +----------+    |  |  |
|  |  |   - Dashboard    |    |  | Worker 0 | | Worker 1 |    |  |  |
|  |  +------------------+    |  | (eval)   | | (eval)   |    |  |  |
|  |                          |  +----------+ +----------+    |  |  |
|  |                          |  +----------+ +----------+    |  |  |
|  |                          |  | Worker 2 | | Worker N |    |  |  |
|  |                          |  | (eval)   | | (eval)   |    |  |  |
|  |                          |  +----------+ +----------+    |  |  |
|  |                          +-------------------------------+  |  |
|  |                                                             |  |
|  |  +-------------------------------+                          |  |
|  |  |  vLLM Judge Server (Optional) |                          |  |
|  |  |  - Self-hosted judge model    |                          |  |
|  |  |  - GPU-accelerated inference  |                          |  |
|  |  +-------------------------------+                          |  |
|  |                                                             |  |
|  +-------------------------------------------------------------+  |
|                                                                   |
|  +-----------------------------+                                  |
|  |    MLflow Tracking Server   |                                  |
|  |    - Experiments            |                                  |
|  |    - Runs with eval results |                                  |
|  |    - Artifacts (datasets)   |                                  |
|  +-----------------------------+                                  |
|                                                                   |
+-------------------------------------------------------------------+
```

---

## LLM-as-a-Judge Options

| Approach | Pros | Cons | Demo Complexity |
|----------|------|------|-----------------|
| **OpenAI GPT-4 as Judge** | Easy setup, high quality | Cost, API dependency | Low |
| **Self-hosted Judge (vLLM)** | No API costs, privacy | GPU infra needed | Medium |
| **lm-evaluation-harness** | Standard benchmarks, reproducible | Less flexible for custom evals | Medium |

### Recommended Approach

Lead with **OpenAI GPT-4 as Judge** for the demo (simple, relatable), but show **self-hosted judge
on Ray** as the production path. This demonstrates MLflow works with both approaches.

---

## Evaluation Dimensions

### Quality Metrics
- Relevance: Does the response address the prompt?
- Coherence: Is the response logically structured?
- Fluency: Is the language natural and grammatically correct?
- Completeness: Does the response fully answer the question?

### Safety Metrics
- Toxicity: Harmful or offensive content detection
- PII Leakage: Personal information exposure
- Bias: Unfair treatment of demographic groups
- Hallucination: Factually incorrect statements

### Task-Specific Metrics
- Code Correctness: For code generation tasks
- Summarization Faithfulness: For summarization tasks
- Translation Quality: For translation tasks
- Instruction Following: For instruction-tuned models

---

## Demo Flow (10-min Lightning Talk)

### 1. Problem Statement (1 min)
- Show fragmented eval landscape
- Custom scripts, LangSmith, Opik, etc.
- No unified tracking or comparison

### 2. Architecture Overview (2 min)
- Ray on EKS diagram
- MLflow as central hub
- Judge options

### 3. Live Demo (5 min)
```
Step 1: Show eval dataset in MLflow
        -> Versioned prompt dataset with expected outputs

Step 2: Kick off Ray job that fans out evaluation
        -> ray job submit --working-dir . -- python run_eval.py

Step 3: Show parallel GPT-4 judge calls
        -> Real-time progress in Ray Dashboard

Step 4: Results streaming into MLflow
        -> Metrics, traces, artifacts appearing in UI

Step 5: Side-by-side comparison
        -> Compare Model A vs Model B on same eval set
```

### 4. Key Takeaways (2 min)
- MLflow = complete LLM eval platform
- Ray = horizontal scale for parallel evaluation
- One system, not five fragmented tools

---

## MLflow GenAI Evaluation API

MLflow's **new GenAI evaluation system** uses `mlflow.genai.evaluate()` with `Scorer` objects.
This is separate from the classic `mlflow.evaluate()` API.

> **Note**: The GenAI system uses `mlflow.genai.evaluate()` and `Scorer` objects.
> The classic system uses `mlflow.evaluate()` and `EvaluationMetric`. They are not interoperable.

### Basic Evaluation Example

```python
import os
import openai
import mlflow
from mlflow.genai.scorers import Correctness, Guidelines

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 1. Define evaluation dataset with inputs and expectations
dataset = [
    {
        "inputs": {"question": "What is MLflow?"},
        "expectations": {"expected_response": "MLflow is an open-source platform for ML lifecycle management."},
    },
    {
        "inputs": {"question": "Can MLflow evaluate LLMs?"},
        "expectations": {"expected_response": "Yes, MLflow has a GenAI evaluation system."},
    },
]

# 2. Define a predict function to generate responses
def predict_fn(question: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}]
    )
    return response.choices[0].message.content

# 3. Run evaluation with built-in and custom scorers
results = mlflow.genai.evaluate(
    data=dataset,
    predict_fn=predict_fn,
    scorers=[
        # Built-in LLM-as-a-Judge scorer
        Correctness(),
        # Custom criteria using LLM judge
        Guidelines(name="is_english", guidelines="The answer must be in English"),
        Guidelines(name="is_concise", guidelines="The answer must be under 100 words"),
    ],
)
```

### Custom Scorer with @scorer Decorator

```python
from mlflow.genai.scorers import scorer

@scorer
def exact_match(expectations, outputs):
    """Check if the output exactly matches the expected response."""
    return expectations["expected_response"] == outputs

@scorer
def response_length(outputs):
    """Score based on response length (prefer concise answers)."""
    length = len(outputs.split())
    if length < 20:
        return {"score": 5, "rationale": "Very concise"}
    elif length < 50:
        return {"score": 4, "rationale": "Concise"}
    elif length < 100:
        return {"score": 3, "rationale": "Moderate length"}
    else:
        return {"score": 2, "rationale": "Too verbose"}

# Use custom scorers in evaluation
results = mlflow.genai.evaluate(
    data=dataset,
    predict_fn=predict_fn,
    scorers=[exact_match, response_length],
)
```

### Built-in Scorers

MLflow provides several built-in LLM-as-a-Judge scorers:

| Scorer | Description |
|--------|-------------|
| `Correctness()` | Evaluates if the response is correct given expectations |
| `Guidelines(name, guidelines)` | Custom criteria evaluation using LLM judge |
| `RelevanceToInput()` | Checks if response is relevant to the input |
| `Safety()` | Detects harmful, toxic, or unsafe content |

### Evaluation Datasets

```python
import mlflow

# Create a versioned evaluation dataset
eval_dataset = mlflow.data.from_dict(
    {
        "inputs": [
            {"question": "What is 2+2?"},
            {"question": "What is the capital of France?"},
        ],
        "expectations": [
            {"answer": "4"},
            {"answer": "Paris"},
        ],
    },
    name="math_and_geography_v1",
)

# Log dataset to MLflow for versioning
with mlflow.start_run():
    mlflow.log_input(eval_dataset, context="evaluation")
```

---

## Ray Parallel Evaluation

### Distributed Evaluation Pattern

```python
import ray
from ray.util import ActorPool

@ray.remote
class LLMJudge:
    def __init__(self, judge_model: str):
        self.judge_model = judge_model
        # Initialize judge (OpenAI client or vLLM)
    
    def evaluate(self, prompt: str, response: str, rubric: dict) -> dict:
        # Call judge model
        # Return structured scores
        pass

def run_parallel_evaluation(
    eval_data: list[dict],
    num_judges: int = 10,
    judge_model: str = "gpt-4"
) -> list[dict]:
    """Fan out evaluation across Ray workers."""
    
    # Create pool of judge actors
    judges = [LLMJudge.remote(judge_model) for _ in range(num_judges)]
    pool = ActorPool(judges)
    
    # Submit all evaluation tasks
    results = list(pool.map(
        lambda judge, item: judge.evaluate.remote(
            item["prompt"],
            item["response"],
            item["rubric"]
        ),
        eval_data
    ))
    
    return results
```

### Self-Hosted Judge with vLLM

```python
@ray.remote(num_gpus=1)
class VLLMJudge:
    def __init__(self, model_name: str = "meta-llama/Llama-3-8B-Instruct"):
        from vllm import LLM
        self.llm = LLM(model=model_name)
    
    def evaluate(self, prompt: str, response: str, rubric: dict) -> dict:
        judge_prompt = self._build_judge_prompt(prompt, response, rubric)
        outputs = self.llm.generate([judge_prompt])
        return self._parse_judge_output(outputs[0])
```

---

## Project Structure

```
llm-evaluation-mlflow/
|-- README.md
|-- pyproject.toml
|-- src/
|   |-- __init__.py
|   |-- config.py              # Configuration dataclasses
|   |-- datasets.py            # Eval dataset loading/versioning
|   |-- judges/
|   |   |-- __init__.py
|   |   |-- base.py            # Abstract Judge interface
|   |   |-- openai_judge.py    # OpenAI GPT-4 judge
|   |   |-- vllm_judge.py      # Self-hosted vLLM judge
|   |-- metrics/
|   |   |-- __init__.py
|   |   |-- quality.py         # Quality metrics
|   |   |-- safety.py          # Safety metrics
|   |   |-- task_specific.py   # Task-specific metrics
|   |-- evaluator.py           # Main evaluation orchestrator
|   |-- mlflow_logger.py       # MLflow integration
|   |-- ray_executor.py        # Ray parallel execution
|-- scripts/
|   |-- run_eval.py            # CLI entrypoint
|   |-- submit_ray_job.sh      # Ray job submission
|-- k8s/
|   |-- raycluster.yaml        # KubeRay cluster config
|   |-- mlflow-server.yaml     # MLflow tracking server
|   |-- vllm-judge.yaml        # Optional vLLM judge deployment
|-- examples/
|   |-- basic_eval.py          # Simple evaluation example
|   |-- parallel_eval.py       # Parallel evaluation example
|   |-- compare_models.py      # Model comparison example
|-- tests/
|   |-- test_judges.py
|   |-- test_metrics.py
|   |-- test_evaluator.py
```

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Eval Tracker** | MLflow | Unified platform, open source, Databricks ecosystem |
| **Eval API** | `mlflow.genai.evaluate()` | Modern GenAI system with Scorer objects |
| **Parallel Execution** | Ray | Scales horizontally, ActorPool for load balancing |
| **Infrastructure** | EKS + KubeRay | Production-grade, autoscaling, GPU support |
| **Primary Judge** | OpenAI GPT-4 | High quality, easy demo, relatable to audience |
| **Production Judge** | vLLM self-hosted | Cost-effective, privacy, no API dependency |
| **Scorers** | Built-in + Custom `@scorer` | `Correctness()`, `Guidelines()`, custom logic |

---

## Key Differentiators vs Alternatives

| Feature | MLflow | LangSmith | Opik | Custom Scripts |
|---------|--------|-----------|------|----------------|
| Open Source | Yes | No | Yes | N/A |
| Unified ML Platform | Yes | No (LLM only) | No (LLM only) | No |
| Model Registry | Yes | No | No | No |
| Experiment Comparison | Yes | Yes | Yes | Manual |
| Custom Metrics | Yes | Yes | Yes | Yes |
| Ray Integration | Native | Plugin | No | Manual |
| Databricks Integration | Native | No | No | No |

---

## Implementation Roadmap

### Phase 1: Core Framework
- [ ] Project scaffolding and config
- [ ] Abstract Judge interface
- [ ] OpenAI GPT-4 judge implementation
- [ ] Basic MLflow logging

### Phase 2: Ray Integration
- [ ] Ray remote judge actors
- [ ] ActorPool for parallel evaluation
- [ ] Ray job submission scripts
- [ ] KubeRay cluster configuration

### Phase 3: Advanced Features
- [ ] vLLM self-hosted judge
- [ ] Multiple evaluation metrics (quality, safety)
- [ ] MLflow experiment comparison UI
- [ ] Evaluation dataset versioning

### Phase 4: Production Readiness
- [ ] EKS deployment manifests
- [ ] MLflow tracking server deployment
- [ ] Autoscaling configuration
- [ ] Monitoring and alerting

### Phase 5: Demo Preparation
- [ ] Demo script and data
- [ ] Slide deck
- [ ] Dry run and timing

---

## References

### MLflow GenAI Documentation
- [MLflow GenAI Evaluation Overview](https://mlflow.org/docs/latest/genai/eval-monitor/) - Main evaluation docs
- [Running Evaluations](https://mlflow.org/docs/latest/genai/eval-monitor/running-evaluations/) - How to run evals
- [Scorers](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/) - Built-in and custom scorers
- [Evaluation Datasets](https://mlflow.org/docs/latest/genai/eval-monitor/evaluation-datasets/) - Dataset management
- [MLflow Tracing](https://mlflow.org/docs/latest/genai/tracing/) - Observability for LLM apps

### Ray and Infrastructure
- [Ray ActorPool](https://docs.ray.io/en/latest/ray-core/actors/actor-utils.html)
- [KubeRay on EKS](https://docs.ray.io/en/latest/cluster/kubernetes/index.html)
- [vLLM](https://docs.vllm.ai/en/latest/)

### Benchmarking
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
