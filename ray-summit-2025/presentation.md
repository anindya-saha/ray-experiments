
Understanding blocks and parallelism:
# Example: 1000 images dataset
dataset = ray.data.from_huggingface(...)  # 1000 rows

# Option 1: Default partitioning (auto)
# Ray might create ~8 blocks of ~125 rows each

# Option 2: Manual partitioning
dataset = dataset.repartition(num_blocks=10)
# Creates exactly 10 blocks of 100 rows each

# During processing:
# - 10 workers can process blocks in parallel
# - Each worker calls caption_preprocess 100 times (once per row)
# - Workers operate independently


Visualization of the process:
```bash
Dataset (1000 images)
    ↓ repartition(10)
[Block0: 100 rows] [Block1: 100 rows] ... [Block9: 100 rows]
    ↓ map(caption_preprocess) - parallel execution
[Worker0]         [Worker1]         ... [Worker9]
 ↓ ↓ ↓            ↓ ↓ ↓                ↓ ↓ ↓
row→preprocess   row→preprocess      row→preprocess
row→preprocess   row→preprocess      row→preprocess
...              ...                  ...
    ↓ Results collected and batched for vLLM
[Preprocessed Block0] [Preprocessed Block1] ... [Preprocessed Block9]
    ↓ vLLM processor batches these for inference
```

Best practices for partitioning:
# 1. For CPU-bound preprocessing (like image resizing)
num_blocks = num_cpus * 2  # Oversubscribe for better utilization
dataset = dataset.repartition(num_blocks)

# 2. For I/O-bound preprocessing (like downloading)
num_blocks = num_cpus * 4  # More blocks to hide I/O latency
dataset = dataset.repartition(num_blocks)

# 3. For memory-intensive preprocessing
num_blocks = available_memory / max_block_size
dataset = dataset.repartition(num_blocks)

# 4. Let Ray decide (often good default)
# Don't repartition - Ray will use heuristics

The key insight: Ray Data handles the parallelization automatically. Your caption_preprocess function just needs to handle a single row, and Ray will efficiently distribute the work across available resources!

sample_batch["image"][0].keys()
dict_keys(['bytes', 'path'])

Now your pipeline should work smoothly with the ImageNet dataset! The flow is:
Load dataset → images stored as {'bytes': ..., 'path': ...}
Preprocess → convert bytes to PIL Image
vLLM → generates captions
Postprocess → extract and format results





uv venv venv --python 3.12
source venv/bin/activate
uv pip install pip

python3.12 -m venv venv
source venv/bin/activate

uv pip install torch torchvision torchaudio torchmetrics --index-url https://download.pytorch.org/whl/cu128


>>> 
>>> ray_dataset.count()
1281167

Adding prometheus and grafana
https://docs.ray.io/en/latest/cluster/metrics.html#grafana

for prometheus
ray metrics launch-prometheus
ray metrics shutdown-prometheus

for grafana
./bin/grafana-server --config /tmp/ray/session_latest/metrics/grafana/grafana.ini web

vLLM issue with inputs
https://docs.vllm.ai/en/latest/features/multimodal_inputs.html#stable-uuids-for-caching-multi_modal_uuids
https://docs.vllm.ai/en/latest/features/multimodal_inputs.html#audio-inputs_1



```bash
01
======================================================================
PIPELINE RESULTS
======================================================================
Total samples: 1000
Total time: 342.32 seconds
Inference time: 0.01 seconds
Throughput: 146234.71 samples/second
Output saved to: outputs/01_captioned_dataset.parquet
(ray-summit-2025) (git: main) asaha@host ~/ray-summit-2025 $ 


>>> ray_dataset.num_blocks()
1



02
======================================================================
PIPELINE RESULTS
======================================================================
Total samples: 1000
Total time: 302.34 seconds
Inference time: 0.01 seconds
Throughput: 159776.92 samples/second
Output saved to: outputs/02_captioned_dataset.parquet
```


## Concerns with `qwen_vl_utils`

```bash
python 04_image_caption_demo.py
```

Main error is : 

```bash
image_grid_thw[image_index][0]
IndexError: list index out of range
```

Full error log:
```bash
File "/home/asaha/ray-summit-2025/.venv/lib/python3.12/site-packages/vllm/executor/ray_utils.py", line 135, in execute_model_ray
[36m(MapWorker(MapBatches(vLLMEngineStageUDF)) pid=674640)[0m ERROR 10-19 12:18:50 [core.py:588]     output = self.worker.model_runner.execute_model(
[36m(MapWorker(MapBatches(vLLMEngineStageUDF)) pid=674640)[0m ERROR 10-19 12:18:50 [core.py:588]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[36m(MapWorker(MapBatches(vLLMEngineStageUDF)) pid=674640)[0m ERROR 10-19 12:18:50 [core.py:588]   File "/home/asaha/ray-summit-2025/.venv/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 116, in decorate_context
[36m(MapWorker(MapBatches(vLLMEngineStageUDF)) pid=674640)[0m ERROR 10-19 12:18:50 [core.py:588]     return func(*args, **kwargs)
[36m(MapWorker(MapBatches(vLLMEngineStageUDF)) pid=674640)[0m ERROR 10-19 12:18:50 [core.py:588]            ^^^^^^^^^^^^^^^^^^^^^
[36m(MapWorker(MapBatches(vLLMEngineStageUDF)) pid=674640)[0m ERROR 10-19 12:18:50 [core.py:588]   File "/home/asaha/ray-summit-2025/.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py", line 1283, in execute_model
[36m(MapWorker(MapBatches(vLLMEngineStageUDF)) pid=674640)[0m ERROR 10-19 12:18:50 [core.py:588]     self._update_states(scheduler_output)
[36m(MapWorker(MapBatches(vLLMEngineStageUDF)) pid=674640)[0m ERROR 10-19 12:18:50 [core.py:588]   File "/home/asaha/ray-summit-2025/.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py", line 456, in _update_states
[36m(MapWorker(MapBatches(vLLMEngineStageUDF)) pid=674640)[0m ERROR 10-19 12:18:50 [core.py:588]     MRotaryEmbedding.get_input_positions_tensor(
[36m(MapWorker(MapBatches(vLLMEngineStageUDF)) pid=674640)[0m ERROR 10-19 12:18:50 [core.py:588]   File "/home/asaha/ray-summit-2025/.venv/lib/python3.12/site-packages/vllm/model_executor/layers/rotary_embedding.py", line 1167, in get_input_positions_tensor
[36m(MapWorker(MapBatches(vLLMEngineStageUDF)) pid=674640)[0m ERROR 10-19 12:18:50 [core.py:588]     return cls._vl_get_input_positions_tensor(
[36m(MapWorker(MapBatches(vLLMEngineStageUDF)) pid=674640)[0m ERROR 10-19 12:18:50 [core.py:588]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[36m(MapWorker(MapBatches(vLLMEngineStageUDF)) pid=674640)[0m ERROR 10-19 12:18:50 [core.py:588]   File "/home/asaha/ray-summit-2025/.venv/lib/python3.12/site-packages/vllm/model_executor/layers/rotary_embedding.py", line 1330, in _vl_get_input_positions_tensor
[36m(MapWorker(MapBatches(vLLMEngineStageUDF)) pid=674640)[0m ERROR 10-19 12:18:50 [core.py:588]     image_grid_thw[image_index][0],
[36m(MapWorker(MapBatches(vLLMEngineStageUDF)) pid=674640)[0m ERROR 10-19 12:18:50 [core.py:588]     ~~~~~~~~~~~~~~^^^^^^^^^^^^^
[36m(MapWorker(MapBatches(vLLMEngineStageUDF)) pid=674640)[0m ERROR 10-19 12:18:50 [core.py:588] IndexError: list index out of range
[36m(MapWorker(MapBatches(vLLMEngineStageUDF)) pid=674640)[0m INFO 10-19 12:18:50 [ray_distributed_executor.py:128] Shutting down Ray distributed executor. If you see error log from logging.cc regarding SIGTERM received, please ignore because this is the expected termination process in Ray.
[36m(MapWorker(MapBatches(vLLMEngineStageUDF)) pid=674640)[0m ERROR 10-19 12:18:50 [async_llm.py:419] AsyncLLM output_handler failed.
```

Map Batches
```bash
sample_batch = ray_dataset.take_batch(batch_size=2)

 {
   "image":"array("[
      {
         "bytes":"b""\\xff\\xd8\\xff\\xe0\\x00\\x10JFIF\\x00\\x01\\x01...c\\xabQK_\\xff\\xd9",
         "path":"n03954731_53652_n03954731.JPEG"
      },
      {
         "bytes":"b""\\xff\\xd8\\xff\\xe0\\x00\\x10JFIF\\x00\\x01\\x01...c\\nZxc81\\xff\\xd9",
         "path":"n06596364_8704_n06596364.JPEG"
      }
   ],
   "dtype=object)",
   "label":"array("[
      726,
      917
   ]")"
}
 ```

 ```bash
 sample_batch.keys()
 dict_keys(['image', 'label'])

 ```bash
sample_batch['image'].shape
(2,)
 ```

Outline
# Scaling Image Captioning Workflows with Ray Data, Ray Data LLM and vLLM

## Talk Outline (30)

### I. Introduction (5 min)
- The Challenge: Processing large-scale image datasets for captioning
- Traditional approaches: Complex actor coordination, difficult-to-maintain systems
- The Solution: Ray Data + Ray Data LLM + vLLM

### II. Core Components (10 min)
1. Ray Data: Distributed data processing framework
2. Ray Data LLM: Batch inference abstraction
3. vLLM: High-performance model serving
4. How they work together

### III. Building the Pipeline (15 min)
1. Data Loading & Processing with Ray Data
2. Implementing Ray Data LLM Processor
3. Integrating vLLM for Vision-Language Models
4. Fault Tolerance & Checkpointing
5. Resource Optimization

### IV. Real-World Patterns & Best Practices (5 min)
- Batching strategies
- GPU utilization optimization
- State management
- Scaling considerations

### V. Demo & Results (5 min)
- Live demo or recorded demo
- Performance metrics
- Lessons learned

### VI. Q&A (5 min)


# Slide Deck Outline

## Slide 1: Title
- Title: Scaling Image Captioning Workflows with Ray Data, Ray Data LLM and vLLM
- Your name, title, company
- Ray Summit 2025

## Slide 2: The Problem
- Our USe Case
- Multiple checkpints are created. Kick off validation for each checkpoint - Draw a Picture
- Take all checkpoints and kick of validation - Draw Picture
- We stick to the first use case for now
- Going forward kick off asynchronous checkpoint validation
- Traditional approaches to large-scale image processing
- Complex actor coordination
- Difficult to maintain and scale
- Poor resource utilization

## Slide 3: The Solution Stack
- Ray Data: Distributed data processing
- Ray Data LLM: Batch inference abstraction
- vLLM: High-performance model serving
- Visual diagram of how they connect - Check the Diagram from End to End LLM Workflow
- Also talk about seggrgation of stages - when the dataset do not change or golden dataset them no need to plugin

## Slide 4: Why This Matters
- Process thousands/millions of images efficiently
- Simple, maintainable code
- Automatic fault tolerance
- Optimal GPU utilization

## Slides 5-7: Ray Data Fundamentals
- Loading data from various sources
- map and map_batches operations
- Automatic parallelization
- Code example

## Slides 8-10: Ray Data LLM
- Processor abstraction
- Preprocess → Inference → Postprocess pipeline
- Built-in batching and resource management
- Code example

## Slides 11-13: vLLM Integration
- Why vLLM for vision models
- Configuration options
- Performance optimizations
- Code example

## Slides 14-16: Production Patterns
- Fault tolerance and checkpointing
- Resource optimization strategies
- Monitoring and debugging
- Real-world considerations

## Slides 16: When you do not want Ray Data LLM
- Talk about the qwen_vl_utils

## Slides 17-18: Performance Results
- Benchmarks: throughput, latency
- Resource utilization graphs
- Comparison with traditional approaches

## Slide 19: Demo
- Live or recorded demo
- Show the pipeline in action

## Slide 20: Key Takeaways
- Simplicity over complexity
- Let Ray handle the hard parts
- Focus on your ML logic
- Production-ready from day one

## Slide 21: Resources & Q&A
- GitHub repo link
- Documentation links
- Contact information

Additional Things to do
Benchamrk
Skip faulty images - Can be done on the fly or prior
Add custom evaluadtion - like LLM based evaluation or caption lengeh based evaluation
Dr

6. Common Questions & Answers
Prepare for these likely questions:
Q: How does this compare to using pure PyTorch DataLoader?
A: Ray Data provides automatic distributed processing, fault tolerance, and better GPU utilization through dynamic batching.
Q: What's the overhead of using Ray Data LLM?
A: Minimal - typically < 5% compared to direct vLLM usage, but with significant benefits in ease of use and fault tolerance.
Q: Can this work with custom models?
A: Yes, any model supported by vLLM can be used, including fine-tuned models.
Q: How do you handle very large images?
A: Ray Data supports streaming and can handle images that don't fit in memory through chunking.
Q: What about multi-modal outputs (e.g., captions + object detection)?
A: The postprocess function can return multiple fields, making multi-modal outputs straightforward.
Would you like me to help you create any specific component in more detail, such as the actual presentation slides, more comprehensive demo scripts, or deployment configurations?


export HF_TOKEN=
export HF_HOME=/mnt/data/hf_cache