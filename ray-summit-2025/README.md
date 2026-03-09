# Ray Summit 2025

+ Developer Guide [developer.md](developer.md)  
+ vLLM Ray Metrics Integration Guide [vllm_ray_metrics_integration.md](vllm_ray_metrics_integration.md)
+ Understanding Performance [performance.md](performance.md)
+ Ray Summit 2025 Presentation [Slides](presentation/scaling-post-training-workflows-anindya-Saha-ray-summit-25.pptx)

---
**Title:** Scaling Post-Training Workflows with Ray Data, Ray Data LLM, and vLLM

**Abstract**  
Post-training workflows for vision-language models processing large-scale image datasets present coordination challenges among multiple processes and stages that often lead to complex, difficult-to-maintain systems. Inspired from real-world experience, we will talk and demo through a use case of image captioning to demonstrate how Ray Data, Ray Data LLM and vLLM simplify building production-scale post training workflows while achieving high throughput and resource efficiency.

![](images/image_with_captions_lite.png)

We'll explore practical patterns on:
- How do we develop a workflow from prototype to production scale gradually
- Use Ray Data's distributed processing for efficient data loading, batching and transformation
- Use Ray Data LLM's Processor abstraction for seamless vLLM integration with preprocessing and postprocessing steps
- Integrate vLLM and scaling up inference on multiple GPUs for high-throughput
- Fully customize preprocess, postprocess steps; Manage states with classes.
- Integrate Prometheus & Grafana for real time performance monitoring
- Optimize GPU resource utilization across distributed workloads

![](images/vllm_metrics.png)

**Target Audience**  
ML engineers building post-training pipelines, teams scaling language model workflows, and practitioners interested in production-ready distributed offine batch inference systems.

**Talk Outline (30 minutes)**  

**Problem Context (8 minutes)**  
+ Post training workflow; Relevant Use Cases
+ Common patterns: preprocess, postprocess, offline batch inferencing
+ Trade-offs between simplicity and scale in existing approaches

**Ray Data Approach (15 minutes)**  
+ Ray Data fundamentals for batch processing
+ Integration patterns with vLLM for language models
+ Code walkthrough: data loading, batching, inference, result handling, monitoring
+ Scaling up resources for each stage independently

**Production Considerations (5 minutes)**
+ Performance monitoring through Prometheus & Grafana dashboard; Also Ray dashboard
+ Performance characteristics and optimization techniques

**Q&A (2 minutes)**  
**Key Technical Points**
+ Practical Ray Data usage patterns for post training workloads
+ vLLM integration for efficient batch inference
+ Resource optimization techniques for GPU-intensive pipelines
+ State management without external coordination systems
