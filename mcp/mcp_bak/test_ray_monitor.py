#!/usr/bin/env python3
"""
Test script for the Ray Job Monitor MCP Server
"""

import json
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the standalone analyzer
from ray_job_analyzer_standalone import RayLogAnalyzer

# Sample log content from your actual Ray job
sample_logs = """
INFO 10-29 21:46:42 [loggers.py:127] Engine 000: Avg prompt throughput: 2800.7 tokens/s, Avg generation throughput: 492.2 tokens/s, Running: 38 reqs, Waiting: 0 reqs, GPU KV cache usage: 4.6%, Prefix cache hit rate: 13.8%
INFO 10-29 21:46:42 [loggers.py:127] Engine 000: Avg prompt throughput: 3406.0 tokens/s, Avg generation throughput: 267.4 tokens/s, Running: 41 reqs, Waiting: 0 reqs, GPU KV cache usage: 4.9%, Prefix cache hit rate: 13.8%
Running Dataset: ... 73%|███...█| 52.0/71.0 [12:35<03:31, 11.1s/row]
MapBatches(vLLMEngineStageUDF): ... 74%|████...| 7.41k/10.1k [12:35<03:02, 14.5 row/s]
Resources: {CPU: 5.0/48.0, GPU: 4.0/4.0, memory: 32.0GB/256.0GB}
Demands: {CPU: 0.0, GPU: 0.0}
Total samples: 10000
Throughput: 32.45 samples/second
Pipeline completed successfully!
"""

def test_analyzer():
    """Test the Ray log analyzer"""
    analyzer = RayLogAnalyzer()
    
    # Process each line
    for line in sample_logs.strip().split('\n'):
        analyzer.analyze_log_line(line)
    
    # Get summary
    summary = analyzer.get_summary()
    
    print("=== Ray Job Monitor Test Results ===\n")
    
    # Job Summary
    print("Job Summary:")
    print(f"  Status: {summary['job_summary']['status']}")
    print(f"  Total Samples: {summary['job_summary']['total_samples']}")
    print(f"  Throughput: {summary['job_summary']['overall_throughput']} samples/sec")
    print()
    
    # Resource Status
    print("Resource Status:")
    rs = summary['resource_status']
    print(f"  CPU: {rs['cpus_used']}/{rs['cpus_total']} ({rs['cpus_used']/rs['cpus_total']*100:.1f}% used)")
    print(f"  GPU: {rs['gpus_used']}/{rs['gpus_total']} ({rs['gpus_used']/rs['gpus_total']*100:.1f}% used)")
    print(f"  Pending Demands: CPU={rs['pending_cpu_demand']}, GPU={rs['pending_gpu_demand']}")
    print()
    
    # Pipeline Stages
    print("Pipeline Progress:")
    for stage_name, stage in summary['pipeline_stages'].items():
        print(f"  {stage_name}:")
        print(f"    Progress: {stage['progress_percent']}%")
        print(f"    Items: {stage['processed_items']}/{stage['total_items']}")
        print(f"    Throughput: {stage['throughput']} {stage['throughput_unit']}")
    print()
    
    # vLLM Metrics
    print("vLLM Engine Metrics:")
    for engine_id, metrics in summary['vllm_metrics'].items():
        print(f"  Engine {engine_id}:")
        print(f"    Prompt Throughput: {metrics['prompt_throughput']} tokens/s")
        print(f"    Generation Throughput: {metrics['generation_throughput']} tokens/s")
        print(f"    Running Requests: {metrics['running_requests']}")
        print(f"    GPU KV Cache: {metrics['gpu_kv_cache_usage']}%")
    print()
    
    # Full JSON output
    print("\nFull JSON Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    test_analyzer()
