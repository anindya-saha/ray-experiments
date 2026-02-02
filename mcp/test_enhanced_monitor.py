#!/usr/bin/env python3
"""
Test script for the Enhanced Ray Job Monitor
Demonstrates both REST API and log analysis approaches
"""

import asyncio
import json
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ray_job_analyzer_standalone import RayLogAnalyzer


# Sample log content from actual Ray job
SAMPLE_LOG_CONTENT = """
INFO 10-29 21:46:42 [loggers.py:127] Engine 000: Avg prompt throughput: 2800.7 tokens/s, Avg generation throughput: 492.2 tokens/s, Running: 38 reqs, Waiting: 0 reqs, GPU KV cache usage: 4.6%, Prefix cache hit rate: 13.8%
INFO 10-29 21:46:42 [loggers.py:127] Engine 001: Avg prompt throughput: 3100.5 tokens/s, Avg generation throughput: 489.3 tokens/s, Running: 42 reqs, Waiting: 0 reqs, GPU KV cache usage: 5.2%, Prefix cache hit rate: 14.2%
Running Dataset: ... 73%|███...█| 52.0/71.0 [12:35<03:31, 11.1s/row]
MapBatches(vLLMEngineStageUDF): ... 74%|████...| 7.41k/10.1k [12:35<03:02, 14.5 row/s]
Resources: {CPU: 5.0/48.0, GPU: 4.0/4.0, memory: 32.0GB/256.0GB}
Demands: {CPU: 0.0, GPU: 0.0}
Map(ImageCaptionPipelineV3.caption_preprocess): ... 82%|████████| 8.28k/10.1k [00:52<00:11, 158 row/s]
Total samples: 10000
Throughput: 32.45 samples/second
Pipeline completed successfully!
"""


def test_log_analysis():
    """Test the log analysis approach (for completed jobs)"""
    print("="*60)
    print("TEST 1: Log Analysis Approach")
    print("="*60)
    print("\nAnalyzing saved logs from a completed job...\n")
    
    analyzer = RayLogAnalyzer()
    
    # Process each line
    for line in SAMPLE_LOG_CONTENT.strip().split('\n'):
        analyzer.analyze_log_line(line)
    
    # Get summary
    summary = analyzer.get_summary()
    
    # Display results
    print("Job Summary:")
    print(f"  Status: {summary['job_summary']['status']}")
    print(f"  Total Samples: {summary['job_summary']['total_samples']}")
    print(f"  Throughput: {summary['job_summary']['overall_throughput']} samples/sec")
    print()
    
    print("Resource Status:")
    rs = summary['resource_status']
    print(f"  CPU: {rs['cpus_used']}/{rs['cpus_total']} ({rs['cpus_used']/rs['cpus_total']*100:.1f}% used)")
    print(f"  GPU: {rs['gpus_used']}/{rs['gpus_total']} ({rs['gpus_used']/rs['gpus_total']*100:.1f}% used)")
    print()
    
    print("Pipeline Progress:")
    for stage_name, stage in summary['pipeline_stages'].items():
        print(f"  {stage_name}: {stage['progress_percent']}% ({stage['processed_items']}/{stage['total_items']})")
    print()
    
    print("vLLM Metrics:")
    for engine_id, metrics in summary['vllm_metrics'].items():
        print(f"  Engine {engine_id}:")
        print(f"    Prompt: {metrics['prompt_throughput']} tokens/s")
        print(f"    Generation: {metrics['generation_throughput']} tokens/s")
    
    return summary


async def test_api_monitoring():
    """Test the REST API approach (for running jobs)"""
    print("\n" + "="*60)
    print("TEST 2: REST API Approach (Simulation)")
    print("="*60)
    print("\nSimulating API responses for a running job...\n")
    
    # Simulate API response for job list
    mock_jobs = [
        {
            "job_id": "raysubmit_abc123",
            "status": "RUNNING",
            "entrypoint": "python image_caption_pipeline.py",
            "start_time": 1698620000,
            "metadata": {"name": "image-captioning", "dataset": "coco"}
        },
        {
            "job_id": "raysubmit_xyz789",
            "status": "SUCCEEDED",
            "entrypoint": "python train_model.py",
            "start_time": 1698610000,
            "end_time": 1698615000,
            "metadata": {"name": "model-training"}
        }
    ]
    
    print("Jobs List:")
    for job in mock_jobs:
        print(f"  • {job['job_id']} - {job['status']} - {job['entrypoint']}")
    print()
    
    # Simulate detailed job info
    job_detail = {
        "job_id": "raysubmit_abc123",
        "status": "RUNNING",
        "entrypoint": "python image_caption_pipeline.py --model llava --batch-size 32",
        "start_time": 1698620000,
        "metadata": {
            "name": "image-captioning",
            "dataset": "coco",
            "model": "llava-v1.5-7b",
            "num_images": 10000
        },
        "runtime_env": {
            "pip": ["torch", "transformers", "pillow"]
        }
    }
    
    print("Detailed Job Info:")
    print(f"  Job ID: {job_detail['job_id']}")
    print(f"  Status: {job_detail['status']}")
    print(f"  Command: {job_detail['entrypoint']}")
    print(f"  Metadata: {json.dumps(job_detail['metadata'], indent=4)}")
    
    # Simulate combining with log analysis
    print("\nCombining API data with log analysis...")
    print("This would fetch logs via API and analyze them for detailed metrics")
    
    return job_detail


def test_mcp_tool_examples():
    """Show example MCP tool usage"""
    print("\n" + "="*60)
    print("TEST 3: MCP Tool Usage Examples")
    print("="*60)
    print("\nExample commands for Claude Desktop or MCP clients:\n")
    
    examples = [
        {
            "scenario": "Monitor all running jobs",
            "command": 'Use list_ray_jobs tool with status_filter "RUNNING"'
        },
        {
            "scenario": "Get details of a specific job",
            "command": 'Use monitor_ray_job tool with job_id "raysubmit_abc123"'
        },
        {
            "scenario": "Analyze saved logs",
            "command": 'Use analyze_ray_logs tool with log_content "<paste logs here>"'
        },
        {
            "scenario": "Check resource bottlenecks",
            "command": 'Use check_resource_allocation tool'
        },
        {
            "scenario": "Get vLLM performance metrics",
            "command": 'Use get_vllm_metrics tool'
        },
        {
            "scenario": "Stop a running job",
            "command": 'Use stop_ray_job tool with job_id "raysubmit_abc123"'
        }
    ]
    
    for example in examples:
        print(f"Scenario: {example['scenario']}")
        print(f"Command:  {example['command']}")
        print()


def test_cli_examples():
    """Show CLI usage examples"""
    print("\n" + "="*60)
    print("TEST 4: CLI Usage Examples")
    print("="*60)
    print("\nCommand-line usage examples:\n")
    
    cli_examples = [
        {
            "description": "Monitor a running job",
            "command": "python analyze_ray_logs_enhanced.py --api --job-id raysubmit_abc123"
        },
        {
            "description": "List all failed jobs",
            "command": "python analyze_ray_logs_enhanced.py --api --status failed"
        },
        {
            "description": "Analyze a log file",
            "command": "python analyze_ray_logs_enhanced.py --file /path/to/ray_job.log"
        },
        {
            "description": "Stream logs in real-time",
            "command": "tail -f ray_job.log | python analyze_ray_logs_enhanced.py"
        },
        {
            "description": "Get JSON output for scripting",
            "command": "python analyze_ray_logs_enhanced.py --api --job-id raysubmit_abc123 --json"
        },
        {
            "description": "Monitor with custom dashboard URL",
            "command": "python analyze_ray_logs_enhanced.py --api --dashboard-url http://ray-head:8265"
        }
    ]
    
    for example in cli_examples:
        print(f"# {example['description']}")
        print(f"$ {example['command']}")
        print()


async def main():
    """Run all tests"""
    print("Enhanced Ray Job Monitor - Test Suite")
    print("====================================\n")
    
    # Test 1: Log analysis
    log_summary = test_log_analysis()
    
    # Test 2: API monitoring (simulated)
    job_detail = await test_api_monitoring()
    
    # Test 3: MCP tool examples
    test_mcp_tool_examples()
    
    # Test 4: CLI examples
    test_cli_examples()
    
    # Summary
    print("="*60)
    print("TEST SUMMARY")
    print("="*60)
    print("\nThe Enhanced Ray Job Monitor provides:")
    print("✓ REST API integration for live job monitoring")
    print("✓ Log analysis for completed jobs and offline analysis")
    print("✓ Unified MCP server with tools for both approaches")
    print("✓ Flexible CLI tool supporting multiple modes")
    print("✓ Comprehensive metrics: resources, pipeline progress, vLLM performance")
    print("\nChoose the right approach based on your needs:")
    print("• Use REST API for monitoring running jobs")
    print("• Use log analysis for post-mortem analysis or when API is unavailable")


if __name__ == "__main__":
    asyncio.run(main())
