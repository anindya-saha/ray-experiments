# Ray Job Monitor - MCP Server & CLI Tool

I've created a comprehensive Ray job monitoring solution with two components:

## 1. MCP Server (`ray_job_monitor_mcp.py`)

An MCP (Model Context Protocol) server that provides intelligent log analysis through natural language interfaces. This allows you to:

- Feed console logs and get structured insights
- Query job status, resource allocation, and performance metrics
- Track pipeline progress and vLLM engine metrics
- Detect errors, warnings, and resource bottlenecks

### Key Features:
- **Resource Monitoring**: Track CPU/GPU usage and pending demands
- **Pipeline Progress**: Monitor each Ray Data stage with throughput metrics
- **vLLM Metrics**: Detailed engine performance (tokens/s, cache usage, queue depth)
- **Intelligent Analysis**: Automatically detects patterns in logs

### Available MCP Tools:
1. `analyze_ray_logs`: Process log content
2. `get_job_status`: Overall job summary
3. `check_resource_allocation`: Resource usage and bottlenecks
4. `get_pipeline_progress`: Stage-by-stage progress
5. `get_vllm_metrics`: vLLM engine performance

## 2. CLI Tool (`analyze_ray_logs.py`)

A standalone command-line tool for real-time log monitoring:

```bash
# Analyze a log file
python analyze_ray_logs.py --file ray_job.log

# Stream logs in real-time
tail -f ray_job.log | python analyze_ray_logs.py

# Output JSON format
python analyze_ray_logs.py --json < ray_job.log
```

### Features:
- **Live Dashboard**: Clear, formatted console output
- **Progress Bars**: Visual representation of pipeline progress
- **Color Coding**: Status indicators (green=success, yellow=warning, red=error)
- **Streaming Mode**: Real-time updates as logs are generated

## 3. Standalone Analyzer (`ray_job_analyzer_standalone.py`)

The core analysis logic without dependencies, perfect for integration into other tools.

## What It Monitors

### Resource Allocation
- CPU and GPU usage vs. availability
- Pending resource demands
- Actor and task queues

### Pipeline Progress
- Stage completion percentage
- Items processed vs. total
- Throughput (items/sec or sec/item)

### vLLM Performance
- Prompt and generation throughput
- Request queue depth
- GPU KV cache utilization
- Prefix cache hit rates

### Job Health
- Overall status (running/completed/failed)
- Error and warning counts
- Total samples processed
- End-to-end throughput

## Integration with Your Pipeline

The monitor is designed to work seamlessly with your image captioning pipelines:

1. **During Development**: Stream logs to the CLI tool for real-time monitoring
2. **In Production**: Use the MCP server for intelligent querying
3. **Post-Analysis**: Analyze saved logs to understand performance bottlenecks

## Example Output

```
================================================================================
                                RAY JOB MONITOR                                 
================================================================================

Job Status: RUNNING
Total Samples: 10,000
Overall Throughput: 32.45 samples/sec

RESOURCE ALLOCATION:
----------------------------------------
CPU:    5.0/48    [██░░░░░░░░░░░░░░░░░░]  10.4%
GPU:    4.0/4     [████████████████████] 100.0%

PIPELINE PROGRESS:
----------------------------------------
Running Dataset:
  [██████████████░░░░░░]  73.0%
  Items: 52/71
  Speed: 11.10 s/row

vLLM ENGINE METRICS:
----------------------------------------
Engine 000:
  Prompt:       2800.7 tokens/s
  Generation:    492.2 tokens/s
  Requests:   38 running, 0 waiting
  GPU Cache:  4.6% used
  Cache Hits: 13.8%
```

This monitoring solution provides the visibility you need to:
- Identify performance bottlenecks
- Ensure resource utilization is optimal
- Track job progress in real-time
- Debug issues quickly with structured log analysis
