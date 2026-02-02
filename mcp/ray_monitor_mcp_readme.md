# Ray Job Monitor MCP Server

This MCP (Model Context Protocol) server monitors Ray job logs and provides real-time insights about job execution, resource allocation, and performance.

## Features

- **Resource Monitoring**: Track CPU/GPU allocation and pending demands
- **Pipeline Progress**: Monitor each stage of your Ray Data pipeline
- **vLLM Metrics**: Track vLLM engine performance (throughput, cache usage, etc.)
- **Error Detection**: Automatically identify errors and warnings
- **Job Status**: Overall job health and completion status

## Installation

1. Install the MCP SDK:
```bash
pip install mcp
```

2. Make the script executable:
```bash
chmod +x ray_job_monitor_mcp.py
```

## Usage

### Starting the MCP Server

```bash
python ray_job_monitor_mcp.py
```

### Available Tools

1. **analyze_ray_logs**: Feed console logs to the analyzer
   - Input: `log_content` (string) - The console output from your Ray job

2. **get_job_status**: Get overall job status summary
   - No inputs required
   - Returns: Job status, throughput, resource usage, errors/warnings count

3. **check_resource_allocation**: Check current resource usage and pending demands
   - No inputs required
   - Returns: CPU/GPU utilization and any resource bottlenecks

4. **get_pipeline_progress**: Get detailed progress of each pipeline stage
   - No inputs required
   - Returns: Progress percentage, items processed, throughput for each stage

5. **get_vllm_metrics**: Get vLLM engine performance metrics
   - No inputs required
   - Returns: Token throughput, request queue, cache usage for each engine

## Example Usage with Claude Desktop

1. Add to your Claude Desktop config:
```json
{
  "mcpServers": {
    "ray-job-monitor": {
      "command": "python",
      "args": ["/home/asaha/ray-summit-2025/ray_job_monitor_mcp.py"]
    }
  }
}
```

2. In Claude, use the tools:
```
# First, analyze your logs
Use the analyze_ray_logs tool with the console output

# Then query status
Use the get_job_status tool

# Check for resource bottlenecks
Use the check_resource_allocation tool

# Monitor pipeline progress
Use the get_pipeline_progress tool

# Check vLLM performance
Use the get_vllm_metrics tool
```

## What the Server Monitors

### Resource Allocation Patterns
- `Resources: {CPU: 5.0/48.0, GPU: 2.0/2.0}`
- `Demands: {CPU: 4.0, GPU: 1.0}`
- Pending actors/tasks

### Pipeline Progress Patterns
- `Running Dataset: ... 73%|███...█| 52.0/71.0 [12:35<03:31, 11.1s/row]`
- `MapBatches(vLLMEngineStageUDF): ... 74%|████...| 7.41k/10.1k [12:35<03:02, 14.5 row/s]`

### vLLM Metrics Patterns
- `Engine 000: Avg prompt throughput: 2800.7 tokens/s, Avg generation throughput: 492.2 tokens/s`
- Running/waiting requests, GPU cache usage, prefix cache hit rate

### Job Completion Patterns
- `Total samples: 1000`
- `Throughput: 32.45 samples/second`
- `Pipeline completed successfully!`

## Integration with Your Ray Pipeline

The server is designed to work with the image captioning pipelines in this repository. Simply copy the console output from your Ray job and use the `analyze_ray_logs` tool to get insights.

## Troubleshooting

- If the server doesn't detect certain patterns, check the log format matches the expected patterns
- The server maintains a buffer of the last 1000 log lines for context
- Resource metrics are updated whenever new allocation information is found in the logs
