# Ray Job Monitor - Enhanced Edition

A comprehensive monitoring solution for Ray jobs that combines REST API access for live monitoring and log analysis for completed jobs.

## Overview

This enhanced monitoring solution provides two complementary approaches:

1. **REST API Monitoring** - Real-time monitoring of running jobs via Ray Dashboard API
2. **Log Analysis** - Deep analysis of job logs for completed jobs or offline analysis

## Components

### 1. Enhanced MCP Server (`ray_job_monitor_enhanced_mcp.py`)

The main MCP server that provides both REST API and log analysis capabilities.

#### REST API Tools (for live monitoring):
- `list_ray_jobs` - List all jobs with filtering options
- `monitor_ray_job` - Monitor a specific running job
- `get_ray_job_logs` - Fetch and analyze job logs
- `stop_ray_job` - Stop a running job

#### Log Analysis Tools (for completed jobs):
- `analyze_ray_logs` - Process log content from files or console
- `get_job_status` - Overall job status from logs
- `check_resource_allocation` - Resource usage analysis
- `get_pipeline_progress` - Pipeline stage progress
- `get_vllm_metrics` - vLLM engine performance metrics

### 2. Enhanced CLI Tool (`analyze_ray_logs_enhanced.py`)

A versatile command-line tool that supports both monitoring modes.

#### API Mode Examples:
```bash
# Monitor a specific running job
python analyze_ray_logs_enhanced.py --api --job-id raysubmit_abc123

# List all running jobs
python analyze_ray_logs_enhanced.py --api --status running

# Monitor with custom dashboard URL
python analyze_ray_logs_enhanced.py --api --job-id raysubmit_abc123 --dashboard-url http://ray-head:8265
```

#### Log Analysis Mode Examples:
```bash
# Analyze a log file
python analyze_ray_logs_enhanced.py --file ray_job.log

# Stream logs in real-time
tail -f ray_job.log | python analyze_ray_logs_enhanced.py

# JSON output for programmatic use
python analyze_ray_logs_enhanced.py --file ray_job.log --json
```

### 3. Original Components (still available)

- `ray_job_monitor_mcp.py` - Original log-based MCP server
- `analyze_ray_logs.py` - Original CLI tool
- `ray_job_analyzer_standalone.py` - Core analysis logic

## Installation

1. Install dependencies:
```bash
pip install mcp aiohttp
```

2. Configure MCP servers (for Claude Desktop or other MCP clients):
```json
{
  "mcpServers": {
    "ray-job-monitor-enhanced": {
      "command": "python",
      "args": ["/path/to/ray_job_monitor_enhanced_mcp.py"],
      "env": {
        "PYTHONPATH": "/path/to/mcp/directory"
      }
    }
  }
}
```

## Usage Scenarios

### Scenario 1: Monitoring Active Jobs

Use the REST API tools when you have running Ray jobs and want real-time updates:

```python
# In Claude or MCP client
Use list_ray_jobs tool with status_filter "RUNNING"
Use monitor_ray_job tool with job_id "raysubmit_123abc"
```

### Scenario 2: Analyzing Completed Jobs

Use log analysis tools when you have saved logs from completed jobs:

```python
# First analyze the logs
Use analyze_ray_logs tool with log_content from your file

# Then query specific aspects
Use get_pipeline_progress tool
Use get_vllm_metrics tool
```

### Scenario 3: Continuous Monitoring

Use the CLI tool for dashboard-style monitoring:

```bash
# Monitor a running job with live updates
python analyze_ray_logs_enhanced.py --api --job-id raysubmit_123 --interval 2
```

## Features

### REST API Monitoring
- **Real-time Updates**: Get current job status without log parsing
- **Job Management**: Start, stop, and list jobs
- **Structured Data**: Access job metadata, timing, and errors
- **Multi-job View**: Monitor all jobs in your cluster

### Log Analysis
- **Pattern Recognition**: Automatically detects Ray log patterns
- **Resource Tracking**: CPU/GPU usage and pending demands
- **Pipeline Progress**: Stage-by-stage completion tracking
- **vLLM Metrics**: Token throughput and cache utilization
- **Error Detection**: Identifies errors and warnings

### Combined Benefits
- **Flexibility**: Use API for live jobs, logs for post-mortem
- **Comprehensive**: Get both high-level status and detailed metrics
- **Resilient**: Falls back to log analysis if API unavailable
- **Historical**: Analyze saved logs from past jobs

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Enhanced MCP Server                    │
├─────────────────────────┬───────────────────────────────┤
│    REST API Client      │      Log Analyzer             │
├─────────────────────────┼───────────────────────────────┤
│ • list_jobs()          │ • analyze_log_line()          │
│ • get_job_details()    │ • get_summary()               │
│ • get_job_logs()       │ • resource_status             │
│ • stop_job()           │ • pipeline_stages             │
│                        │ • vllm_metrics                │
└─────────────────────────┴───────────────────────────────┘
                    ↓                    ↓
        ┌──────────────────┐   ┌──────────────────┐
        │ Ray Dashboard API │   │   Log Files      │
        │ (Port 8265)       │   │   Console Output │
        └──────────────────┘   └──────────────────┘
```

## Advanced Usage

### Custom Dashboard URL

If your Ray cluster is remote or uses a different port:

```bash
# CLI
python analyze_ray_logs_enhanced.py --api --dashboard-url http://ray-cluster:8265

# MCP Tool
Use list_ray_jobs tool with dashboard_url "http://ray-cluster:8265"
```

### Filtering Jobs

Filter jobs by status when listing:

```python
# Show only failed jobs
Use list_ray_jobs tool with status_filter "FAILED"
```

### Continuous Integration

Use JSON output for CI/CD pipelines:

```bash
# Get job status as JSON
python analyze_ray_logs_enhanced.py --api --job-id $JOB_ID --json | jq '.status'
```

## Troubleshooting

### Ray Dashboard Not Accessible

If you get connection errors:
1. Check Ray Dashboard is running: `ray dashboard`
2. Verify the URL and port (default: http://localhost:8265)
3. Check firewall/security group settings
4. Fall back to log analysis mode

### Log Patterns Not Detected

If certain metrics aren't detected:
1. Ensure Ray is configured with appropriate logging levels
2. Check log format matches expected patterns
3. Update regex patterns in `ray_job_analyzer_standalone.py`

### Performance Considerations

- API calls are rate-limited by Ray Dashboard
- Log analysis keeps last 1000 lines in memory
- Use `--interval` flag to control update frequency

## Contributing

To add new log patterns or metrics:
1. Update regex patterns in `ray_job_analyzer_standalone.py`
2. Add new data structures if needed
3. Update the summary generation logic

## License

Same as Ray Summit 2025 project license.
