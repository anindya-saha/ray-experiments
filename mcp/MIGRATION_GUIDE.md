# Migration Guide: From Original to Enhanced Ray Monitor

This guide helps you migrate from the original log-based Ray monitor to the enhanced version that supports both REST API and log analysis.

## What's New

The enhanced monitor adds:
- **REST API Support**: Direct integration with Ray Dashboard API
- **Live Job Monitoring**: Real-time updates for running jobs
- **Job Management**: Ability to stop jobs
- **Backward Compatibility**: All original log analysis features preserved

## File Mapping

| Original File | Enhanced Version | Notes |
|--------------|------------------|-------|
| `ray_job_monitor_mcp.py` | `ray_job_monitor_enhanced_mcp.py` | Enhanced MCP server |
| `analyze_ray_logs.py` | `analyze_ray_logs_enhanced.py` | Enhanced CLI tool |
| `ray_job_analyzer_standalone.py` | (same, used by both) | Core analysis logic |

## MCP Configuration Changes

### Original Configuration
```json
{
  "mcpServers": {
    "ray-job-monitor": {
      "command": "python",
      "args": ["/path/to/ray_job_monitor_mcp.py"]
    }
  }
}
```

### Enhanced Configuration
```json
{
  "mcpServers": {
    "ray-job-monitor-enhanced": {
      "command": "python",
      "args": ["/path/to/ray_job_monitor_enhanced_mcp.py"]
    }
  }
}
```

## Tool Name Changes

### Log Analysis Tools (Unchanged)
- `analyze_ray_logs` - Same functionality
- `get_job_status` - Same functionality
- `check_resource_allocation` - Same functionality
- `get_pipeline_progress` - Same functionality
- `get_vllm_metrics` - Same functionality

### New REST API Tools
- `list_ray_jobs` - List all jobs with filtering
- `monitor_ray_job` - Monitor specific job via API
- `get_ray_job_logs` - Get logs via API (includes analysis)
- `stop_ray_job` - Stop a running job

## CLI Usage Changes

### Original CLI (Log Analysis Only)
```bash
# Analyze log file
python analyze_ray_logs.py --file ray_job.log

# Stream logs
tail -f ray_job.log | python analyze_ray_logs.py
```

### Enhanced CLI (Both Modes)
```bash
# API mode - monitor running job
python analyze_ray_logs_enhanced.py --api --job-id raysubmit_123

# API mode - list jobs
python analyze_ray_logs_enhanced.py --api --status running

# Log mode - same as original
python analyze_ray_logs_enhanced.py --file ray_job.log
tail -f ray_job.log | python analyze_ray_logs_enhanced.py
```

## Common Migration Scenarios

### Scenario 1: You Only Analyze Completed Job Logs

**No changes needed!** The enhanced tool is backward compatible:
```bash
# These work exactly the same
python analyze_ray_logs_enhanced.py --file old_job.log
python analyze_ray_logs_enhanced.py < saved_logs.txt
```

### Scenario 2: You Want to Monitor Running Jobs

**New capability!** Use the API mode:
```bash
# Monitor a specific job
python analyze_ray_logs_enhanced.py --api --job-id raysubmit_abc123

# Watch all running jobs
python analyze_ray_logs_enhanced.py --api --status running
```

### Scenario 3: You Use MCP/Claude Desktop

Update your configuration to use the enhanced server, then:
```
# Old way (still works)
Use analyze_ray_logs tool with log_content "..."

# New way (for running jobs)
Use list_ray_jobs tool
Use monitor_ray_job tool with job_id "raysubmit_123"
```

## Feature Comparison

| Feature | Original | Enhanced |
|---------|----------|----------|
| Analyze log files | ✅ | ✅ |
| Stream log analysis | ✅ | ✅ |
| Parse resource usage | ✅ | ✅ |
| Track pipeline progress | ✅ | ✅ |
| vLLM metrics | ✅ | ✅ |
| Monitor running jobs | ❌ | ✅ |
| List all jobs | ❌ | ✅ |
| Stop jobs | ❌ | ✅ |
| Get logs via API | ❌ | ✅ |
| Remote cluster support | ❌ | ✅ |

## Breaking Changes

None! The enhanced version is fully backward compatible. All original tools and commands continue to work.

## Recommendations

1. **For new deployments**: Use the enhanced version
2. **For existing scripts**: No changes required, but consider upgrading for new features
3. **For CI/CD**: The JSON output format is unchanged
4. **For remote clusters**: Use `--dashboard-url` parameter

## Rollback

If you need to rollback:
1. Change MCP config back to original `ray_job_monitor_mcp.py`
2. Use original `analyze_ray_logs.py` for CLI
3. All log analysis features remain the same

## Getting Help

- Check `README.md` for detailed documentation
- Run with `--help` for CLI options
- Test with `test_enhanced_monitor.py` for examples
