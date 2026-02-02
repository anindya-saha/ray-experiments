#!/usr/bin/env python3
"""
Enhanced MCP Server for Ray Job Monitoring
Combines REST API access for live jobs and log analysis for completed jobs
"""

import asyncio
import json
import re
import aiohttp
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

# MCP SDK imports
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from mcp.server.models import InitializationOptions

# Import the standalone analyzer for log parsingß
from ray_job_analyzer_standalone import RayLogAnalyzer


@dataclass
class RayJobInfo:
    """Ray job information from API"""
    job_id: str
    status: str
    entrypoint: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class RayAPIClient:
    """Client for Ray Dashboard REST API"""
    
    def __init__(self, dashboard_url: str = "http://localhost:8265"):
        self.dashboard_url = dashboard_url
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def close(self):
        """Close the session"""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def list_jobs(self) -> List[Dict[str, Any]]:
        """List all jobs"""
        try:
            session = await self._get_session()
            async with session.get(f"{self.dashboard_url}/api/jobs/") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data if isinstance(data, list) else []
        except Exception as e:
            print(f"Error listing jobs: {e}")
        return []
    
    async def get_job_details(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get specific job details"""
        try:
            session = await self._get_session()
            async with session.get(f"{self.dashboard_url}/api/jobs/{job_id}") as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception as e:
            print(f"Error getting job details: {e}")
        return None
    
    async def get_job_logs(self, job_id: str) -> Optional[str]:
        """Get job logs"""
        try:
            session = await self._get_session()
            async with session.get(f"{self.dashboard_url}/api/jobs/{job_id}/logs") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("logs", "")
        except Exception as e:
            print(f"Error getting job logs: {e}")
        return None
    
    async def stop_job(self, job_id: str) -> bool:
        """Stop a running job"""
        try:
            session = await self._get_session()
            async with session.post(f"{self.dashboard_url}/api/jobs/{job_id}/stop") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("stopped", False)
        except Exception as e:
            print(f"Error stopping job: {e}")
        return False
    
    async def submit_job(self, entrypoint: str, runtime_env: Optional[Dict] = None, 
                        metadata: Optional[Dict] = None) -> Optional[str]:
        """Submit a new job"""
        try:
            session = await self._get_session()
            payload = {
                "entrypoint": entrypoint,
                "runtime_env": runtime_env or {},
                "metadata": metadata or {}
            }
            async with session.post(f"{self.dashboard_url}/api/jobs/", json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("job_id")
        except Exception as e:
            print(f"Error submitting job: {e}")
        return None


class RayJobMonitor:
    """Combined monitor using both REST API and log analysis"""
    
    def __init__(self, dashboard_url: str = "http://localhost:8265"):
        self.api_client = RayAPIClient(dashboard_url)
        self.log_analyzer = RayLogAnalyzer()
    
    async def close(self):
        """Clean up resources"""
        await self.api_client.close()
    
    def parse_job_info(self, job_data: Dict[str, Any]) -> RayJobInfo:
        """Parse job data from API into RayJobInfo"""
        return RayJobInfo(
            job_id=job_data.get("job_id", ""),
            status=job_data.get("status", "UNKNOWN"),
            entrypoint=job_data.get("entrypoint", ""),
            start_time=job_data.get("start_time"),
            end_time=job_data.get("end_time"),
            metadata=job_data.get("metadata", {}),
            error_message=job_data.get("error_message")
        )
    
    async def get_job_summary_from_api(self, job_id: str) -> Dict[str, Any]:
        """Get job summary using REST API"""
        job_details = await self.api_client.get_job_details(job_id)
        if not job_details:
            return {"error": f"Job {job_id} not found"}
        
        job_info = self.parse_job_info(job_details)
        summary = {
            "job_id": job_info.job_id,
            "status": job_info.status,
            "entrypoint": job_info.entrypoint,
            "metadata": job_info.metadata,
            "error_message": job_info.error_message
        }
        
        if job_info.start_time:
            summary["start_time"] = datetime.fromtimestamp(job_info.start_time).isoformat()
            if job_info.end_time:
                summary["end_time"] = datetime.fromtimestamp(job_info.end_time).isoformat()
                summary["duration_seconds"] = job_info.end_time - job_info.start_time
            elif job_info.status == "RUNNING":
                summary["duration_seconds"] = datetime.now().timestamp() - job_info.start_time
        
        # Get logs for additional analysis if job is running or recently completed
        if job_info.status in ["RUNNING", "SUCCEEDED", "FAILED"]:
            logs = await self.api_client.get_job_logs(job_id)
            if logs:
                # Analyze logs
                for line in logs.split('\n'):
                    if line.strip():
                        self.log_analyzer.analyze_log_line(line)
                
                # Add analysis results to summary
                analysis = self.log_analyzer.get_summary()
                summary["resource_status"] = analysis["resource_status"]
                summary["pipeline_stages"] = analysis["pipeline_stages"]
                summary["vllm_metrics"] = analysis["vllm_metrics"]
                summary["analysis"] = {
                    "total_samples": analysis["job_summary"]["total_samples"],
                    "overall_throughput": analysis["job_summary"]["overall_throughput"],
                    "errors": len(analysis["job_summary"]["errors"]),
                    "warnings": len(analysis["job_summary"]["warnings"])
                }
        
        return summary
    
    def analyze_log_content(self, log_content: str) -> Dict[str, Any]:
        """Analyze log content (for completed jobs or external logs)"""
        # Reset analyzer for fresh analysis
        self.log_analyzer = RayLogAnalyzer()
        
        for line in log_content.split('\n'):
            if line.strip():
                self.log_analyzer.analyze_log_line(line)
        
        return self.log_analyzer.get_summary()


# Global monitor instance
monitor: Optional[RayJobMonitor] = None

# MCP Server instance
server = Server("ray-job-monitor-enhanced")


@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available tools"""
    return [
        # REST API based tools for live monitoring
        Tool(
            name="list_ray_jobs",
            description="List all Ray jobs (running and completed) using REST API",
            inputSchema={
                "type": "object",
                "properties": {
                    "dashboard_url": {
                        "type": "string",
                        "description": "Ray dashboard URL (default: http://localhost:8265)"
                    },
                    "status_filter": {
                        "type": "string",
                        "description": "Filter by status: PENDING, RUNNING, SUCCEEDED, FAILED, or ALL (default: ALL)"
                    }
                }
            }
        ),
        Tool(
            name="monitor_ray_job",
            description="Monitor a specific Ray job using REST API (best for running jobs)",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "The Ray job ID to monitor"
                    },
                    "dashboard_url": {
                        "type": "string",
                        "description": "Ray dashboard URL (default: http://localhost:8265)"
                    }
                },
                "required": ["job_id"]
            }
        ),
        Tool(
            name="get_ray_job_logs",
            description="Get and analyze logs for a specific Ray job",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "The Ray job ID"
                    },
                    "dashboard_url": {
                        "type": "string",
                        "description": "Ray dashboard URL (default: http://localhost:8265)"
                    },
                    "tail_lines": {
                        "type": "integer",
                        "description": "Number of recent log lines to show (default: all)"
                    }
                },
                "required": ["job_id"]
            }
        ),
        Tool(
            name="stop_ray_job",
            description="Stop a running Ray job",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "The Ray job ID to stop"
                    },
                    "dashboard_url": {
                        "type": "string",
                        "description": "Ray dashboard URL (default: http://localhost:8265)"
                    }
                },
                "required": ["job_id"]
            }
        ),
        
        # Log analysis tools for completed jobs or external logs
        Tool(
            name="analyze_ray_logs",
            description="Analyze Ray job logs from console output or log files (best for completed jobs)",
            inputSchema={
                "type": "object",
                "properties": {
                    "log_content": {
                        "type": "string",
                        "description": "The console log content to analyze"
                    }
                },
                "required": ["log_content"]
            }
        ),
        Tool(
            name="get_job_status",
            description="Get current job status from analyzed logs",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="check_resource_allocation",
            description="Check resource allocation and pending demands from analyzed logs",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="get_pipeline_progress",
            description="Get pipeline stage progress from analyzed logs",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="get_vllm_metrics",
            description="Get vLLM engine metrics from analyzed logs",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls"""
    global monitor
    
    # Initialize monitor if needed
    dashboard_url = arguments.get("dashboard_url", "http://localhost:8265")
    if monitor is None or monitor.api_client.dashboard_url != dashboard_url:
        if monitor:
            await monitor.close()
        monitor = RayJobMonitor(dashboard_url)
    
    try:
        # REST API based tools
        if name == "list_ray_jobs":
            jobs = await monitor.api_client.list_jobs()
            status_filter = arguments.get("status_filter", "ALL").upper()
            
            if status_filter != "ALL":
                jobs = [j for j in jobs if j.get("status") == status_filter]
            
            if not jobs:
                return [TextContent(type="text", text=f"No Ray jobs found with status filter: {status_filter}")]
            
            # Group jobs by status
            jobs_by_status = {}
            for job in jobs:
                status = job.get("status", "UNKNOWN")
                if status not in jobs_by_status:
                    jobs_by_status[status] = []
                jobs_by_status[status].append(job)
            
            output = f"Ray Jobs Summary (Total: {len(jobs)})\n" + "="*60 + "\n\n"
            
            # Show jobs grouped by status
            for status in ["RUNNING", "PENDING", "SUCCEEDED", "FAILED", "STOPPED"]:
                if status in jobs_by_status:
                    output += f"\n{status} ({len(jobs_by_status[status])} jobs):\n" + "-"*40 + "\n"
                    for job in jobs_by_status[status][:10]:  # Show max 10 per status
                        job_info = monitor.parse_job_info(job)
                        output += f"  • {job_info.job_id}\n"
                        output += f"    Entrypoint: {job_info.entrypoint}\n"
                        if job_info.start_time:
                            start_dt = datetime.fromtimestamp(job_info.start_time)
                            output += f"    Started: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}\n"
                        if job_info.error_message:
                            output += f"    Error: {job_info.error_message[:100]}...\n"
                    if len(jobs_by_status[status]) > 10:
                        output += f"  ... and {len(jobs_by_status[status]) - 10} more\n"
            
            return [TextContent(type="text", text=output)]
        
        elif name == "monitor_ray_job":
            job_id = arguments["job_id"]
            summary = await monitor.get_job_summary_from_api(job_id)
            
            if "error" in summary:
                return [TextContent(type="text", text=summary["error"])]
            
            output = f"Ray Job Monitor: {job_id}\n" + "="*60 + "\n\n"
            output += f"Status: {summary['status']}\n"
            output += f"Entrypoint: {summary['entrypoint']}\n"
            
            if "start_time" in summary:
                output += f"Started: {summary['start_time']}\n"
            if "end_time" in summary:
                output += f"Ended: {summary['end_time']}\n"
            if "duration_seconds" in summary:
                duration = summary['duration_seconds']
                output += f"Duration: {duration//60:.0f}m {duration%60:.0f}s\n"
            
            if summary.get("error_message"):
                output += f"\n❌ Error: {summary['error_message']}\n"
            
            # Add resource and performance metrics if available
            if "resource_status" in summary:
                rs = summary["resource_status"]
                output += f"\nResource Allocation:\n"
                output += f"  CPU: {rs['cpus_used']}/{rs['cpus_total']} ({rs['cpus_used']/rs['cpus_total']*100:.1f}%)\n"
                output += f"  GPU: {rs['gpus_used']}/{rs['gpus_total']} ({rs['gpus_used']/rs['gpus_total']*100:.1f}%)\n"
                
                if rs['pending_cpu_demand'] > 0 or rs['pending_gpu_demand'] > 0:
                    output += f"\n⚠️  Pending Demands:\n"
                    output += f"  CPU: {rs['pending_cpu_demand']}, GPU: {rs['pending_gpu_demand']}\n"
            
            if "analysis" in summary:
                analysis = summary["analysis"]
                output += f"\nJob Performance:\n"
                output += f"  Total Samples: {analysis['total_samples']}\n"
                output += f"  Throughput: {analysis['overall_throughput']:.2f} samples/sec\n"
                output += f"  Errors: {analysis['errors']}, Warnings: {analysis['warnings']}\n"
            
            if "pipeline_stages" in summary and summary["pipeline_stages"]:
                output += f"\nPipeline Progress:\n"
                for stage_name, stage in summary["pipeline_stages"].items():
                    output += f"  {stage_name}: {stage['progress_percent']:.1f}% "
                    output += f"({stage['processed_items']}/{stage['total_items']})\n"
            
            return [TextContent(type="text", text=output)]
        
        elif name == "get_ray_job_logs":
            job_id = arguments["job_id"]
            tail_lines = arguments.get("tail_lines")
            
            logs = await monitor.api_client.get_job_logs(job_id)
            if logs is None:
                return [TextContent(type="text", text=f"Could not retrieve logs for job {job_id}")]
            
            # Analyze logs
            analysis = monitor.analyze_log_content(logs)
            
            output = f"Ray Job Logs Analysis: {job_id}\n" + "="*60 + "\n\n"
            
            # Add analysis summary
            job_sum = analysis['job_summary']
            output += f"Status: {job_sum['status']}\n"
            if job_sum['total_samples'] > 0:
                output += f"Total Samples: {job_sum['total_samples']}\n"
                output += f"Throughput: {job_sum['overall_throughput']:.2f} samples/sec\n"
            
            output += f"Errors: {len(job_sum['errors'])}, Warnings: {len(job_sum['warnings'])}\n"
            
            # Show recent logs
            output += "\n" + "-"*40 + "\nRecent Logs:\n"
            log_lines = logs.split('\n')
            if tail_lines and tail_lines < len(log_lines):
                log_lines = log_lines[-tail_lines:]
            
            # Truncate if too long
            if len('\n'.join(log_lines)) > 5000:
                output += '\n'.join(log_lines[-50:])  # Last 50 lines
                output += f"\n\n... (showing last 50 lines of {len(log_lines)} total)"
            else:
                output += '\n'.join(log_lines)
            
            return [TextContent(type="text", text=output)]
        
        elif name == "stop_ray_job":
            job_id = arguments["job_id"]
            stopped = await monitor.api_client.stop_job(job_id)
            
            if stopped:
                return [TextContent(type="text", text=f"✅ Successfully stopped job {job_id}")]
            else:
                return [TextContent(type="text", text=f"❌ Failed to stop job {job_id}")]
        
        # Log analysis tools (using the existing analyzer)
        elif name == "analyze_ray_logs":
            log_content = arguments.get("log_content", "")
            summary = monitor.analyze_log_content(log_content)
            
            return [TextContent(
                type="text",
                text=f"Log analysis complete. Use other tools to query specific aspects:\n"
                     f"- get_job_status: Overall job status\n"
                     f"- check_resource_allocation: Resource usage\n"
                     f"- get_pipeline_progress: Pipeline stages\n"
                     f"- get_vllm_metrics: vLLM performance\n\n"
                     f"Summary: {json.dumps(summary, indent=2)}"
            )]
        
        elif name == "get_job_status":
            summary = monitor.log_analyzer.get_summary()
            job = summary['job_summary']
            
            status_msg = f"Ray Job Status (from logs):\n" + "="*40 + "\n"
            status_msg += f"Status: {job['status']}\n"
            status_msg += f"Total Samples: {job['total_samples']}\n"
            status_msg += f"Overall Throughput: {job['overall_throughput']:.2f} samples/second\n"
            status_msg += f"Errors: {len(job['errors'])}\n"
            status_msg += f"Warnings: {len(job['warnings'])}\n"
            
            return [TextContent(type="text", text=status_msg)]
        
        elif name == "check_resource_allocation":
            status = monitor.log_analyzer.resource_status
            
            allocation_msg = f"Resource Allocation (from logs):\n" + "="*40 + "\n"
            if status.cpus_total > 0:
                allocation_msg += f"CPU: {status.cpus_used}/{status.cpus_total} ({status.cpus_used/status.cpus_total*100:.1f}%)\n"
            if status.gpus_total > 0:
                allocation_msg += f"GPU: {status.gpus_used}/{status.gpus_total} ({status.gpus_used/status.gpus_total*100:.1f}%)\n"
            
            if status.pending_cpu_demand > 0 or status.pending_gpu_demand > 0:
                allocation_msg += f"\n⚠️  Pending Demands:\n"
                allocation_msg += f"  CPU: {status.pending_cpu_demand}\n"
                allocation_msg += f"  GPU: {status.pending_gpu_demand}\n"
                allocation_msg += f"  Actors waiting: {status.pending_actors}\n"
                allocation_msg += f"  Tasks waiting: {status.pending_tasks}\n"
            else:
                allocation_msg += "\n✅ No pending resource demands"
            
            return [TextContent(type="text", text=allocation_msg)]
        
        elif name == "get_pipeline_progress":
            stages = monitor.log_analyzer.pipeline_stages
            
            if not stages:
                return [TextContent(type="text", text="No pipeline stages detected in logs.")]
            
            progress_msg = "Pipeline Progress (from logs):\n" + "="*40 + "\n"
            for stage_name, stage in stages.items():
                progress_bar = "█" * int(stage.progress_percent / 5) + "░" * (20 - int(stage.progress_percent / 5))
                progress_msg += f"\n{stage_name}:\n"
                progress_msg += f"  [{progress_bar}] {stage.progress_percent:.1f}%\n"
                progress_msg += f"  Items: {stage.processed_items}/{stage.total_items}\n"
                progress_msg += f"  Speed: {stage.throughput:.2f} {stage.throughput_unit}\n"
            
            return [TextContent(type="text", text=progress_msg)]
        
        elif name == "get_vllm_metrics":
            metrics = monitor.log_analyzer.vllm_metrics
            
            if not metrics:
                return [TextContent(type="text", text="No vLLM metrics detected in logs.")]
            
            metrics_msg = "vLLM Engine Metrics (from logs):\n" + "="*40 + "\n"
            for engine_id, metric in metrics.items():
                metrics_msg += f"\nEngine {engine_id}:\n"
                metrics_msg += f"  Prompt: {metric.prompt_throughput:.1f} tokens/s\n"
                metrics_msg += f"  Generation: {metric.generation_throughput:.1f} tokens/s\n"
                metrics_msg += f"  Requests: {metric.running_requests} running, {metric.waiting_requests} waiting\n"
                metrics_msg += f"  GPU Cache: {metric.gpu_kv_cache_usage:.1f}%\n"
                metrics_msg += f"  Cache Hits: {metric.prefix_cache_hit_rate:.1f}%\n"
            
            return [TextContent(type="text", text=metrics_msg)]
        
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
            
    except aiohttp.ClientError as e:
        return [TextContent(
            type="text", 
            text=f"Error connecting to Ray Dashboard at {dashboard_url}: {str(e)}\n\n"
                 f"Make sure the Ray Dashboard is running and accessible.\n"
                 f"You can still use the log analysis tools with the 'analyze_ray_logs' command."
        )]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def cleanup():
    """Cleanup resources"""
    global monitor
    if monitor:
        await monitor.close()
        monitor = None


async def main():
    """Run the MCP server"""
    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, InitializationOptions)
    finally:
        await cleanup()


if __name__ == "__main__":
    asyncio.run(main())
