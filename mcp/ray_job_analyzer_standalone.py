#!/usr/bin/env python3
"""
Standalone Ray Job Log Analyzer (without MCP dependencies)
"""

import re
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict


@dataclass
class ResourceStatus:
    """Track resource allocation status"""
    cpus_used: float = 0.0
    cpus_total: float = 0.0
    gpus_used: float = 0.0
    gpus_total: float = 0.0
    pending_cpu_demand: float = 0.0
    pending_gpu_demand: float = 0.0
    pending_actors: int = 0
    pending_tasks: int = 0


@dataclass
class PipelineStatus:
    """Track pipeline execution status"""
    stage_name: str = ""
    progress_percent: float = 0.0
    processed_items: int = 0
    total_items: int = 0
    throughput: float = 0.0
    throughput_unit: str = ""


@dataclass
class VLLMMetrics:
    """Track vLLM engine metrics"""
    engine_id: str = ""
    prompt_throughput: float = 0.0
    generation_throughput: float = 0.0
    running_requests: int = 0
    waiting_requests: int = 0
    gpu_kv_cache_usage: float = 0.0
    prefix_cache_hit_rate: float = 0.0


@dataclass
class JobSummary:
    """Overall job summary"""
    status: str = "unknown"
    start_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    total_samples: int = 0
    overall_throughput: float = 0.0
    errors: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class RayLogAnalyzer:
    """Analyzes Ray job logs to extract status information"""
    
    def __init__(self):
        self.resource_status = ResourceStatus()
        self.pipeline_stages: Dict[str, PipelineStatus] = {}
        self.vllm_metrics: Dict[str, VLLMMetrics] = {}
        self.job_summary = JobSummary()
        self.log_buffer: List[str] = []
        
    def analyze_log_line(self, line: str) -> None:
        """Analyze a single log line and update status"""
        self.log_buffer.append(line)
        if len(self.log_buffer) > 1000:
            self.log_buffer = self.log_buffer[-500:]  # Keep last 500 lines
        
        # Check for resource allocation patterns
        self._check_resource_allocation(line)
        
        # Check for pipeline progress
        self._check_pipeline_progress(line)
        
        # Check for vLLM metrics
        self._check_vllm_metrics(line)
        
        # Check for errors and warnings
        self._check_errors_warnings(line)
        
        # Check for job completion
        self._check_job_completion(line)
    
    def _check_resource_allocation(self, line: str) -> None:
        """Extract resource allocation information"""
        # Pattern: "Resources: {CPU: 5.0/48.0, GPU: 2.0/2.0, ...}"
        resource_match = re.search(r'Resources:\s*{([^}]+)}', line)
        if resource_match:
            resources = resource_match.group(1)
            
            # Extract CPU usage
            cpu_match = re.search(r'CPU:\s*([\d.]+)/([\d.]+)', resources)
            if cpu_match:
                self.resource_status.cpus_used = float(cpu_match.group(1))
                self.resource_status.cpus_total = float(cpu_match.group(2))
            
            # Extract GPU usage
            gpu_match = re.search(r'GPU:\s*([\d.]+)/([\d.]+)', resources)
            if gpu_match:
                self.resource_status.gpus_used = float(gpu_match.group(1))
                self.resource_status.gpus_total = float(gpu_match.group(2))
        
        # Pattern for pending demands
        pending_match = re.search(r'Demands:\s*{([^}]+)}', line)
        if pending_match:
            demands = pending_match.group(1)
            
            # Extract pending CPU demand
            cpu_demand = re.search(r'CPU:\s*([\d.]+)', demands)
            if cpu_demand:
                self.resource_status.pending_cpu_demand = float(cpu_demand.group(1))
            
            # Extract pending GPU demand
            gpu_demand = re.search(r'GPU:\s*([\d.]+)', demands)
            if gpu_demand:
                self.resource_status.pending_gpu_demand = float(gpu_demand.group(1))
        
        # Pattern for pending actors/tasks
        pending_actors = re.search(r'(\d+)\s+pending\s+actors?', line, re.IGNORECASE)
        if pending_actors:
            self.resource_status.pending_actors = int(pending_actors.group(1))
            
        pending_tasks = re.search(r'(\d+)\s+pending\s+tasks?', line, re.IGNORECASE)
        if pending_tasks:
            self.resource_status.pending_tasks = int(pending_tasks.group(1))
    
    def _check_pipeline_progress(self, line: str) -> None:
        """Extract pipeline execution progress"""
        # Pattern: "Running Dataset: ... 73%|███...█| 52.0/71.0 [12:35<03:31, 11.1s/row]"
        progress_match = re.search(
            r'([\w\s]+Dataset|MapBatches\([\w]+\)):\s*.*?(\d+)%.*?\s+([\d.]+)/([\d.]+).*?\[([\d:]+)<.*?,\s*([\d.]+)\s*(\w+/\w+)\]',
            line
        )
        if progress_match:
            stage_name = progress_match.group(1).strip()
            progress_percent = float(progress_match.group(2))
            processed = float(progress_match.group(3))
            total = float(progress_match.group(4))
            throughput = float(progress_match.group(6))
            unit = progress_match.group(7)
            
            self.pipeline_stages[stage_name] = PipelineStatus(
                stage_name=stage_name,
                progress_percent=progress_percent,
                processed_items=int(processed),
                total_items=int(total),
                throughput=throughput,
                throughput_unit=unit
            )
    
    def _check_vllm_metrics(self, line: str) -> None:
        """Extract vLLM engine metrics"""
        # Pattern: "Engine 000: Avg prompt throughput: 2800.7 tokens/s, ..."
        vllm_match = re.search(
            r'Engine\s+(\d+):\s*Avg prompt throughput:\s*([\d.]+)\s*tokens/s,\s*'
            r'Avg generation throughput:\s*([\d.]+)\s*tokens/s,\s*'
            r'Running:\s*(\d+)\s*reqs,\s*Waiting:\s*(\d+)\s*reqs,\s*'
            r'GPU KV cache usage:\s*([\d.]+)%,\s*'
            r'Prefix cache hit rate:\s*([\d.]+)%',
            line
        )
        if vllm_match:
            engine_id = vllm_match.group(1)
            self.vllm_metrics[engine_id] = VLLMMetrics(
                engine_id=engine_id,
                prompt_throughput=float(vllm_match.group(2)),
                generation_throughput=float(vllm_match.group(3)),
                running_requests=int(vllm_match.group(4)),
                waiting_requests=int(vllm_match.group(5)),
                gpu_kv_cache_usage=float(vllm_match.group(6)),
                prefix_cache_hit_rate=float(vllm_match.group(7))
            )
    
    def _check_errors_warnings(self, line: str) -> None:
        """Check for errors and warnings"""
        if "ERROR" in line or "FAILED" in line.upper():
            self.job_summary.errors.append(line.strip())
        elif "WARNING" in line or "WARN" in line.upper():
            self.job_summary.warnings.append(line.strip())
    
    def _check_job_completion(self, line: str) -> None:
        """Check for job completion patterns"""
        # Pattern: "Total samples: 1000"
        samples_match = re.search(r'Total samples:\s*(\d+)', line)
        if samples_match:
            self.job_summary.total_samples = int(samples_match.group(1))
        
        # Pattern: "Throughput: 32.45 samples/second"
        throughput_match = re.search(r'Throughput:\s*([\d.]+)\s*samples/second', line)
        if throughput_match:
            self.job_summary.overall_throughput = float(throughput_match.group(1))
        
        # Pattern: "Pipeline completed successfully!"
        if "Pipeline completed successfully" in line:
            self.job_summary.status = "completed"
        elif "Pipeline failed" in line:
            self.job_summary.status = "failed"
        elif any(stage.progress_percent > 0 for stage in self.pipeline_stages.values()):
            self.job_summary.status = "running"
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive job summary"""
        return {
            "job_summary": asdict(self.job_summary),
            "resource_status": asdict(self.resource_status),
            "pipeline_stages": {k: asdict(v) for k, v in self.pipeline_stages.items()},
            "vllm_metrics": {k: asdict(v) for k, v in self.vllm_metrics.items()},
            "recent_logs": self.log_buffer[-20:] if self.log_buffer else []
        }
