#!/usr/bin/env python3
"""
Example of using the Ray monitor components in your own Python scripts
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ray_job_analyzer_standalone import RayLogAnalyzer


# Example 1: Analyze logs from a file
def analyze_log_file(log_file_path):
    """Analyze a Ray job log file"""
    analyzer = RayLogAnalyzer()
    
    with open(log_file_path, 'r') as f:
        for line in f:
            analyzer.analyze_log_line(line.strip())
    
    summary = analyzer.get_summary()
    
    print(f"Job Status: {summary['job_summary']['status']}")
    print(f"Total Samples: {summary['job_summary']['total_samples']}")
    print(f"Throughput: {summary['job_summary']['overall_throughput']:.2f} samples/sec")
    
    # Check for resource issues
    rs = summary['resource_status']
    if rs['pending_cpu_demand'] > 0 or rs['pending_gpu_demand'] > 0:
        print("⚠️  Resource bottleneck detected!")
        print(f"   Pending CPU: {rs['pending_cpu_demand']}")
        print(f"   Pending GPU: {rs['pending_gpu_demand']}")
    
    return summary


# Example 2: Monitor a running job via API
async def monitor_running_job(job_id, dashboard_url="http://localhost:8265"):
    """Monitor a running Ray job via REST API"""
    import aiohttp
    
    async with aiohttp.ClientSession() as session:
        # Get job details
        async with session.get(f"{dashboard_url}/api/jobs/{job_id}") as resp:
            if resp.status != 200:
                print(f"Error: Could not find job {job_id}")
                return
            
            job_info = await resp.json()
            print(f"Job Status: {job_info['status']}")
            print(f"Entrypoint: {job_info['entrypoint']}")
        
        # Get and analyze logs
        async with session.get(f"{dashboard_url}/api/jobs/{job_id}/logs") as resp:
            if resp.status == 200:
                log_data = await resp.json()
                logs = log_data.get("logs", "")
                
                # Analyze the logs
                analyzer = RayLogAnalyzer()
                for line in logs.split('\n'):
                    if line.strip():
                        analyzer.analyze_log_line(line)
                
                summary = analyzer.get_summary()
                
                # Show pipeline progress
                for stage_name, stage in summary['pipeline_stages'].items():
                    print(f"{stage_name}: {stage['progress_percent']:.1f}% complete")


# Example 3: List and filter jobs
async def list_active_jobs(dashboard_url="http://localhost:8265"):
    """List all active Ray jobs"""
    import aiohttp
    
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{dashboard_url}/api/jobs/") as resp:
            if resp.status == 200:
                jobs = await resp.json()
                
                # Filter for running jobs
                running_jobs = [j for j in jobs if j.get("status") == "RUNNING"]
                
                print(f"Found {len(running_jobs)} running jobs:")
                for job in running_jobs:
                    print(f"  - {job['job_id']}: {job['entrypoint']}")
                
                return running_jobs


# Example usage
if __name__ == "__main__":
    print("Ray Job Monitor - Usage Examples\n")
    
    # Example 1: Analyze a log file
    if os.path.exists("sample_ray_log.txt"):
        print("1. Analyzing log file:")
        print("-" * 40)
        analyze_log_file("sample_ray_log.txt")
        print()
    
    # Example 2: List active jobs (requires Ray cluster)
    print("2. Listing active jobs:")
    print("-" * 40)
    try:
        asyncio.run(list_active_jobs())
    except Exception as e:
        print(f"Could not connect to Ray dashboard: {e}")
        print("Make sure Ray is running with dashboard enabled")
    
    print("\nFor more examples, see the test files and documentation.")
