#!/usr/bin/env python3
"""
Enhanced command-line tool to analyze Ray job logs
Supports both REST API monitoring and log file analysis

Usage: 
  # Monitor running job via API
  python analyze_ray_logs_enhanced.py --api --job-id raysubmit_abc123
  
  # Monitor all running jobs
  python analyze_ray_logs_enhanced.py --api --status running
  
  # Analyze log file
  python analyze_ray_logs_enhanced.py --file ray_job.log
  
  # Stream logs in real-time
  tail -f ray_job.log | python analyze_ray_logs_enhanced.py
"""

import sys
import json
import argparse
import asyncio
import time
from datetime import datetime
from typing import Optional, Dict, Any
import aiohttp

from ray_job_analyzer_standalone import RayLogAnalyzer


class RayAPIMonitor:
    """Monitor Ray jobs via REST API"""
    
    def __init__(self, dashboard_url: str = "http://localhost:8265"):
        self.dashboard_url = dashboard_url
    
    async def list_jobs(self, status_filter: Optional[str] = None) -> list:
        """List jobs, optionally filtered by status"""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{self.dashboard_url}/api/jobs/") as resp:
                    if resp.status == 200:
                        jobs = await resp.json()
                        if status_filter:
                            jobs = [j for j in jobs if j.get("status", "").lower() == status_filter.lower()]
                        return jobs
            except Exception as e:
                print(f"Error connecting to Ray Dashboard: {e}")
        return []
    
    async def get_job_details(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job details"""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{self.dashboard_url}/api/jobs/{job_id}") as resp:
                    if resp.status == 200:
                        return await resp.json()
            except Exception as e:
                print(f"Error getting job details: {e}")
        return None
    
    async def get_job_logs(self, job_id: str) -> Optional[str]:
        """Get job logs"""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{self.dashboard_url}/api/jobs/{job_id}/logs") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("logs", "")
            except Exception as e:
                print(f"Error getting job logs: {e}")
        return None


def format_progress_bar(percent: float, width: int = 20) -> str:
    """Create a text progress bar"""
    filled = int(percent / 100 * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}]"


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds//60:.0f}m {seconds%60:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours:.0f}h {minutes:.0f}m"


def print_job_list(jobs: list):
    """Print formatted job list"""
    print("="*80)
    print("RAY JOBS".center(80))
    print("="*80)
    print()
    
    if not jobs:
        print("No jobs found.")
        return
    
    # Group by status
    by_status = {}
    for job in jobs:
        status = job.get("status", "UNKNOWN")
        if status not in by_status:
            by_status[status] = []
        by_status[status].append(job)
    
    # Print each status group
    for status in ["RUNNING", "PENDING", "SUCCEEDED", "FAILED", "STOPPED"]:
        if status in by_status:
            color = {
                "RUNNING": "\033[33m",    # Yellow
                "PENDING": "\033[36m",    # Cyan
                "SUCCEEDED": "\033[32m",  # Green
                "FAILED": "\033[31m",     # Red
                "STOPPED": "\033[90m"     # Gray
            }.get(status, "")
            
            print(f"{color}{status}\033[0m ({len(by_status[status])} jobs):")
            print("-"*40)
            
            for job in by_status[status][:5]:  # Show max 5 per status
                job_id = job.get("job_id", "unknown")
                entrypoint = job.get("entrypoint", "")
                start_time = job.get("start_time")
                
                print(f"  {job_id}")
                print(f"    {entrypoint}")
                
                if start_time:
                    start_dt = datetime.fromtimestamp(start_time)
                    print(f"    Started: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    if status == "RUNNING":
                        duration = time.time() - start_time
                        print(f"    Running for: {format_duration(duration)}")
                    elif job.get("end_time"):
                        duration = job["end_time"] - start_time
                        print(f"    Duration: {format_duration(duration)}")
                
                if job.get("error_message"):
                    print(f"    \033[31mError: {job['error_message'][:60]}...\033[0m")
                
                print()
            
            if len(by_status[status]) > 5:
                print(f"  ... and {len(by_status[status]) - 5} more\n")


def print_job_monitor(job_details: Dict[str, Any], analyzer: RayLogAnalyzer):
    """Print formatted job monitoring output"""
    print("\033[2J\033[H", end="")  # Clear screen
    
    print("="*80)
    print(f"RAY JOB MONITOR - {job_details['job_id']}".center(80))
    print("="*80)
    print()
    
    # Job info
    status = job_details.get("status", "UNKNOWN")
    status_color = {
        "RUNNING": "\033[33m",
        "SUCCEEDED": "\033[32m",
        "FAILED": "\033[31m",
        "PENDING": "\033[36m"
    }.get(status, "")
    
    print(f"Status: {status_color}{status}\033[0m")
    print(f"Entrypoint: {job_details.get('entrypoint', 'N/A')}")
    
    if job_details.get("start_time"):
        start_dt = datetime.fromtimestamp(job_details["start_time"])
        print(f"Started: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if status == "RUNNING":
            duration = time.time() - job_details["start_time"]
            print(f"Duration: {format_duration(duration)}")
    
    if job_details.get("error_message"):
        print(f"\n\033[31mError: {job_details['error_message']}\033[0m")
    
    # Analysis from logs
    summary = analyzer.get_summary()
    job_sum = summary['job_summary']
    
    if job_sum['total_samples'] > 0:
        print(f"\nSamples Processed: {job_sum['total_samples']:,}")
        print(f"Throughput: {job_sum['overall_throughput']:.2f} samples/sec")
    
    # Resource status
    rs = summary['resource_status']
    if rs['cpus_total'] > 0:
        print("\nRESOURCE ALLOCATION:")
        print("-"*40)
        
        cpu_percent = rs['cpus_used'] / rs['cpus_total'] * 100
        print(f"CPU:  {rs['cpus_used']:>5.1f}/{rs['cpus_total']:<5.0f} {format_progress_bar(cpu_percent)} {cpu_percent:>5.1f}%")
        
        if rs['gpus_total'] > 0:
            gpu_percent = rs['gpus_used'] / rs['gpus_total'] * 100
            print(f"GPU:  {rs['gpus_used']:>5.1f}/{rs['gpus_total']:<5.0f} {format_progress_bar(gpu_percent)} {gpu_percent:>5.1f}%")
    
    # Pipeline progress
    if summary['pipeline_stages']:
        print("\nPIPELINE PROGRESS:")
        print("-"*40)
        for stage_name, stage in list(summary['pipeline_stages'].items())[:3]:  # Show top 3
            print(f"\n{stage_name}:")
            print(f"  {format_progress_bar(stage['progress_percent'])}  {stage['progress_percent']:.1f}%")
            print(f"  Items: {stage['processed_items']}/{stage['total_items']}")
            if stage['throughput'] > 0:
                print(f"  Speed: {stage['throughput']:.2f} {stage['throughput_unit']}")
    
    # vLLM metrics
    if summary['vllm_metrics']:
        print("\nvLLM ENGINE METRICS:")
        print("-"*40)
        for engine_id, metrics in list(summary['vllm_metrics'].items())[:2]:  # Show top 2
            print(f"\nEngine {engine_id}:")
            print(f"  Prompt:     {metrics['prompt_throughput']:>7.1f} tokens/s")
            print(f"  Generation: {metrics['generation_throughput']:>7.1f} tokens/s")
            print(f"  Requests:   {metrics['running_requests']} running, {metrics['waiting_requests']} waiting")
            if metrics['gpu_kv_cache_usage'] > 0:
                print(f"  GPU Cache:  {metrics['gpu_kv_cache_usage']:.1f}% used")


def print_log_analysis(analyzer: RayLogAnalyzer, clear_screen: bool = True):
    """Print formatted log analysis (original functionality)"""
    if clear_screen:
        print("\033[2J\033[H", end="")
    
    summary = analyzer.get_summary()
    
    # Header
    print("="*80)
    print("RAY LOG ANALYZER".center(80))
    print("="*80)
    print()
    
    # Job Status
    job = summary['job_summary']
    status_color = {
        "running": "\033[33m",
        "completed": "\033[32m",
        "failed": "\033[31m",
        "unknown": "\033[90m"
    }.get(job['status'], "")
    
    print(f"Job Status: {status_color}{job['status'].upper()}\033[0m")
    if job['total_samples'] > 0:
        print(f"Total Samples: {job['total_samples']:,}")
    if job['overall_throughput'] > 0:
        print(f"Overall Throughput: {job['overall_throughput']:.2f} samples/sec")
    print()
    
    # Resource Status
    rs = summary['resource_status']
    print("RESOURCE ALLOCATION:")
    print("-"*40)
    
    if rs['cpus_total'] > 0:
        cpu_percent = rs['cpus_used'] / rs['cpus_total'] * 100
        print(f"CPU:  {rs['cpus_used']:>5.1f}/{rs['cpus_total']:<5.0f} {format_progress_bar(cpu_percent)} {cpu_percent:>5.1f}%")
    
    if rs['gpus_total'] > 0:
        gpu_percent = rs['gpus_used'] / rs['gpus_total'] * 100
        print(f"GPU:  {rs['gpus_used']:>5.1f}/{rs['gpus_total']:<5.0f} {format_progress_bar(gpu_percent)} {gpu_percent:>5.1f}%")
    
    if rs['pending_cpu_demand'] > 0 or rs['pending_gpu_demand'] > 0:
        print("\n⚠️  PENDING DEMANDS:")
        if rs['pending_cpu_demand'] > 0:
            print(f"  CPU: {rs['pending_cpu_demand']}")
        if rs['pending_gpu_demand'] > 0:
            print(f"  GPU: {rs['pending_gpu_demand']}")
    print()
    
    # Pipeline Progress
    if summary['pipeline_stages']:
        print("PIPELINE PROGRESS:")
        print("-"*40)
        for stage_name, stage in summary['pipeline_stages'].items():
            print(f"\n{stage_name}:")
            print(f"  {format_progress_bar(stage['progress_percent'])}  {stage['progress_percent']:.1f}%")
            print(f"  Items: {stage['processed_items']}/{stage['total_items']}")
            if stage['throughput'] > 0:
                print(f"  Speed: {stage['throughput']:.2f} {stage['throughput_unit']}")
    
    # vLLM Metrics
    if summary['vllm_metrics']:
        print("\nvLLM ENGINE METRICS:")
        print("-"*40)
        for engine_id, metrics in summary['vllm_metrics'].items():
            print(f"\nEngine {engine_id}:")
            print(f"  Prompt:     {metrics['prompt_throughput']:>7.1f} tokens/s")
            print(f"  Generation: {metrics['generation_throughput']:>7.1f} tokens/s")
            print(f"  Requests:   {metrics['running_requests']} running, {metrics['waiting_requests']} waiting")
            print(f"  GPU Cache:  {metrics['gpu_kv_cache_usage']:.1f}% used")
            print(f"  Cache Hits: {metrics['prefix_cache_hit_rate']:.1f}%")
    
    # Errors and Warnings
    if job['errors'] or job['warnings']:
        print("\nISSUES DETECTED:")
        print("-"*40)
        if job['errors']:
            print(f"❌ Errors: {len(job['errors'])}")
            for error in job['errors'][:3]:
                print(f"   - {error[:70]}...")
        if job['warnings']:
            print(f"⚠️  Warnings: {len(job['warnings'])}")
            for warning in job['warnings'][:3]:
                print(f"   - {warning[:70]}...")


async def monitor_job_api(api_monitor: RayAPIMonitor, job_id: str, interval: int = 5):
    """Monitor a specific job via API"""
    analyzer = RayLogAnalyzer()
    
    while True:
        job_details = await api_monitor.get_job_details(job_id)
        if not job_details:
            print(f"Error: Could not find job {job_id}")
            break
        
        # Get logs and analyze
        logs = await api_monitor.get_job_logs(job_id)
        if logs:
            analyzer = RayLogAnalyzer()  # Fresh analysis each time
            for line in logs.split('\n'):
                if line.strip():
                    analyzer.analyze_log_line(line)
        
        # Display
        print_job_monitor(job_details, analyzer)
        
        # Check if job is done
        if job_details.get("status") in ["SUCCEEDED", "FAILED", "STOPPED"]:
            print(f"\n\nJob {job_details['status']}. Monitoring stopped.")
            break
        
        # Wait before next update
        await asyncio.sleep(interval)


async def main_async(args):
    """Async main function for API operations"""
    api_monitor = RayAPIMonitor(args.dashboard_url)
    
    if args.job_id:
        # Monitor specific job
        await monitor_job_api(api_monitor, args.job_id, args.interval)
    else:
        # List jobs
        jobs = await api_monitor.list_jobs(args.status)
        if args.json:
            print(json.dumps(jobs, indent=2))
        else:
            print_job_list(jobs)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Enhanced Ray job log analyzer with REST API support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Monitor running job via API
  %(prog)s --api --job-id raysubmit_abc123
  
  # List all running jobs
  %(prog)s --api --status running
  
  # Analyze log file
  %(prog)s --file ray_job.log
  
  # Stream logs from stdin
  tail -f ray_job.log | %(prog)s
  
  # JSON output
  %(prog)s --api --status running --json
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--api", action="store_true", 
                           help="Use REST API to monitor jobs")
    mode_group.add_argument("--file", type=str,
                           help="Analyze a log file")
    
    # API options
    parser.add_argument("--dashboard-url", default="http://localhost:8265",
                       help="Ray dashboard URL (default: http://localhost:8265)")
    parser.add_argument("--job-id", type=str,
                       help="Monitor specific job ID (API mode)")
    parser.add_argument("--status", type=str, choices=["running", "pending", "succeeded", "failed", "all"],
                       help="Filter jobs by status (API mode)")
    parser.add_argument("--interval", type=int, default=5,
                       help="Update interval in seconds for monitoring (default: 5)")
    
    # Output options
    parser.add_argument("--json", action="store_true",
                       help="Output in JSON format")
    parser.add_argument("--no-clear", action="store_true",
                       help="Don't clear screen between updates")
    
    args = parser.parse_args()
    
    # Handle API mode
    if args.api:
        try:
            asyncio.run(main_async(args))
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")
        return
    
    # Handle file mode or stdin (original functionality)
    analyzer = RayLogAnalyzer()
    
    if args.file:
        # Read from file
        try:
            with open(args.file, 'r') as f:
                for line in f:
                    analyzer.analyze_log_line(line.strip())
        except FileNotFoundError:
            print(f"Error: File '{args.file}' not found")
            sys.exit(1)
        
        # Print final analysis
        if args.json:
            print(json.dumps(analyzer.get_summary(), indent=2))
        else:
            print_log_analysis(analyzer, clear_screen=False)
    else:
        # Stream from stdin
        try:
            while True:
                line = sys.stdin.readline()
                if not line:
                    break
                
                analyzer.analyze_log_line(line.strip())
                
                if not args.json:
                    print_log_analysis(analyzer, clear_screen=not args.no_clear)
        except KeyboardInterrupt:
            pass
        
        # Final output
        if args.json:
            print(json.dumps(analyzer.get_summary(), indent=2))


if __name__ == "__main__":
    main()
