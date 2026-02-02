#!/usr/bin/env python3
"""
Command-line tool to analyze Ray job logs
Usage: python analyze_ray_logs.py < logfile.txt
       or: tail -f ray_job.log | python analyze_ray_logs.py
"""

import sys
import json
import argparse
from ray_job_analyzer_standalone import RayLogAnalyzer


def format_progress_bar(percent: float, width: int = 20) -> str:
    """Create a text progress bar"""
    filled = int(percent / 100 * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}]"


def print_status(analyzer: RayLogAnalyzer, clear_screen: bool = True):
    """Print formatted status to console"""
    if clear_screen:
        # Clear screen and move cursor to top
        print("\033[2J\033[H", end="")
    
    summary = analyzer.get_summary()
    
    # Header
    print("=" * 80)
    print("RAY JOB MONITOR".center(80))
    print("=" * 80)
    print()
    
    # Job Status
    job = summary['job_summary']
    status_color = {
        "running": "\033[33m",  # Yellow
        "completed": "\033[32m",  # Green
        "failed": "\033[31m",  # Red
        "unknown": "\033[90m"  # Gray
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
    print("-" * 40)
    
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
        if rs['pending_actors'] > 0:
            print(f"  Actors: {rs['pending_actors']}")
        if rs['pending_tasks'] > 0:
            print(f"  Tasks: {rs['pending_tasks']}")
    print()
    
    # Pipeline Progress
    if summary['pipeline_stages']:
        print("PIPELINE PROGRESS:")
        print("-" * 40)
        for stage_name, stage in summary['pipeline_stages'].items():
            print(f"\n{stage_name}:")
            progress_bar = format_progress_bar(stage['progress_percent'])
            print(f"  {progress_bar} {stage['progress_percent']:>5.1f}%")
            print(f"  Items: {stage['processed_items']:,}/{stage['total_items']:,}")
            print(f"  Speed: {stage['throughput']:.2f} {stage['throughput_unit']}")
        print()
    
    # vLLM Metrics
    if summary['vllm_metrics']:
        print("vLLM ENGINE METRICS:")
        print("-" * 40)
        for engine_id, metrics in summary['vllm_metrics'].items():
            print(f"\nEngine {engine_id}:")
            print(f"  Prompt:     {metrics['prompt_throughput']:>8.1f} tokens/s")
            print(f"  Generation: {metrics['generation_throughput']:>8.1f} tokens/s")
            print(f"  Requests:   {metrics['running_requests']} running, {metrics['waiting_requests']} waiting")
            print(f"  GPU Cache:  {metrics['gpu_kv_cache_usage']:.1f}% used")
            print(f"  Cache Hits: {metrics['prefix_cache_hit_rate']:.1f}%")
        print()
    
    # Errors/Warnings
    if job['errors']:
        print(f"\033[31m❌ ERRORS ({len(job['errors'])})\033[0m")
        for error in job['errors'][-3:]:  # Show last 3 errors
            print(f"  {error[:100]}...")
        print()
    
    if job['warnings']:
        print(f"\033[33m⚠️  WARNINGS ({len(job['warnings'])})\033[0m")
        for warning in job['warnings'][-3:]:  # Show last 3 warnings
            print(f"  {warning[:100]}...")
        print()


def main():
    parser = argparse.ArgumentParser(description="Analyze Ray job logs")
    parser.add_argument("--json", action="store_true", help="Output JSON format")
    parser.add_argument("--no-clear", action="store_true", help="Don't clear screen")
    parser.add_argument("--file", type=str, help="Read from file instead of stdin")
    args = parser.parse_args()
    
    analyzer = RayLogAnalyzer()
    
    # Read from file or stdin
    if args.file:
        with open(args.file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            analyzer.analyze_log_line(line.strip())
        
        if args.json:
            print(json.dumps(analyzer.get_summary(), indent=2))
        else:
            print_status(analyzer, clear_screen=False)
    else:
        # Read from stdin (streaming mode)
        try:
            for line in sys.stdin:
                analyzer.analyze_log_line(line.strip())
                
                if not args.json:
                    print_status(analyzer, clear_screen=not args.no_clear)
        except KeyboardInterrupt:
            # Final summary on exit
            if args.json:
                print(json.dumps(analyzer.get_summary(), indent=2))
            else:
                print("\n\nFinal Summary:")
                print_status(analyzer, clear_screen=False)


if __name__ == "__main__":
    main()
