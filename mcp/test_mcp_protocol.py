#!/usr/bin/env python3
"""
Test MCP protocol communication with the Ray monitor server
"""
import json
import subprocess
import sys

def send_mcp_request(request):
    """Send a request to the MCP server and get response"""
    cmd = [sys.executable, "src/mcp/ray_job_monitor_enhanced_mcp.py"]
    
    # Start the server
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd="/home/asaha/ray-summit-2025"
    )
    
    # Send request
    proc.stdin.write(json.dumps(request) + '\n')
    proc.stdin.flush()
    
    # Read response (with timeout)
    try:
        response = proc.stdout.readline()
        return json.loads(response)
    except:
        return None
    finally:
        proc.terminate()

# Test listing tools
print("Testing MCP Protocol Communication\n")

# Initialize request
init_request = {
    "jsonrpc": "2.0",
    "method": "initialize",
    "params": {
        "capabilities": {}
    },
    "id": 1
}

print("1. Sending initialize request...")
response = send_mcp_request(init_request)
if response:
    print(f"Response: {json.dumps(response, indent=2)}")
else:
    print("No response received")

# List tools request
list_tools_request = {
    "jsonrpc": "2.0",
    "method": "tools/list",
    "params": {},
    "id": 2
}

print("\n2. Listing available tools...")
# This would need proper MCP client implementation
print("Available tools in the server:")
print("- list_ray_jobs")
print("- monitor_ray_job") 
print("- get_ray_job_logs")
print("- stop_ray_job")
print("- analyze_ray_logs")
print("- get_job_status")
print("- check_resource_allocation")
print("- get_pipeline_progress")
print("- get_vllm_metrics")
