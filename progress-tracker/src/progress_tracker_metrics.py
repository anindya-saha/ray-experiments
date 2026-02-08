import ray
import time
import logging
from typing import Dict, Any
from ray.util.metrics import Counter, Gauge

logger = logging.getLogger(__name__)


@ray.remote(concurrency_groups={"writes": 1, "reads": 1})
class ProgressTracker:
    """Actor to track success/failure counts across distributed workers.

    Uses concurrency groups to run increment() and get_counts() on separate
    threads. All increment() calls are serialized in the "writes" group so
    no updates are lost. get_counts() runs in the "reads" group so it is
    never starved behind queued writes.
    Ref: https://docs.ray.io/en/latest/ray-core/actors/concurrency_group_api.html
    """

    def __init__(self):
        self.success = 0
        self.failure = 0
        self.start_time = time.time()
        self.last_update_time = self.start_time

        # Ray metrics — exported to the Ray dashboard                                                  
        self._success_counter = Counter(                                                               
            "progress_tracker_success_total",                                                          
            description="Total successfully processed records",                                        
        )                                                                                              
        self._failure_counter = Counter(                                                               
            "progress_tracker_failure_total",                                                          
            description="Total failed records",                                                        
        )                                                                                              
        self._throughput_gauge = Gauge(                                                                
            "progress_tracker_records_per_sec",                                                        
            description="Current processing throughput (records/sec)",                                 
        )

    @ray.method(concurrency_group="writes")
    def increment(self, success: int = 0, failure: int = 0):
        self.success += success
        self.failure += failure
        self.last_update_time = time.time()

        # Update Ray metrics
        if success:                                                                                    
            self._success_counter.inc(success)                                                         
        if failure:                                                                                    
            self._failure_counter.inc(failure)                                                         
        elapsed = self.last_update_time - self.start_time                                              
        total = self.success + self.failure                                                            
        rate = total / elapsed if elapsed > 0 else 0                                                   
        self._throughput_gauge.set(rate) 

    @ray.method(concurrency_group="reads")
    def get_counts(self) -> Dict[str, Any]:
        elapsed = self.last_update_time - self.start_time
        total = self.success + self.failure
        rate = total / elapsed if elapsed > 0 else 0
        return {
            "success": self.success,
            "failure": self.failure,
            "total": total,
            "elapsed_sec": round(elapsed, 1),
            "records_per_sec": round(rate, 1),
        }


def process_batch(batch: Dict[str, Any], tracker) -> Dict[str, Any]:
    """Example batch processor - replace with your logic."""
    success, failure = 0, 0
    results = []

    for item in batch["data"]:
        try:
            # === Your processing logic here ===
            processed = item * 2  # Example transformation
            results.append(processed)
            success += 1
        except Exception:
            results.append(None)
            failure += 1

    # Update tracker (fire-and-forget)
    tracker.increment.remote(success=success, failure=failure)

    return {"data": results}


# =============================================================================
# Main execution
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        ray.init()

        # Create tracker actor
        tracker = ProgressTracker.remote()

        # Create sample dataset (replace with your actual data source)
        ds = ray.data.from_items([{"data": list(range(1000))} for _ in range(100000)])

        # Process the dataset
        result_ds = ds.map_batches(
            process_batch,
            fn_kwargs={"tracker": tracker},
            batch_size=100,
        )

        # Trigger execution (write to sink or consume)
        result_ds.write_parquet("~/ray-experiments/progress-tracker/output.parquet")

    finally:

        if ray.is_initialized():
            ray.shutdown()
