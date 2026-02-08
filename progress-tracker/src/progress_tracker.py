import ray
import time
import logging
import threading
from typing import Dict, Any, Optional

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

    @ray.method(concurrency_group="writes")
    def increment(self, success: int = 0, failure: int = 0):
        self.success += success
        self.failure += failure
        self.last_update_time = time.time()

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


class ProgressReporter:
    """Background thread that polls the tracker and logs progress."""

    def __init__(self, tracker, interval: float = 2.0, expected_total: Optional[int] = None):
        self.tracker = tracker
        self.interval = interval
        self.expected_total = expected_total
        self._stop_event = threading.Event()
        self._thread = None

    def _poll_loop(self):
        while not self._stop_event.is_set():
            try:
                counts = ray.get(self.tracker.get_counts.remote())
                self._print_progress(counts)
            except Exception as e:
                logger.warning(f"Progress polling error: {e}")
            time.sleep(self.interval)

    def _print_progress(self, counts: Dict[str, Any]):
        total = counts["total"]
        success = counts["success"]
        failure = counts["failure"]
        rate = counts["records_per_sec"]
        elapsed = counts["elapsed_sec"]

        if self.expected_total is not None:
            pct = (total / self.expected_total) * 100
            logger.info(
                f"Progress: {total:,}/{self.expected_total:,} ({pct:.1f}%) | "
                f"success: {success:,} | failure: {failure:,} | "
                f"{rate:,.0f} rec/s | {elapsed}s elapsed"
            )
        else:
            logger.info(
                f"Processed: {total:,} | success: {success:,} | failure: {failure:,} | "
                f"{rate:,.0f} rec/s | {elapsed}s elapsed"
            )

    def start(self):
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        # Print final counts
        counts = ray.get(self.tracker.get_counts.remote())
        print()
        print("-" * 50)
        print(f"Final: {counts['total']:,} total | {counts['success']:,} success | {counts['failure']:,} failure")
        print(f"Time: {counts['elapsed_sec']}s | Rate: {counts['records_per_sec']:,.0f} rec/s")
        print("-" * 50)
        return counts


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

        # Start progress reporter
        reporter = ProgressReporter(
            tracker=tracker,
            interval=0.5,  # Print every 0.5 seconds
            expected_total=100_000,
        )
        reporter.start()

        # Process the dataset
        result_ds = ds.map_batches(
            process_batch,
            fn_kwargs={"tracker": tracker},
            batch_size=100,
        )

        # Trigger execution (write to sink or consume)
        result_ds.write_parquet("~/ray-experiments/progress-tracker/output.parquet")

    finally:
        # Always stop reporter to get final counts
        final_counts = reporter.stop()

        if ray.is_initialized():
            ray.shutdown()
