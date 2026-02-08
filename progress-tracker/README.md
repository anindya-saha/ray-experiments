# Ray Job Progress Tracker

## 1. Console logging with `ProgressReporter` (`progress_tracker.py`)

A `ProgressReporter` background thread polls the `ProgressTracker` actor at a configurable interval and logs progress to the console. The actor tracks counts internally; the reporter reads them via `get_counts()`.

Track success/failure counts and throughput across distributed Ray Data `map_batches` workers.

### Architecture

```
┌─────────────────────┐       increment()         ┌──────────────────────┐
│  map_batches worker │ ──── fire-and-forget ──►  │                      │
├─────────────────────┤                           │   ProgressTracker    │
│  map_batches worker │ ──── fire-and-forget ──►  │   (Ray actor)        │
├─────────────────────┤                           │                      │
│  map_batches worker │ ──── fire-and-forget ──►  │   success, failure,  │
└─────────────────────┘                           │   timestamps         │
                                                  └──────────┬───────────┘
                                                             │
                                                  get_counts() every N sec
                                                             │
                                                             v
                                                  ┌──────────────────────┐
                                                  │  ProgressReporter    │
                                                  │  (background thread) │
                                                  │  logs to console     │
                                                  └──────────────────────┘
```

There are three components:

- **`ProgressTracker`** — a Ray actor that accumulates success/failure counters. Workers call `increment()` as fire-and-forget (`tracker.increment.remote()`), so processing is never blocked by progress bookkeeping.

- **`ProgressReporter`** — a local background thread that polls the tracker at a configurable interval and logs progress (count, percentage, throughput, elapsed time).

- **`process_batch`** — a `map_batches`-compatible function that processes each batch and reports results back to the tracker.

### Design Decisions

**Concurrency groups for reads and writes**

By default, Ray actors process method calls one at a time in FIFO order. When `map_batches` workers flood the actor with `increment()` calls, a `get_counts()` call from the reporter gets queued behind them and cannot be served until the backlog clears. This starves the reporter — it only sees results at the very end.

A naive fix like `max_concurrency=2` gives the actor a thread pool of 2 threads, but Ray makes no guarantee about which methods land on which threads. Two `increment()` calls could run simultaneously, and since `+=` is a read-modify-write across multiple bytecodes, concurrent increments would lose updates.

The solution is Ray's **concurrency groups**, which pin methods to dedicated thread pools:

```python
@ray.remote(concurrency_groups={"writes": 1, "reads": 1})
class ProgressTracker:
    @ray.method(concurrency_group="writes")
    def increment(self, ...): ...

    @ray.method(concurrency_group="reads")
    def get_counts(self): ...
```

- **"writes" group (1 thread)** — all `increment()` calls from all workers are serialized through a single thread. No matter how many workers call `increment.remote()` concurrently, the calls queue within this group and execute one at a time. No updates are lost.
- **"reads" group (1 thread)** — `get_counts()` runs on its own dedicated thread, completely independent of the writes queue. The reporter is never starved.

The two groups run concurrently (separate thread pools), so reads and writes overlap freely. But writes among themselves are always sequential — exactly the guarantee we need.


**Logging**

Ray Data prints its own progress bars and status lines to the console. Using Python `logging` outputs on separate lines and coexists cleanly with Ray's console output.


**How to run it**

Setup

```bash
uv sync
```
Run

```bash
uv run python src/progress_tracker.py
```

```bash
python3 src/progress_tracker.py 
2026-02-08 00:40:19,675 INFO worker.py:1998 -- Started a local Ray instance. View the dashboard at http://127.0.0.1:8265 
/home/asaha/ray-experiments/progress-tracker/.venv/lib/python3.12/site-packages/ray/_private/worker.py:2046: FutureWarning: Tip: In future versions of Ray, Ray will no longer override accelerator visible devices env var if num_gpus=0 or num_gpus=None (default). To enable this behavior and turn off this error message, set RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
  warnings.warn(
2026-02-08 00:40:33,134 INFO logging.py:397 -- Registered dataset logger for dataset dataset_3_0
2026-02-08 00:40:33,143 INFO streaming_executor.py:178 -- Starting execution of Dataset dataset_3_0. Full logs are in /tmp/ray/session_2026-02-08_00-40-17_130068_2917011/logs/ray-data
2026-02-08 00:40:33,143 INFO streaming_executor.py:179 -- Execution plan of Dataset dataset_3_0: InputDataBuffer[Input] -> TaskPoolMapOperator[MapBatches(process_batch)->Write]
2026-02-08 00:40:33,143 INFO streaming_executor.py:686 -- [dataset]: A new progress UI is available. To enable, set `ray.data.DataContext.get_current().enable_rich_progress_bars = True` and `ray.data.DataContext.get_current().use_ray_tqdm = False`.
[dataset]: Run `pip install tqdm` to enable progress reporting.
2026-02-08 00:40:33,144 WARNING resource_manager.py:136 -- ⚠️  Ray's object store is configured to use only 42.9% of available memory (39.9GiB out of 93.2GiB total). For optimal Ray Data performance, we recommend setting the object store to at least 50% of available memory. You can do this by setting the 'object_store_memory' parameter when calling ray.init() or by setting the RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION environment variable.
2026-02-08 00:40:33,260 INFO __main__: Progress: 0/100,000 (0.0%) | success: 0 | failure: 0 | 0 rec/s | 0.0s elapsed
2026-02-08 00:40:33,763 INFO __main__: Progress: 0/100,000 (0.0%) | success: 0 | failure: 0 | 0 rec/s | 0.0s elapsed
2026-02-08 00:40:34,269 INFO __main__: Progress: 49,800/100,000 (49.8%) | success: 49,800 | failure: 0 | 3,658 rec/s | 13.6s elapsed
2026-02-08 00:40:34,679 INFO streaming_executor.py:304 -- ✔️  Dataset dataset_3_0 execution finished in 1.54 seconds
2026-02-08 00:40:34,827 INFO __main__: Progress: 100,000/100,000 (100.0%) | success: 100,000 | failure: 0 | 7,148 rec/s | 14.0s elapsed
2026-02-08 00:40:34,827 INFO dataset.py:5344 -- Data sink Parquet finished. 100000 rows and 763.7MiB data written.

--------------------------------------------------
Final: 100,000 total | 100,000 success | 0 failure
Time: 14.0s | Rate: 7,148 rec/s
--------------------------------------------------
```


## 2. Ray dashboard metrics (`progress_tracker_metrics.py`)

Instead of polling and logging, the `ProgressTracker` actor exports metrics directly via `ray.util.metrics`. These are scraped automatically and visible on the Ray dashboard (Metrics tab) at `http://127.0.0.1:8265`.

| Metric | Type | Description |
|---|---|---|
| `progress_tracker_success_total` | Counter | Cumulative successful records |
| `progress_tracker_failure_total` | Counter | Cumulative failed records |
| `progress_tracker_records_per_sec` | Gauge | Current throughput (records/sec) |

No `ProgressReporter` thread is needed — the dashboard provides the observability. Metrics are updated inside `increment()` on the "writes" thread, so they stay consistent with the counters.

**How to run it**


```bash
uv run python src/progress_tracker_metrics.py
```

