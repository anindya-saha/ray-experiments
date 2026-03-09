"""
Test: can Ray Data handle a 0-row pa.Table returned from map_batches
when other batches return tables with extra columns?

Setup:
  - 20 input rows with structural columns only
  - Actor's __call__ adds a dynamic metadata column ("weather") to
    successful extractions, but returns a 0-row table (no metadata
    columns) when the batch is deliberately "failed"
  - Batch 3 (0-indexed) is the one that "fails" and returns empty

If _align_struct_fields handles this, the pipeline completes and
writes a parquet file. If not, it crashes with an AssertionError.
"""

import pyarrow as pa
import ray


class Extractor:
    def __init__(self):
        self._call_count = 0

    def __call__(self, batch: pa.Table) -> pa.Table:
        self._call_count += 1
        batch_num = self._call_count

        chum_uris = batch.column("chum_uri").to_pylist()
        timestamps = batch.column("timestamp").to_pylist()

        if batch_num == 3:
            print(f"[Batch {batch_num}] Simulating failure -- returning 0-row table")
            schema = pa.schema([
                pa.field("chum_uri", pa.string()),
                pa.field("timestamp", pa.float64()),
            ])
            return pa.Table.from_pylist([], schema=schema)

        samples = []
        for uri, ts in zip(chum_uris, timestamps):
            samples.append({
                "chum_uri": uri,
                "timestamp": ts,
                "weather": "sunny" if ts < 5.0 else "rainy",
            })

        print(f"[Batch {batch_num}] Returning {len(samples)} rows with 'weather' column")
        return pa.Table.from_pylist(samples, schema=pa.schema([
            pa.field("chum_uri", pa.string()),
            pa.field("timestamp", pa.float64()),
            pa.field("weather", pa.string()),
        ]))


def main():
    ray.init()

    rows = [{"chum_uri": f"chum://{i}", "timestamp": float(i)} for i in range(20)]
    ds = ray.data.from_items(rows)

    result = ds.map_batches(
        Extractor,
        batch_format="pyarrow",
        batch_size=4,
        concurrency=1,
    )

    output_path = "/tmp/empty_table_test_output"
    result.write_parquet(output_path)

    print(f"\nWrote output to {output_path}")
    print(f"Reading back to verify...")

    verify = ray.data.read_parquet(output_path)
    print(f"Total rows: {verify.count()}")
    print(f"Schema: {verify.schema()}")
    print(verify.take_all())

    ray.shutdown()


if __name__ == "__main__":
    main()
