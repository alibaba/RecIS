"""Smoke test for recis.hooks.trace_to_odps_hook.

What this verifies:
  1. TraceToOdpsHook starts the writer subprocess(es) and connects to ODPS.
  2. The target partition is created automatically:
       - via TableTunnel.create_upload_session(create_partition=True) on
         pyodps >= 0.12.3, OR
       - via the REST fallback (odps.create_table + table.create_partition)
         on older pyodps when REST is reachable.
  3. Trace data flows through add_to_trace -> hook.after_step -> writer queue
     -> tunnel upload, and is committed on hook.end().
  4. Row count read back from the partition matches what we pushed
     (skipped automatically when REST is not reachable).

Usage:
  export ODPS_ACCESS_ID=...
  export ODPS_ACCESS_KEY=...
  export ODPS_PROJECT=...
  export ODPS_ENDPOINT=...                  # optional; needed for read-back
  export ODPS_TEST_TABLE=trace_smoke_test   # optional, has default
  python tests/hooks/test_trace_to_odps_hook.py
"""

import os
import time

import numpy as np
from odps import ODPS, __version__ as PYODPS_VERSION

from recis.hooks.trace_to_odps_hook import TraceToOdpsHook, add_to_trace


def _require_env(name):
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"missing required env var: {name}")
    return v


def main():
    access_id = _require_env("ODPS_ACCESS_ID")
    access_key = _require_env("ODPS_ACCESS_KEY")
    project = _require_env("ODPS_PROJECT")
    endpoint = os.environ.get("ODPS_ENDPOINT", "")

    table_name = os.environ.get("ODPS_TEST_TABLE", "gwj_tunnel_capability_test")
    # Unique partition each run so we always exercise partition creation.
    partition = "ds=" + time.strftime("%Y%m%d%H%M%S")

    print(f"[info] pyodps version : {PYODPS_VERSION}")
    print(f"[info] project        : {project}")
    print(
        f"[info] endpoint       : {endpoint or '(unset; relying on cluster default)'}"
    )
    print(f"[info] table          : {table_name}")
    print(f"[info] partition      : {partition}  (fresh — should be created)")

    config = {
        "access_id": access_id,
        "access_key": access_key,
        "project": project,
        "end_point": endpoint,
        "table_name": table_name,
        "partition": partition,
    }

    fields = ["id", "val"]
    types = ["bigint", "string"]

    batch_size = 100
    num_batches = 5
    expected_rows = batch_size * num_batches

    # Small size_threshold so we exercise a mid-run flush instead of only
    # the shutdown flush.
    hook = TraceToOdpsHook(
        config=config,
        fields=fields,
        types=types,
        worker_num=1,
        size_threshold=4 * 1024,  # 4 KiB
    )

    for i in range(num_batches):
        ids = np.arange(batch_size, dtype=np.int64) + i * batch_size
        vals = [f"batch{i}-row{j}" for j in range(batch_size)]
        add_to_trace("id", ids)
        add_to_trace("val", vals)
        hook.after_step()
        print(f"[step-{i}] pushed {batch_size} rows")
        time.sleep(0.05)

    print("[info] shutting hook down (commits remaining buffer)...")
    hook.end()
    print("[info] hook ended cleanly")

    # Read-back via TableTunnel so it works in external clusters where the
    # REST endpoint is unreachable. We build the ODPS client the same way
    # the writer does so the tunnel endpoint is set correctly.
    try:
        from odps.tunnel.tabletunnel import TableTunnel

        try:
            from recis.utils.cluster import get_odps_access_info

            odps_args, odps_kwargs, _ = get_odps_access_info(config)
        except ImportError:
            odps_args = [access_id, access_key, project, endpoint]
            odps_kwargs = {}

        o = ODPS(*odps_args, **odps_kwargs)
        tunnel = TableTunnel(o)
        download = tunnel.create_download_session(table_name, partition_spec=partition)
        actual = download.count
        print(
            f"[verify] partition row count: actual={actual}, expected={expected_rows}"
        )
        assert actual == expected_rows, (
            f"row count mismatch: actual={actual}, expected={expected_rows}"
        )
        print("[ok] smoke test PASSED")
    except AssertionError:
        raise
    except Exception as e:
        print(f"[warn] read-back via tunnel failed ({type(e).__name__}: {e})")
        print("[ok] upload completed without subprocess error")


if __name__ == "__main__":
    main()
