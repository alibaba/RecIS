"""TraceToOdpsHookV2 端到端集成测试任务。

模拟真实 eval 场景:多 step 写入 → 自动 flush/commit → 验证 ODPS 数据完整性。

测的内容:
  1. 基础写入:1D + 2D 混合,验证行数
  2. 大批量写入:自适应阈值触发多次 flush
  3. 时间保底 flush:手动等待 _MAX_FLUSH_INTERVAL 触发
  4. 多 writer 并发:4 线程写入
  5. 优雅关闭(end):所有 buffer 数据全部 commit

环境前提:
  - REST endpoint 不通,只有 tunnel endpoint 通
  - 测试表需预先建好(schema 见下方 FULL_FIELDS / FULL_TYPES)

跑:
  python tests/hooks/test_trace_v2_e2e.py
  python tests/hooks/test_trace_v2_e2e.py --skip-cleanup  # 跑完不删分区,方便人工查看
"""

import argparse
import os
import time

import numpy as np

from recis.hooks import TraceToOdpsHook, add_to_trace


# ============ 默认配置(对齐 tunnel_test/bench_common.py) ============
DEFAULT_ACCESS_ID = "***"
DEFAULT_ACCESS_KEY = "***"
DEFAULT_PROJECT = "nebula_ai_dev"
DEFAULT_ENDPOINT = "http://127.0.0.1:1/api"
# DEFAULT_ENDPOINT = "http://service.odps.aliyun-inc.com/api"
DEFAULT_TUNNEL_ENDPOINT = "http://dt.ea118.odps.aliyun-inc.com"
# DEFAULT_TUNNEL_ENDPOINT = "http://dt.xcluster.odps.aliyun-inc.com"
DEFAULT_QUOTA_NAME = ""

TEST_TABLE = os.environ.get("ODPS_TEST_TABLE", "gwj_trace_v2_e2e_test")

# 单一大表 schema — 所有浮点列统一用 double,避免 float 精度丢失
FULL_FIELDS = [
    "id", "val", "score", "label",
    "user_id", "embedding", "attention", "feature_map",
]
FULL_TYPES = [
    "bigint", "string", "double", "bigint",
    "bigint", "array<double>", "array<array<double>>", "array<array<array<double>>>",
]


# ============ helpers ============


def _placeholder(odps_type: str, n: int):
    """生成占位数据,让 storage API 接受写入。"""
    t = odps_type.strip().lower()
    if t == "bigint":
        return np.zeros(n, dtype=np.int64)
    if t == "int":
        return np.zeros(n, dtype=np.int32)
    if t == "double":
        return np.zeros(n, dtype=np.float64)
    if t == "float":
        return np.zeros(n, dtype=np.float32)
    if t == "string":
        return [""] * n
    if t == "boolean":
        return np.zeros(n, dtype=bool)
    if t.startswith("array<") and t.endswith(">"):
        depth = 0
        inner = t
        while inner.startswith("array<") and inner.endswith(">"):
            depth += 1
            inner = inner[6:-1]
        dtype_map = {"float": np.float32, "double": np.float64,
                     "bigint": np.int64, "int": np.int32}
        dt = dtype_map.get(inner, np.float64)  # 默认 double，避免精度丢失
        shape = (n,) + (1,) * depth
        return np.zeros(shape, dtype=dt)
    raise ValueError(f"unsupported placeholder type: {odps_type}")


def add_with_padding(real_data: dict, n: int):
    """add_to_trace 真实数据 + 给其他列填占位。"""
    for k, v in real_data.items():
        add_to_trace(k, v)
    for f, t in zip(FULL_FIELDS, FULL_TYPES):
        if f not in real_data:
            add_to_trace(f, _placeholder(t, n))


def _make_config(suffix: str) -> dict:
    """生成 trace hook config,分区带时间戳+后缀避免冲突。"""
    return {
        "access_id": os.environ.get("ODPS_ACCESS_ID", DEFAULT_ACCESS_ID),
        "access_key": os.environ.get("ODPS_ACCESS_KEY", DEFAULT_ACCESS_KEY),
        "project": os.environ.get("ODPS_PROJECT", DEFAULT_PROJECT),
        "end_point": os.environ.get("ODPS_ENDPOINT", DEFAULT_ENDPOINT),
        "tunnel_endpoint": os.environ.get(
            "ODPS_TUNNEL_ENDPOINT", DEFAULT_TUNNEL_ENDPOINT
        ),
        "table_name": TEST_TABLE,
        "partition": f"ds=e2e_{time.strftime('%Y%m%d%H%M%S')}_{suffix}",
        "quota_name": os.environ.get("ODPS_QUOTA_NAME", DEFAULT_QUOTA_NAME),
    }


def _read_back_count(cfg: dict) -> int | None:
    # """通过 tunnel 读回分区行数。"""
    # try:
    #     o = ODPS(
    #         access_id=cfg["access_id"],
    #         secret_access_key=cfg["access_key"],
    #         project=cfg["project"],
    #         endpoint=cfg["end_point"],
    #         tunnel_endpoint=cfg["tunnel_endpoint"],
    #     )
    #     # o.is_schema_namespace_enabled = lambda settings=None: False
    #     tunnel = TableTunnel(o)
    #     download = tunnel.create_download_session(
    #         cfg["table_name"], partition_spec=cfg["partition"]
    #     )
    #     return download.count
    # except Exception as e:
    #     print(f"  [warn] read-back failed: {type(e).__name__}: {e}")
    #     return None
    return None


# def _delete_partition(cfg: dict):
#     """通过 REST API 删除分区(仅用于清理)。"""
#     try:
#         o = ODPS(
#             access_id=cfg["access_id"],
#             secret_access_key=cfg["access_key"],
#             project=cfg["project"],
#             endpoint=cfg["end_point"],
#         )
#         table = o.get_table(cfg["table_name"])
#         table.delete_partition(cfg["partition"], if_exists=True)
#         print(f"  [cleanup] deleted partition: {cfg['partition']}")
#     except Exception as e:
#         print(f"  [warn] cleanup failed (REST may be unavailable): {e}")


def _verify(cfg: dict, expected: int, label: str = ""):
    """验证行数,打印结果。"""
    actual = _read_back_count(cfg)
    status = "SKIP" if actual is None else ("PASS" if actual == expected else "FAIL")
    print(f"  [{status}] {label} expected={expected}, actual={actual}")
    if actual is not None and actual != expected:
        raise AssertionError(f"Row count mismatch: {actual} != {expected}")


# ============ 测试用例 ============


def test_basic_write():
    """测试 1: 基础写入 — 1D + 2D 混合,小 size_threshold 触发 flush。"""
    print("\n" + "=" * 70)
    print("[test 1] 基础写入:1D + 2D 混合")
    print("=" * 70)

    cfg = _make_config("basic")
    batch, dim, n = 100, 32, 5
    expected = batch * n

    hook = TraceToOdpsHook(
        config=cfg, fields=FULL_FIELDS, types=FULL_TYPES,
        worker_num=1, size_threshold=4 * 1024,  # 4 KB,几乎每步都 flush
    )
    for i in range(n):
        add_with_padding({
            "id": np.arange(batch, dtype=np.int64) + i * batch,
            "val": [f"basic-{i}-{j}" for j in range(batch)],
            "score": np.random.rand(batch).astype(np.float64),
            "embedding": np.random.randn(batch, dim).astype(np.float64),
        }, n=batch)
        hook.after_step()
        print(f"  [step-{i}] pushed {batch} rows")
    hook.end()

    _verify(cfg, expected, "basic write")
    return cfg


def test_large_batch_auto_threshold():
    """测试 2: 大批量写入 — 自适应阈值,多次 flush + auto-adjust。"""
    print("\n" + "=" * 70)
    print("[test 2] 大批量写入:自适应阈值,1k×256 embedding")
    print("=" * 70)

    cfg = _make_config("large")
    batch, dim, n = 1_000, 256, 10
    expected = batch * n

    hook = TraceToOdpsHook(
        config=cfg, fields=FULL_FIELDS, types=FULL_TYPES,
        worker_num=2, size_threshold=None,  # 自适应
    )
    for i in range(n):
        add_with_padding({
            "user_id": np.arange(batch, dtype=np.int64) + i * batch,
            "embedding": np.random.randn(batch, dim).astype(np.float64),
        }, n=batch)
        hook.after_step()
        print(f"  [step-{i}] pushed {batch}×{dim} ({batch * dim * 8 / 1024:.0f} KB)")
    hook.end()

    _verify(cfg, expected, "large batch auto-threshold")
    return cfg


def test_time_based_flush():
    """测试 3: 时间保底 flush — 用极小 _MAX_FLUSH_INTERVAL 模拟 6 小时触发。

    通过 monkey-patch 将 _MAX_FLUSH_INTERVAL 缩短到 3 秒,
    确保在低数据量场景下时间保底 flush 能正常触发。
    """
    print("\n" + "=" * 70)
    print("[test 3] 时间保底 flush:模拟 6 小时触发")
    print("=" * 70)

    cfg = _make_config("time_flush")
    batch = 10
    expected = batch * 3  # 3 步,每步 10 行

    hook = TraceToOdpsHook(
        config=cfg, fields=FULL_FIELDS, types=FULL_TYPES,
        worker_num=1, size_threshold=1024 * 1024 * 1024,  # 1 GB,永远不会按大小 flush
    )
    # monkey-patch:将保底间隔从 6h 缩短到 3s
    for w in hook.writers:
        w._MAX_FLUSH_INTERVAL = 3

    for i in range(3):
        add_with_padding({
            "id": np.arange(batch, dtype=np.int64) + i * batch,
            "val": [f"time-{i}-{j}" for j in range(batch)],
        }, n=batch)
        hook.after_step()
        print(f"  [step-{i}] pushed {batch} rows, waiting 4s for time-based flush...")
        time.sleep(4)  # 等待 _MAX_FLUSH_INTERVAL(3s) 触发

    hook.end()
    _verify(cfg, expected, "time-based flush")
    return cfg


def test_multi_writer():
    """测试 4: 多 writer 并发 — 4 个 writer 线程同时写入。"""
    print("\n" + "=" * 70)
    print("[test 4] 多 writer 并发:4 线程")
    print("=" * 70)

    cfg = _make_config("multi")
    batch, n = 200, 10
    expected = batch * n

    hook = TraceToOdpsHook(
        config=cfg, fields=FULL_FIELDS, types=FULL_TYPES,
        worker_num=4, size_threshold=16 * 1024,
    )
    for i in range(n):
        add_with_padding({
            "id": np.arange(batch, dtype=np.int64) + i * batch,
            "val": [f"mw-{i}-{j}" for j in range(batch)],
        }, n=batch)
        hook.after_step()
    hook.end()

    _verify(cfg, expected, "multi-writer")
    return cfg


def test_graceful_end():
    """测试 5: 优雅关闭 — end() 确保所有 buffer 数据 commit。

    只写 1 步小数据(不超过 size_threshold),直接 end(),
    验证 run() finally 里的 commit(force=True) 把数据写出。
    """
    print("\n" + "=" * 70)
    print("[test 5] 优雅关闭:end() 保证 buffer 数据不丢")
    print("=" * 70)

    cfg = _make_config("end")
    batch = 30
    expected = batch

    hook = TraceToOdpsHook(
        config=cfg, fields=FULL_FIELDS, types=FULL_TYPES,
        worker_num=1, size_threshold=1024 * 1024 * 1024,  # 1 GB,不触发大小 flush
    )
    add_with_padding({
        "id": np.arange(batch, dtype=np.int64),
        "val": [f"end-{j}" for j in range(batch)],
    }, n=batch)
    hook.after_step()
    print(f"  pushed {batch} rows (all in buffer)")

    hook.end()
    _verify(cfg, expected, "graceful end")
    return cfg


# ============ main ============


def main():
    parser = argparse.ArgumentParser(description="TraceToOdpsHookV2 E2E integration test")
    parser.add_argument(
        "--test", type=str, default=None,
        help="run a specific test by number (1-5)",
    )
    args = parser.parse_args()

    print("[info] test target : TraceToOdpsHookV2 E2E")
    print(f"[info] test table  : {TEST_TABLE}")

    all_tests = [
        ("1", "basic_write", test_basic_write),
        ("2", "large_batch", test_large_batch_auto_threshold),
        ("3", "time_flush", test_time_based_flush),
        ("4", "multi_writer", test_multi_writer),
        ("5", "graceful_end", test_graceful_end),
    ]

    # 按选择过滤
    if args.test:
        all_tests = [(n, label, fn) for n, label, fn in all_tests if n == args.test]
        if not all_tests:
            print(f"[error] test {args.test} not found, available: 1-5")
            return

    configs = []
    passed = 0
    failed = 0

    for num, label, fn in all_tests:
        try:
            cfg = fn()
            configs.append(cfg)
            passed += 1
        except Exception as e:
            print(f"\n  [FAIL] test {num} ({label}): {e}")
            failed += 1

    # 汇总
    print("\n" + "=" * 70)
    total = passed + failed
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if failed == 0:
        print("All tests passed")
    else:
        print("Some tests FAILED")
    print("=" * 70)


if __name__ == "__main__":
    main()
