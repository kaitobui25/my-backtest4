"""Timing helper for engine optimization comparison (not a pytest test)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine_equiv_utils import make_test_configs, make_test_data, run_and_collect

df = make_test_data()
configs = make_test_configs()
t0 = time.perf_counter()
run_and_collect(df, configs, str(Path("tests") / "_time_tmp.parquet"))
elapsed = time.perf_counter() - t0
for ext in (".parquet", ".csv"):
    p = Path("tests") / f"_time_tmp{ext}"
    if p.exists():
        p.unlink()
print(f"ENGINE_TIME_SEC={elapsed:.4f}")
