"""One-off generator for the engine equivalence baseline.

Run with:  .venv\\Scripts\\python.exe tests\\gen_baseline.py

Produces tests/baseline_engine.json containing per-config_id results from the
CURRENT engine. This file is committed and used by the equivalence tests so
that later optimizations can prove result-equivalence.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine_equiv_utils import (
    BASELINE_FIELDS,
    make_test_configs,
    make_test_data,
    run_and_collect,
)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root
        ).decode().strip()
    except Exception:
        sha = "unknown"

    df = make_test_data()
    configs = make_test_configs()
    tmp_path = repo_root / "tests" / "_bench_tmp.parquet"
    results = run_and_collect(df, configs, str(tmp_path))
    for ext in (".parquet", ".csv"):
        p = tmp_path.with_suffix(ext)
        if p.exists():
            p.unlink()

    baseline = {
        "baseline_commit_sha": sha,
        "engine_settings": {"fee": 0.0005, "slippage": 0.0002, "batch_size": 48},
        "dataset": {"seed": 42, "n_rows": 3000},
        "config_count": len(configs),
        "fields": BASELINE_FIELDS,
        "results": results,
    }
    out = repo_root / "tests" / "baseline_engine.json"
    out.write_text(
        json.dumps(baseline, indent=2, default=str), encoding="utf-8"
    )
    print(f"wrote {out} | configs={len(results)} | engine_sha={sha}")


if __name__ == "__main__":
    main()
