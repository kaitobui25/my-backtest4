from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from btsearch.config import RunSettings
from btsearch.engine import VectorBTBatchEngine
from engine_equiv_utils import (
    BASELINE_FIELDS,
    compare_results,
    make_test_configs,
    make_test_data,
    run_and_collect,
)

BASELINE_PATH = Path(__file__).parent / "baseline_engine.json"


def _baseline():
    assert BASELINE_PATH.exists(), (
        "baseline_engine.json missing; run tests\\gen_baseline.py first."
    )
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def _to_map(result_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {
        row["config_id"]: {f: row[f] for f in BASELINE_FIELDS}
        for _, row in result_df.iterrows()
    }


def _run_old_per_batch(df, configs, path, batch_size=48):
    """Reproduce the pre-optimization behavior: a fresh IndicatorCache per
    batch (cache=None), as the engine did before the cache-per-run change."""
    engine = VectorBTBatchEngine(RunSettings(batch_size=batch_size))
    frames = []
    for start in range(0, len(configs), batch_size):
        batch = configs[start:start + batch_size]
        frames.append(engine._run_batch(df, batch, None))
    return pd.concat(frames, ignore_index=True)


# A. Cache equivalence: old (per-batch) and new (per-run) must match baseline.
def test_A_cache_equivalence(tmp_path):
    baseline = _baseline()["results"]
    df = make_test_data()
    configs = make_test_configs()

    new_run = run_and_collect(df, configs, tmp_path / "new.parquet")
    old_run = _to_map(_run_old_per_batch(df, configs, tmp_path / "old.parquet"))

    compare_results(baseline, new_run)
    compare_results(baseline, old_run)


# B. Batch-size invariance: 1, 48, 96 must all match.
@pytest.mark.parametrize("batch_size", [1, 48, 96])
def test_B_batch_size_invariance(tmp_path, batch_size):
    baseline = _baseline()["results"]
    df = make_test_data()
    configs = make_test_configs()

    actual = run_and_collect(
        df, configs, tmp_path / f"bs{batch_size}.parquet", batch_size=batch_size
    )
    compare_results(baseline, actual)


# C. Ordering invariance: original / grouped / reversed must all match.
def test_C_ordering_invariance(tmp_path):
    baseline = _baseline()["results"]
    df = make_test_data()
    configs = make_test_configs()

    original = run_and_collect(
        df, configs, tmp_path / "orig.parquet", group=False
    )
    grouped = run_and_collect(
        df, configs, tmp_path / "grouped.parquet", group=True
    )
    reversed_order = run_and_collect(
        df, list(reversed(configs)), tmp_path / "rev.parquet", group=False
    )

    compare_results(baseline, original)
    compare_results(baseline, grouped)
    compare_results(baseline, reversed_order)


# D. Dataset / fold isolation: separate runs build separate caches; running
# dataset A, then B, then A again must give identical A results.
def test_D_dataset_isolation(tmp_path, monkeypatch):
    import btsearch.engine as engine_mod
    from btsearch.indicators import IndicatorCache

    created = []

    class _RecordingCache(IndicatorCache):
        def __init__(self, df):
            created.append(id(df))
            super().__init__(df)

    monkeypatch.setattr(engine_mod, "IndicatorCache", _RecordingCache)

    df_A = make_test_data(seed=1)
    df_B = make_test_data(seed=2)
    configs = make_test_configs()

    res_A1 = run_and_collect(df_A, configs, tmp_path / "A1.parquet")
    res_B = run_and_collect(df_B, configs, tmp_path / "B.parquet")
    res_A2 = run_and_collect(df_A, configs, tmp_path / "A2.parquet")

    compare_results(res_A1, res_A2)
    # Every engine.run created a cache bound to its own dataframe object; the
    # third run (A again) used df_A's identity, proving no cross-run sharing.
    assert created == [id(df_A), id(df_B), id(df_A)], created

    assert set(res_A1) == set(res_B)  # same configs evaluated on both datasets


# E. Checkpoint / resume equivalence.
def test_E_checkpoint_resume(tmp_path):
    baseline = _baseline()["results"]
    df = make_test_data()
    configs = make_test_configs()
    settings = RunSettings()
    engine = VectorBTBatchEngine(settings)

    clean = _to_map(
        engine.run(df, configs, tmp_path / "clean.parquet", False, "clean")
    )

    half = configs[: len(configs) // 2]
    engine.run(df, half, tmp_path / "resume.parquet", False, "partial")
    resumed_df = engine.run(
        df, configs, tmp_path / "resume.parquet", True, "resume"
    )
    resumed = _to_map(resumed_df)

    # No config missing, none duplicated.
    assert len(resumed_df) == len(configs)
    assert (resumed_df["config_id"].value_counts() == 1).all()
    assert set(resumed) == set(clean)

    compare_results(clean, resumed)
    # The resumed run also matches the original baseline.
    compare_results(baseline, resumed)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
