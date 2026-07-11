import pandas as pd
import pytest

from btsearch.config import RunSettings, StrategyConfig
from btsearch.walk_forward import build_expanding_folds, run_walk_forward


def test_walk_forward_has_no_overlap():
    index = pd.date_range(
        "2021-01-01",
        "2026-07-01",
        freq="15min",
        tz="UTC",
    )
    first_validation = pd.Timestamp("2024-04-01", tz="UTC")
    folds = build_expanding_folds(
        index,
        first_validation_start=first_validation,
    )
    assert folds
    assert folds[0].validation_start >= first_validation
    for fold in folds:
        assert fold.train_end == fold.validation_start
        assert fold.train_start < fold.train_end
        assert fold.validation_start < fold.validation_end


def _make_configs(n: int) -> list[StrategyConfig]:
    families = ["trend", "momentum", "mean_reversion"]
    return [
        StrategyConfig(
            family=families[i % len(families)],
            direction="long",
            params={"p": i},
        )
        for i in range(n)
    ]


def _fake_results(configs):
    rows = []
    for cfg in configs:
        rec = cfg.to_record()
        rec.update({
            "trades": 200,
            "expectancy_r": 0.1,
            "winrate": 0.5,
            "avg_win_r": 0.3,
            "avg_loss_r": -0.2,
            "profit_factor_r": 1.5,
            "max_drawdown_r": 1.0,
            "cost_r_per_trade": 0.05,
        })
        rows.append(rec)
    return pd.DataFrame(rows)


class _RecordingEngine:
    instances: list["_RecordingEngine"] = []

    def __init__(self, settings):
        self.settings = settings
        self.calls: list[tuple[str, list[str]]] = []
        _RecordingEngine.instances.append(self)

    def run(self, df, configs, checkpoint_path, resume, label):
        self.calls.append((label, [c.config_id for c in configs]))
        return _fake_results(configs)


def _run_walk_forward(candidates, tmp_path, monkeypatch):
    import btsearch.walk_forward as wf_mod

    _RecordingEngine.instances = []
    monkeypatch.setattr(wf_mod, "VectorBTBatchEngine", _RecordingEngine)
    index = pd.date_range(
        "2021-01-01", "2026-07-01", freq="15min", tz="UTC"
    )
    return run_walk_forward(
        pd.DataFrame(index=index),
        candidates,
        RunSettings(),
        tmp_path,
        False,
        pd.Timestamp("2024-04-01", tz="UTC"),
    )


def test_shortlist_frozen_before_walk_forward(tmp_path, monkeypatch):
    candidates = _make_configs(8)
    result = _run_walk_forward(candidates, tmp_path, monkeypatch)

    assert _RecordingEngine.instances, "engine was never created"
    all_calls = [
        call for engine in _RecordingEngine.instances
        for call in engine.calls
    ]
    validation_calls = [c for c in all_calls if "VALIDATION" in c[0]]
    train_calls = [c for c in all_calls if "TRAIN" in c[0]]
    assert train_calls == [], (
        "train must not be re-run for per-fold selection"
    )
    assert validation_calls, "no validation folds were evaluated"

    expected_ids = sorted(c.config_id for c in candidates)
    for label, ids in validation_calls:
        assert sorted(ids) == expected_ids, (
            f"fold {label} did not receive the frozen shortlist"
        )
    assert not result.empty


def test_every_shortlisted_config_tested_on_every_fold(tmp_path, monkeypatch):
    candidates = _make_configs(5)
    result = _run_walk_forward(candidates, tmp_path, monkeypatch)

    folds = result["fold_id"].nunique()
    assert folds >= 1
    per_config_folds = result.groupby("config_id")["fold_id"].nunique()
    assert (per_config_folds == folds).all()

