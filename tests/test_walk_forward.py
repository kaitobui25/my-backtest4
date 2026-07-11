import pandas as pd
from btsearch.walk_forward import build_expanding_folds


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
