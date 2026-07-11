from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import math
import time
from typing import Iterable

import numpy as np
import pandas as pd

from btsearch.checkpoint import CheckpointStore
from btsearch.config import RunSettings, StrategyConfig
from btsearch.indicators import IndicatorCache
from btsearch.metrics import calculate_net_r, evidence_label, summarize_r
from btsearch.strategies import make_signals
from btsearch.strategies.common import apply_time_filter


def shift_to_next_open(signal: pd.Series) -> pd.Series:
    return signal.shift(1, fill_value=False).astype(bool)


def _batches(
    configs: list[StrategyConfig], size: int
) -> Iterable[list[StrategyConfig]]:
    for start in range(0, len(configs), size):
        yield configs[start:start + size]


def _group_key(cfg: StrategyConfig) -> tuple:
    """Deterministic ordering key: family, then indicator parameters, then
    direction, then config_id. Grouping only changes execution order, never
    config content, so checkpoint/resume and ranking are unaffected."""
    params_key = tuple(
        (k, str(v)) for k, v in sorted(cfg.params.items())
    )
    return (cfg.family, params_key, cfg.direction, cfg.config_id)


class VectorBTBatchEngine:
    def __init__(self, settings: RunSettings):
        settings.validate()
        self.settings = settings

    def _run_batch(
        self,
        df: pd.DataFrame,
        batch: list[StrategyConfig],
        cache: IndicatorCache | None = None,
    ) -> pd.DataFrame:
        try:
            import vectorbt as vbt
        except ImportError as exc:
            raise RuntimeError(
                "Chưa cài vectorbt. Chạy run.ps1 hoặc pip install -r requirements.txt"
            ) from exc

        if cache is None:
            # Legacy per-batch behavior retained only for equivalence testing.
            cache = IndicatorCache(df)
        else:
            # Defensive: the cache must belong to the exact dataframe used by
            # this batch so indicators are never read from the wrong data.
            assert cache.df is df, (
                "IndicatorCache was built from a different dataframe than the "
                "one passed to this batch."
            )
        n_rows = len(df)
        n_cols = len(batch)
        index = df.index
        columns = [cfg.config_id for cfg in batch]

        long_entries = np.zeros((n_rows, n_cols), dtype=bool)
        short_entries = np.zeros((n_rows, n_cols), dtype=bool)
        sl_stop = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
        tp_stop = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
        entry_open = df["open"].to_numpy(dtype=np.float64)

        for col, cfg in enumerate(batch):
            long_raw, short_raw = make_signals(cache, cfg)
            long_entry = shift_to_next_open(long_raw)
            short_entry = shift_to_next_open(short_raw)
            long_entry, short_entry = apply_time_filter(
                long_entry, short_entry, cfg
            )
            long_entries[:, col] = long_entry.to_numpy(bool)
            short_entries[:, col] = short_entry.to_numpy(bool)

            atr_window = int(cfg.params.get("atr_window", 14))
            atr_for_entry = cache.atr(atr_window).shift(1).to_numpy(float)
            risk_pct = (
                float(cfg.params["sl_atr"])
                * atr_for_entry
                / entry_open
            )
            risk_pct[(risk_pct <= 0) | ~np.isfinite(risk_pct)] = np.nan
            sl_stop[:, col] = risk_pct
            tp_stop[:, col] = risk_pct * float(cfg.params["rr"])

        common = dict(
            close=df["close"],
            entries=pd.DataFrame(long_entries, index=index, columns=columns),
            exits=False,
            short_entries=pd.DataFrame(
                short_entries, index=index, columns=columns
            ),
            short_exits=False,
            size=1.0,
            price=df["open"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            sl_stop=pd.DataFrame(sl_stop, index=index, columns=columns),
            tp_stop=pd.DataFrame(tp_stop, index=index, columns=columns),
            stop_entry_price="price",
            fees=self.settings.fee,
            slippage=self.settings.slippage,
            init_cash=1_000_000_000.0,
            accumulate=False,
            upon_opposite_entry="ignore",
            freq="15min",
        )

        try:
            portfolio = vbt.Portfolio.from_signals(
                engine="auto", **common
            )
        except (TypeError, ValueError) as exc:
            if "engine" not in str(exc).lower():
                raise
            portfolio = vbt.Portfolio.from_signals(**common)

        records = portfolio.trades.records.copy()
        required = {
            "col", "size", "entry_idx", "entry_price", "pnl", "status"
        }
        if len(records) and not required.issubset(records.columns):
            raise RuntimeError(
                "Trade records thiếu field cần thiết: "
                f"{sorted(required.difference(records.columns))}"
            )

        if len(records):
            records = records.loc[records["status"] == 1].copy()
        else:
            records = pd.DataFrame(columns=list(required))

        if len(records):
            record_cols = records["col"].to_numpy(np.int64)
            entry_idx = records["entry_idx"].to_numpy(np.int64)
            risk_pct = sl_stop[entry_idx, record_cols]
            records["net_r"] = calculate_net_r(
                pnl=records["pnl"].to_numpy(float),
                size=records["size"].to_numpy(float),
                entry_price=records["entry_price"].to_numpy(float),
                initial_risk_pct=risk_pct,
            )
            # Approximate modeled round-trip cost in R. Fees are exact in
            # vectorbt PnL; slippage is embedded in execution prices, so this
            # separate column is a transparent notional estimate.
            records["modeled_cost_r"] = np.divide(
                2.0 * (self.settings.fee + self.settings.slippage),
                risk_pct,
                out=np.full_like(risk_pct, np.nan, dtype=float),
                where=(risk_pct > 0) & np.isfinite(risk_pct),
            )
        else:
            records["net_r"] = pd.Series(dtype=float)
            records["modeled_cost_r"] = pd.Series(dtype=float)

        rows: list[dict] = []
        for col, cfg in enumerate(batch):
            config_records = records.loc[records["col"] == col]
            values = config_records["net_r"].to_numpy(float)
            row = cfg.to_record()
            row.update(summarize_r(values))
            row["cost_r_per_trade"] = (
                float(config_records["modeled_cost_r"].mean())
                if len(config_records)
                else np.nan
            )
            row["evidence"] = evidence_label(int(row["trades"]))
            row["fee_per_fill"] = self.settings.fee
            row["slippage_per_fill"] = self.settings.slippage
            rows.append(row)
        return pd.DataFrame(rows)

    def run(
        self,
        df: pd.DataFrame,
        configs: list[StrategyConfig],
        checkpoint_path: str | Path,
        resume: bool,
        label: str,
        group: bool = True,
    ) -> pd.DataFrame:
        store = CheckpointStore(checkpoint_path)
        if not resume:
            for path in (store.parquet_path, store.csv_path):
                if path.exists():
                    path.unlink()

        pending = store.pending(configs) if resume else configs
        if not pending:
            print(f"[{label}] checkpoint đã hoàn tất.")
            return store.load()

        # Deterministic grouping keeps similar configs adjacent so that
        # indicator computation is reused more effectively across batches.
        # It only changes execution order; config content, checkpoint/resume
        # (keyed by config_id) and final ranking are all unaffected.
        if group:
            pending = sorted(pending, key=_group_key)

        # One IndicatorCache per run, shared across every batch and discarded
        # when run() returns. It is never shared between separate engine.run()
        # calls (Phase 1, Phase 2, TRAIN, VALIDATION, or walk-forward folds).
        cache = IndicatorCache(df)

        total_batches = math.ceil(
            len(pending) / self.settings.batch_size
        )
        for batch_no, batch in enumerate(
            _batches(pending, self.settings.batch_size), start=1
        ):
            started = time.perf_counter()
            rows = self._run_batch(df, batch, cache)
            store.append(rows)
            elapsed = time.perf_counter() - started
            print(
                f"[{label}] batch {batch_no}/{total_batches} | "
                f"{len(batch)} configs | {elapsed:.2f}s",
                flush=True,
            )

        result = store.load()
        return result.sort_values(
            ["expectancy_r", "trades"],
            ascending=[False, False],
            na_position="last",
        ).reset_index(drop=True)
