from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import vectorbt as vbt


TARGET_R = 0.225


@dataclass(frozen=True)
class Config:
    config_id: str
    strategy: str
    ema_window: int
    signal_window: int
    signal_level: float
    atr_window: int
    sl_atr: float
    rr: float
    direction: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fast BTC 15m strategy screener using vectorbt."
    )
    p.add_argument("--data", required=True, help="Path to OHLCV parquet file.")
    p.add_argument("--mode", choices=["instant", "full"], default="instant")
    p.add_argument("--fee", type=float, default=0.0005,
                   help="Fee rate per fill. Default 0.0005 = 0.05%%.")
    p.add_argument("--slippage", type=float, default=0.0002,
                   help="Slippage rate per fill. Default 0.0002 = 0.02%%.")
    p.add_argument("--target-r", type=float, default=TARGET_R)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--output", default="output")
    return p.parse_args()


def _parse_datetime(values: pd.Series) -> pd.DatetimeIndex:
    if pd.api.types.is_numeric_dtype(values):
        clean = pd.to_numeric(values, errors="coerce")
        max_abs = clean.abs().max()
        if pd.isna(max_abs):
            raise ValueError("Datetime column is empty.")
        if max_abs > 1e17:
            unit = "ns"
        elif max_abs > 1e14:
            unit = "us"
        elif max_abs > 1e11:
            unit = "ms"
        else:
            unit = "s"
        parsed = pd.to_datetime(clean, unit=unit, utc=True, errors="coerce")
    else:
        parsed = pd.to_datetime(values, utc=True, errors="coerce")
    return pd.DatetimeIndex(parsed)


def load_ohlcv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy data: {path}")

    df = pd.read_parquet(path)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(str(x) for x in col if str(x) != "") for col in df.columns]

    original_columns = list(df.columns)
    lower_map = {str(c).strip().lower(): c for c in original_columns}

    rename = {}
    aliases = {
        "open": ["open", "o"],
        "high": ["high", "h"],
        "low": ["low", "l"],
        "close": ["close", "c"],
        "volume": ["volume", "vol", "v"],
    }
    for target, names in aliases.items():
        for name in names:
            if name in lower_map:
                rename[lower_map[name]] = target
                break
    df = df.rename(columns=rename)

    if not isinstance(df.index, pd.DatetimeIndex):
        datetime_col = None
        for candidate in [
            "datetime", "date", "timestamp", "time", "open_time",
            "opentime", "candle_begin_time"
        ]:
            if candidate in lower_map:
                datetime_col = lower_map[candidate]
                break
        if datetime_col is None:
            raise ValueError(
                "Không tìm thấy datetime. Cần DatetimeIndex hoặc cột "
                "datetime/date/timestamp/time/open_time."
            )
        df.index = _parse_datetime(df[datetime_col])
    else:
        idx = pd.DatetimeIndex(df.index)
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_convert("UTC")
        df.index = idx

    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc: {missing}. Hiện có: {list(df.columns)}")

    keep = required + (["volume"] if "volume" in df.columns else [])
    df = df[keep].copy()
    for col in keep:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df[~df.index.isna()]
        .sort_index()
        .loc[lambda x: ~x.index.duplicated(keep="last")]
        .dropna(subset=required)
    )

    invalid = (
        (df["high"] < df[["open", "close"]].max(axis=1))
        | (df["low"] > df[["open", "close"]].min(axis=1))
        | (df["high"] < df["low"])
    )
    if invalid.any():
        raise ValueError(f"Data OHLC sai ở {int(invalid.sum())} candle.")

    if len(df) < 5_000:
        raise ValueError(f"Data quá ngắn: chỉ có {len(df):,} candle.")

    return df


def make_configs(mode: str) -> list[Config]:
    configs: list[Config] = []
    n = 0

    if mode == "instant":
        ema_windows = [50, 100, 200]
        atr_windows = [14]
        sl_atrs = [1.25, 1.75]
        rrs = [1.5, 2.0, 2.5]
        directions = ["long", "short", "both"]

        # Trend breakout
        for ema, window, atr, sl, rr, direction in itertools.product(
            ema_windows, [20, 48, 96], atr_windows, sl_atrs, rrs, directions
        ):
            n += 1
            configs.append(Config(
                f"C{n:04d}", "donchian_breakout", ema, window, 0.0,
                atr, sl, rr, direction
            ))

        # Pullback / mean-reversion in trend
        for ema, rsi_window, level, atr, sl, rr, direction in itertools.product(
            ema_windows, [7, 14], [20.0, 30.0],
            atr_windows, sl_atrs, rrs, ["long", "short"]
        ):
            n += 1
            configs.append(Config(
                f"C{n:04d}", "rsi_pullback", ema, rsi_window, level,
                atr, sl, rr, direction
            ))

        # Bollinger re-entry
        for ema, window, std_mult, atr, sl, rr, direction in itertools.product(
            [100, 200], [20, 40], [2.0],
            atr_windows, sl_atrs, rrs, ["long", "short"]
        ):
            n += 1
            configs.append(Config(
                f"C{n:04d}", "bollinger_reentry", ema, window, std_mult,
                atr, sl, rr, direction
            ))
    else:
        ema_windows = [34, 50, 100, 150, 200]
        atr_windows = [7, 14, 28]
        sl_atrs = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5]
        rrs = [1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
        directions = ["long", "short", "both"]

        for ema, window, atr, sl, rr, direction in itertools.product(
            ema_windows, [12, 20, 32, 48, 72, 96],
            atr_windows, sl_atrs, rrs, directions
        ):
            n += 1
            configs.append(Config(
                f"C{n:05d}", "donchian_breakout", ema, window, 0.0,
                atr, sl, rr, direction
            ))

        for ema, rsi_window, level, atr, sl, rr, direction in itertools.product(
            ema_windows, [5, 7, 10, 14, 21], [15.0, 20.0, 25.0, 30.0, 35.0],
            atr_windows, sl_atrs, rrs, directions
        ):
            n += 1
            configs.append(Config(
                f"C{n:05d}", "rsi_pullback", ema, rsi_window, level,
                atr, sl, rr, direction
            ))

        for ema, window, std_mult, atr, sl, rr, direction in itertools.product(
            ema_windows, [12, 20, 30, 40, 60], [1.25, 1.5, 1.75, 2.0, 2.25],
            atr_windows, sl_atrs, rrs, directions
        ):
            n += 1
            configs.append(Config(
                f"C{n:05d}", "bollinger_reentry", ema, window, std_mult,
                atr, sl, rr, direction
            ))

    return configs


def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False, min_periods=window).mean()


def atr(df: pd.DataFrame, window: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    return out.fillna(50.0)


class IndicatorCache:
    def __init__(self, df: pd.DataFrame, configs: Iterable[Config]):
        self.df = df
        self.close = df["close"]
        configs = list(configs)

        self.ema = {
            w: ema(self.close, w)
            for w in sorted({c.ema_window for c in configs})
        }
        self.atr = {
            w: atr(df, w)
            for w in sorted({c.atr_window for c in configs})
        }

        don_windows = sorted({
            c.signal_window for c in configs
            if c.strategy == "donchian_breakout"
        })
        self.don_high = {
            w: df["high"].shift(1).rolling(w, min_periods=w).max()
            for w in don_windows
        }
        self.don_low = {
            w: df["low"].shift(1).rolling(w, min_periods=w).min()
            for w in don_windows
        }

        rsi_windows = sorted({
            c.signal_window for c in configs
            if c.strategy == "rsi_pullback"
        })
        self.rsi = {w: rsi(self.close, w) for w in rsi_windows}

        bb_windows = sorted({
            c.signal_window for c in configs
            if c.strategy == "bollinger_reentry"
        })
        self.bb_mean = {
            w: self.close.rolling(w, min_periods=w).mean()
            for w in bb_windows
        }
        self.bb_std = {
            w: self.close.rolling(w, min_periods=w).std(ddof=0)
            for w in bb_windows
        }

    def raw_signals(self, cfg: Config) -> tuple[pd.Series, pd.Series]:
        close = self.close
        trend = self.ema[cfg.ema_window]

        if cfg.strategy == "donchian_breakout":
            upper = self.don_high[cfg.signal_window]
            lower = self.don_low[cfg.signal_window]
            long_raw = (
                (close > upper)
                & (close.shift(1) <= upper.shift(1))
                & (close > trend)
            )
            short_raw = (
                (close < lower)
                & (close.shift(1) >= lower.shift(1))
                & (close < trend)
            )

        elif cfg.strategy == "rsi_pullback":
            value = self.rsi[cfg.signal_window]
            low_level = cfg.signal_level
            high_level = 100.0 - cfg.signal_level
            long_raw = (
                (value.shift(1) < low_level)
                & (value >= low_level)
                & (close > trend)
            )
            short_raw = (
                (value.shift(1) > high_level)
                & (value <= high_level)
                & (close < trend)
            )

        elif cfg.strategy == "bollinger_reentry":
            mean = self.bb_mean[cfg.signal_window]
            std = self.bb_std[cfg.signal_window]
            lower = mean - cfg.signal_level * std
            upper = mean + cfg.signal_level * std
            long_raw = (close.shift(1) < lower.shift(1)) & (close >= lower)
            short_raw = (close.shift(1) > upper.shift(1)) & (close <= upper)

        else:
            raise ValueError(f"Unknown strategy: {cfg.strategy}")

        if cfg.direction == "long":
            short_raw = pd.Series(False, index=close.index)
        elif cfg.direction == "short":
            long_raw = pd.Series(False, index=close.index)

        conflict = long_raw & short_raw
        return (
            long_raw.mask(conflict, False).fillna(False),
            short_raw.mask(conflict, False).fillna(False),
        )


def iter_batches(items: list[Config], size: int) -> Iterable[list[Config]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


def summarize_r(values: np.ndarray) -> dict[str, float]:
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return {
            "trades": 0,
            "winrate": np.nan,
            "avg_win_r": np.nan,
            "avg_loss_r": np.nan,
            "expectancy_r": np.nan,
            "profit_factor_r": np.nan,
            "total_r": 0.0,
            "max_drawdown_r": np.nan,
        }

    wins = values[values > 0]
    losses = values[values < 0]
    equity = np.cumsum(values)
    peaks = np.maximum.accumulate(np.r_[0.0, equity])
    dd = peaks[1:] - equity

    return {
        "trades": int(n),
        "winrate": float((values > 0).mean()),
        "avg_win_r": float(wins.mean()) if len(wins) else np.nan,
        "avg_loss_r": float(losses.mean()) if len(losses) else np.nan,
        "expectancy_r": float(values.mean()),
        "profit_factor_r": (
            float(wins.sum() / abs(losses.sum()))
            if len(losses) and losses.sum() != 0
            else np.inf
        ),
        "total_r": float(values.sum()),
        "max_drawdown_r": float(dd.max()) if len(dd) else 0.0,
    }


def run_batch(
    df: pd.DataFrame,
    cache: IndicatorCache,
    batch: list[Config],
    fee: float,
    slippage: float,
) -> list[dict]:
    n_rows = len(df)
    n_cols = len(batch)
    index = df.index
    columns = [c.config_id for c in batch]

    long_entries = np.zeros((n_rows, n_cols), dtype=bool)
    short_entries = np.zeros((n_rows, n_cols), dtype=bool)
    sl_stop = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
    tp_stop = np.full((n_rows, n_cols), np.nan, dtype=np.float64)

    entry_open = df["open"].to_numpy(dtype=np.float64)

    for j, cfg in enumerate(batch):
        long_raw, short_raw = cache.raw_signals(cfg)

        # Signal is known at candle close; enter at next candle open.
        long_entries[:, j] = long_raw.shift(1, fill_value=False).to_numpy(bool)
        short_entries[:, j] = short_raw.shift(1, fill_value=False).to_numpy(bool)

        atr_for_entry = cache.atr[cfg.atr_window].shift(1).to_numpy(dtype=np.float64)
        risk_pct = cfg.sl_atr * atr_for_entry / entry_open
        risk_pct[(risk_pct <= 0) | ~np.isfinite(risk_pct)] = np.nan
        sl_stop[:, j] = risk_pct
        tp_stop[:, j] = risk_pct * cfg.rr

    common = dict(
        close=df["close"],
        entries=pd.DataFrame(long_entries, index=index, columns=columns),
        exits=False,
        short_entries=pd.DataFrame(short_entries, index=index, columns=columns),
        short_exits=False,
        size=1.0,
        price=df["open"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        sl_stop=pd.DataFrame(sl_stop, index=index, columns=columns),
        tp_stop=pd.DataFrame(tp_stop, index=index, columns=columns),
        stop_entry_price="price",
        fees=fee,
        slippage=slippage,
        init_cash=1_000_000_000.0,
        accumulate=False,
        upon_opposite_entry="ignore",
        freq="15min",
    )

    try:
        pf = vbt.Portfolio.from_signals(engine="auto", **common)
    except (TypeError, ValueError) as exc:
        # Compatibility fallback for older vectorbt builds.
        if "engine" not in str(exc).lower():
            raise
        pf = vbt.Portfolio.from_signals(**common)

    records = pf.trades.records.copy()
    if len(records) == 0:
        records = pd.DataFrame(columns=[
            "col", "size", "entry_idx", "entry_price", "pnl", "status"
        ])

    required_fields = {"col", "size", "entry_idx", "entry_price", "pnl", "status"}
    missing = required_fields.difference(records.columns)
    if missing:
        raise RuntimeError(
            f"Vectorbt trade record thiếu field {sorted(missing)}. "
            f"Fields hiện có: {list(records.columns)}"
        )

    records = records.loc[records["status"] == 1].copy()
    if len(records):
        cols = records["col"].to_numpy(dtype=np.int64)
        entry_idxs = records["entry_idx"].to_numpy(dtype=np.int64)
        risk_pct = sl_stop[entry_idxs, cols]
        initial_risk = (
            records["size"].to_numpy(dtype=np.float64)
            * records["entry_price"].to_numpy(dtype=np.float64)
            * risk_pct
        )
        pnl = records["pnl"].to_numpy(dtype=np.float64)
        records["net_r"] = np.divide(
            pnl,
            initial_risk,
            out=np.full_like(pnl, np.nan),
            where=(initial_risk > 0) & np.isfinite(initial_risk),
        )
    else:
        records["net_r"] = pd.Series(dtype=float)

    out: list[dict] = []
    for j, cfg in enumerate(batch):
        values = records.loc[records["col"] == j, "net_r"].to_numpy(dtype=float)
        row = asdict(cfg)
        row.update(summarize_r(values))
        out.append(row)
    return out


def run_configs(
    df: pd.DataFrame,
    configs: list[Config],
    fee: float,
    slippage: float,
    batch_size: int,
    label: str,
) -> pd.DataFrame:
    if len(df) < 1_000:
        raise ValueError(f"{label} quá ngắn: {len(df):,} candle.")

    cache = IndicatorCache(df, configs)
    rows: list[dict] = []
    total_batches = math.ceil(len(configs) / batch_size)

    for batch_no, batch in enumerate(iter_batches(configs, batch_size), start=1):
        started = time.perf_counter()
        rows.extend(run_batch(df, cache, batch, fee, slippage))
        elapsed = time.perf_counter() - started
        print(
            f"[{label}] batch {batch_no}/{total_batches} "
            f"({len(batch)} config): {elapsed:.2f}s",
            flush=True,
        )

    result = pd.DataFrame(rows)
    result["target_hit"] = result["expectancy_r"] >= TARGET_R
    return result.sort_values(
        ["expectancy_r", "trades"], ascending=[False, False], na_position="last"
    ).reset_index(drop=True)


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    exact_train_end = pd.Timestamp("2024-07-01", tz="UTC")
    exact_val_end = pd.Timestamp("2025-07-01", tz="UTC")

    if df.index.min() < pd.Timestamp("2022-01-01", tz="UTC") and df.index.max() >= exact_val_end:
        train = df.loc[df.index < exact_train_end]
        validation = df.loc[(df.index >= exact_train_end) & (df.index < exact_val_end)]
        oos = df.loc[df.index >= exact_val_end]
        method = (
            "Fixed split: TRAIN < 2024-07-01; "
            "VALIDATION 2024-07-01..2025-06-30; OOS >= 2025-07-01"
        )
    else:
        n = len(df)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)
        train = df.iloc[:train_end]
        validation = df.iloc[train_end:val_end]
        oos = df.iloc[val_end:]
        method = "Fallback chronological split: 70% TRAIN / 15% VALIDATION / 15% OOS"

    return train, validation, oos, method


def config_lookup(configs: list[Config]) -> dict[str, Config]:
    return {c.config_id: c for c in configs}


def print_table(df: pd.DataFrame, title: str, target: float, n: int = 10) -> None:
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)
    cols = [
        "config_id", "strategy", "direction", "ema_window", "signal_window",
        "signal_level", "sl_atr", "rr", "trades", "winrate",
        "avg_win_r", "avg_loss_r", "expectancy_r", "profit_factor_r",
        "max_drawdown_r"
    ]
    show = df.head(n)[cols].copy()
    for col in [
        "winrate", "avg_win_r", "avg_loss_r", "expectancy_r",
        "profit_factor_r", "max_drawdown_r"
    ]:
        show[col] = show[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "nan")
    print(show.to_string(index=False))
    hits = int((df["expectancy_r"] >= target).sum())
    print(f"\nTarget >= {target:+.3f}R/trade: {hits}/{len(df)} config đạt.")


def save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    data_path = Path(args.data).expanduser().resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"vectorbt: {getattr(vbt, '__version__', 'unknown')}")
    print(f"Data: {data_path}")
    print(f"Mode: {args.mode}")
    print(f"Fee/fill: {args.fee:.6f}; slippage/fill: {args.slippage:.6f}")
    print(f"Target: {args.target_r:+.3f}R/trade")

    df = load_ohlcv(data_path)
    print(
        f"Candles: {len(df):,} | "
        f"{df.index.min()} -> {df.index.max()}"
    )

    configs = make_configs(args.mode)
    print(f"Configs: {len(configs):,}")

    train, validation, oos, split_method = split_data(df)
    print(split_method)
    print(
        f"TRAIN={len(train):,}, VALIDATION={len(validation):,}, OOS={len(oos):,}"
    )

    train_result = run_configs(
        train, configs, args.fee, args.slippage, args.batch_size, "TRAIN"
    )
    train_result["target_hit"] = train_result["expectancy_r"] >= args.target_r
    train_result.to_parquet(output_dir / "01_train_all.parquet", index=False)
    train_result.to_csv(output_dir / "01_train_all.csv", index=False)
    print_table(train_result, "TRAIN — TOP CONFIGS", args.target_r)

    # Use TRAIN only to create the shortlist.
    shortlist_ids = train_result.head(20)["config_id"].tolist()
    lookup = config_lookup(configs)
    shortlist = [lookup[x] for x in shortlist_ids]

    validation_result = run_configs(
        validation, shortlist, args.fee, args.slippage,
        min(args.batch_size, 20), "VALIDATION"
    )
    validation_result["target_hit"] = (
        validation_result["expectancy_r"] >= args.target_r
    )
    validation_result.to_parquet(
        output_dir / "02_validation_shortlist.parquet", index=False
    )
    validation_result.to_csv(
        output_dir / "02_validation_shortlist.csv", index=False
    )
    print_table(
        validation_result, "VALIDATION — TRAIN SHORTLIST", args.target_r
    )

    # Freeze top five by VALIDATION order before touching OOS.
    final_ids = validation_result.head(5)["config_id"].tolist()
    final_configs = [lookup[x] for x in final_ids]

    oos_result = run_configs(
        oos, final_configs, args.fee, args.slippage,
        min(args.batch_size, 5), "FINAL_OOS"
    )
    # Preserve frozen validation order; do not select by OOS rank.
    order_map = {cid: i for i, cid in enumerate(final_ids)}
    oos_result["_frozen_order"] = oos_result["config_id"].map(order_map)
    oos_result = oos_result.sort_values("_frozen_order").drop(columns="_frozen_order")
    oos_result["target_hit"] = oos_result["expectancy_r"] >= args.target_r
    oos_result.to_parquet(output_dir / "03_final_oos.parquet", index=False)
    oos_result.to_csv(output_dir / "03_final_oos.csv", index=False)
    print_table(oos_result, "FINAL OOS — FROZEN TOP 5", args.target_r, n=5)

    best_id = final_ids[0]
    train_row = train_result.loc[train_result["config_id"] == best_id].iloc[0]
    val_row = validation_result.loc[
        validation_result["config_id"] == best_id
    ].iloc[0]
    oos_row = oos_result.loc[oos_result["config_id"] == best_id].iloc[0]

    best = asdict(lookup[best_id])
    best["train"] = {
        "trades": int(train_row["trades"]),
        "expectancy_r": float(train_row["expectancy_r"]),
    }
    best["validation"] = {
        "trades": int(val_row["trades"]),
        "expectancy_r": float(val_row["expectancy_r"]),
    }
    best["oos"] = {
        "trades": int(oos_row["trades"]),
        "expectancy_r": float(oos_row["expectancy_r"]),
    }
    best["target_r"] = args.target_r
    best["oos_target_hit"] = bool(oos_row["expectancy_r"] >= args.target_r)
    best["fee_per_fill"] = args.fee
    best["slippage_per_fill"] = args.slippage
    best["split_method"] = split_method
    save_json(output_dir / "BEST_FROZEN_CONFIG.json", best)

    status = (
        "ĐẠT TARGET TRÊN FINAL OOS"
        if best["oos_target_hit"]
        else "CHƯA ĐẠT TARGET TRÊN FINAL OOS"
    )
    report = f"""RESULT: {status}

Frozen config selected by VALIDATION:
{json.dumps(asdict(lookup[best_id]), indent=2, ensure_ascii=False)}

TRAIN:      trades={int(train_row['trades'])}, expectancy={float(train_row['expectancy_r']):+.6f}R
VALIDATION: trades={int(val_row['trades'])}, expectancy={float(val_row['expectancy_r']):+.6f}R
FINAL OOS:  trades={int(oos_row['trades'])}, expectancy={float(oos_row['expectancy_r']):+.6f}R

Target: {args.target_r:+.6f}R/trade
Fee per fill: {args.fee}
Slippage per fill: {args.slippage}

Lưu ý: đây là fast screening. Trước khi trade thật phải chạy lại config
bằng Exact Engine với same-bar/both-hit rule đã khóa.
"""
    (output_dir / "FINAL_RESULT.txt").write_text(report, encoding="utf-8")

    elapsed = time.perf_counter() - started
    print("\n" + report)
    print(f"Total runtime: {elapsed:.2f}s")
    print(f"Output: {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nĐã dừng bởi người dùng.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
