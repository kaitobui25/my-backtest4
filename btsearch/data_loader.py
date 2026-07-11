from __future__ import annotations

from pathlib import Path
import pandas as pd


DATETIME_ALIASES = (
    "datetime", "date", "timestamp", "time", "open_time",
    "opentime", "candle_begin_time",
)


def _parse_datetime(values: pd.Series) -> pd.DatetimeIndex:
    if pd.api.types.is_numeric_dtype(values):
        clean = pd.to_numeric(values, errors="coerce")
        max_abs = clean.abs().max()
        if pd.isna(max_abs):
            raise ValueError("Datetime column rỗng.")
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


def load_ohlcv(path: str | Path) -> pd.DataFrame:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy data: {path}")

    df = pd.read_parquet(path)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join(str(x) for x in col if str(x))
            for col in df.columns
        ]

    original = list(df.columns)
    lower = {str(col).strip().lower(): col for col in original}

    aliases = {
        "open": ("open", "o"),
        "high": ("high", "h"),
        "low": ("low", "l"),
        "close": ("close", "c"),
        "volume": ("volume", "vol", "v"),
    }
    rename: dict[object, str] = {}
    for canonical, names in aliases.items():
        for name in names:
            if name in lower:
                rename[lower[name]] = canonical
                break
    df = df.rename(columns=rename)

    if not isinstance(df.index, pd.DatetimeIndex):
        dt_col = next(
            (lower[name] for name in DATETIME_ALIASES if name in lower),
            None,
        )
        if dt_col is None:
            raise ValueError(
                "Cần DatetimeIndex hoặc cột datetime/date/timestamp/"
                "time/open_time."
            )
        df.index = _parse_datetime(df[dt_col])
    else:
        index = pd.DatetimeIndex(df.index)
        df.index = (
            index.tz_localize("UTC")
            if index.tz is None
            else index.tz_convert("UTC")
        )

    required = ["open", "high", "low", "close"]
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột {missing}; hiện có {list(df.columns)}")

    keep = required + (["volume"] if "volume" in df.columns else [])
    df = df[keep].copy()
    for col in keep:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.loc[~df.index.isna()]
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
        raise ValueError(f"OHLC sai tại {int(invalid.sum())} candle.")

    if len(df) < 5_000:
        raise ValueError(f"Data quá ngắn: {len(df):,} candle.")

    return df
