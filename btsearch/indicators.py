from __future__ import annotations

import numpy as np
import pandas as pd


class IndicatorCache:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.close = df["close"]
        self._cache: dict[tuple, pd.Series | tuple[pd.Series, ...]] = {}

    def ema(self, window: int) -> pd.Series:
        key = ("ema", window)
        if key not in self._cache:
            self._cache[key] = self.close.ewm(
                span=window, adjust=False, min_periods=window
            ).mean()
        return self._cache[key]  # type: ignore[return-value]

    def sma(self, window: int) -> pd.Series:
        key = ("sma", window)
        if key not in self._cache:
            self._cache[key] = self.close.rolling(
                window, min_periods=window
            ).mean()
        return self._cache[key]  # type: ignore[return-value]

    def std(self, window: int) -> pd.Series:
        key = ("std", window)
        if key not in self._cache:
            self._cache[key] = self.close.rolling(
                window, min_periods=window
            ).std(ddof=0)
        return self._cache[key]  # type: ignore[return-value]

    def atr(self, window: int) -> pd.Series:
        key = ("atr", window)
        if key not in self._cache:
            prev_close = self.close.shift(1)
            tr = pd.concat(
                [
                    self.df["high"] - self.df["low"],
                    (self.df["high"] - prev_close).abs(),
                    (self.df["low"] - prev_close).abs(),
                ],
                axis=1,
            ).max(axis=1)
            self._cache[key] = tr.ewm(
                alpha=1.0 / window,
                adjust=False,
                min_periods=window,
            ).mean()
        return self._cache[key]  # type: ignore[return-value]

    def rsi(self, window: int) -> pd.Series:
        key = ("rsi", window)
        if key not in self._cache:
            delta = self.close.diff()
            gain = delta.clip(lower=0.0)
            loss = -delta.clip(upper=0.0)
            avg_gain = gain.ewm(
                alpha=1.0 / window,
                adjust=False,
                min_periods=window,
            ).mean()
            avg_loss = loss.ewm(
                alpha=1.0 / window,
                adjust=False,
                min_periods=window,
            ).mean()
            rs = avg_gain / avg_loss.replace(0.0, np.nan)
            self._cache[key] = (
                100.0 - 100.0 / (1.0 + rs)
            ).fillna(50.0)
        return self._cache[key]  # type: ignore[return-value]

    def rolling_high(self, window: int) -> pd.Series:
        key = ("rolling_high", window)
        if key not in self._cache:
            # shift(1): current candle is never part of its own breakout level.
            self._cache[key] = self.df["high"].shift(1).rolling(
                window, min_periods=window
            ).max()
        return self._cache[key]  # type: ignore[return-value]

    def rolling_low(self, window: int) -> pd.Series:
        key = ("rolling_low", window)
        if key not in self._cache:
            self._cache[key] = self.df["low"].shift(1).rolling(
                window, min_periods=window
            ).min()
        return self._cache[key]  # type: ignore[return-value]

    def roc(self, window: int) -> pd.Series:
        key = ("roc", window)
        if key not in self._cache:
            self._cache[key] = self.close.pct_change(window)
        return self._cache[key]  # type: ignore[return-value]

    def macd(
        self, fast: int, slow: int, signal: int
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        key = ("macd", fast, slow, signal)
        if key not in self._cache:
            macd_line = self.ema(fast) - self.ema(slow)
            signal_line = macd_line.ewm(
                span=signal,
                adjust=False,
                min_periods=signal,
            ).mean()
            histogram = macd_line - signal_line
            self._cache[key] = (macd_line, signal_line, histogram)
        return self._cache[key]  # type: ignore[return-value]

    def volume_mean(self, window: int) -> pd.Series:
        if "volume" not in self.df.columns:
            raise ValueError("Data không có volume.")
        key = ("volume_mean", window)
        if key not in self._cache:
            self._cache[key] = self.df["volume"].rolling(
                window, min_periods=window
            ).mean()
        return self._cache[key]  # type: ignore[return-value]

    def bollinger(
        self, window: int, std_mult: float
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        key = ("bollinger", window, std_mult)
        if key not in self._cache:
            mean = self.sma(window)
            std = self.std(window)
            self._cache[key] = (
                mean - std_mult * std,
                mean,
                mean + std_mult * std,
            )
        return self._cache[key]  # type: ignore[return-value]

    def bandwidth(self, window: int, std_mult: float) -> pd.Series:
        key = ("bandwidth", window, std_mult)
        if key not in self._cache:
            lower, middle, upper = self.bollinger(window, std_mult)
            self._cache[key] = (upper - lower) / middle.replace(0.0, np.nan)
        return self._cache[key]  # type: ignore[return-value]
