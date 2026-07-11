from btsearch.indicators import IndicatorCache


def test_rolling_high_excludes_current_candle(sample_ohlcv):
    df = sample_ohlcv.copy()
    df.iloc[100, df.columns.get_loc("high")] = 10_000.0
    cache = IndicatorCache(df)
    level = cache.rolling_high(20)

    assert level.iloc[100] < 10_000.0
    assert level.iloc[101] == 10_000.0


def test_rolling_low_excludes_current_candle(sample_ohlcv):
    df = sample_ohlcv.copy()
    df.iloc[100, df.columns.get_loc("low")] = -10_000.0
    cache = IndicatorCache(df)
    level = cache.rolling_low(20)

    assert level.iloc[100] > -10_000.0
    assert level.iloc[101] == -10_000.0
