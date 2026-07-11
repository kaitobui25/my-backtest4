import pandas as pd
from btsearch.engine import shift_to_next_open


def test_signal_enters_next_candle():
    signal = pd.Series([False, True, False, False])
    shifted = shift_to_next_open(signal)
    assert shifted.tolist() == [False, False, True, False]
