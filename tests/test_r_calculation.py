import numpy as np
from btsearch.metrics import calculate_net_r


def test_r_calculation():
    result = calculate_net_r(
        pnl=np.array([100.0, -50.0]),
        size=np.array([2.0, 2.0]),
        entry_price=np.array([1000.0, 1000.0]),
        initial_risk_pct=np.array([0.05, 0.05]),
    )
    assert np.allclose(result, np.array([1.0, -0.5]))
