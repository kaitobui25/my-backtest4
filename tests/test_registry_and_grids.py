from btsearch.grids import build_coarse_configs
from btsearch.strategies import REGISTRY


def test_coarse_grid_has_many_configs():
    configs = build_coarse_configs(has_volume=True)
    assert len(configs) >= 5_000


def test_config_ids_are_unique():
    configs = build_coarse_configs(has_volume=True)
    ids = [config.config_id for config in configs]
    assert len(ids) == len(set(ids))


def test_every_family_is_registered():
    configs = build_coarse_configs(has_volume=True)
    assert {config.family for config in configs}.issubset(REGISTRY)


def test_bollinger_regime_parameter_is_real():
    configs = [
        config for config in build_coarse_configs(has_volume=False)
        if config.family == "bollinger_reentry"
    ]
    assert configs
    assert all("regime_ema" in config.params for config in configs)
