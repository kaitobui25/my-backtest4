from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha1
import json
from typing import Any


TARGET_REFERENCE_R = 0.225


@dataclass(frozen=True)
class StrategyConfig:
    family: str
    direction: str
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def config_id(self) -> str:
        payload = {
            "family": self.family,
            "direction": self.direction,
            "params": self.params,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return sha1(encoded.encode("utf-8")).hexdigest()[:14]

    def to_record(self) -> dict[str, Any]:
        return {
            "config_id": self.config_id,
            "family": self.family,
            "direction": self.direction,
            "params_json": json.dumps(
                self.params, sort_keys=True, ensure_ascii=False
            ),
        }


@dataclass(frozen=True)
class RunSettings:
    fee: float = 0.0005
    slippage: float = 0.0002
    batch_size: int = 48
    min_trades_ranking: int = 100
    target_reference_r: float = TARGET_REFERENCE_R
    top_per_family: int = 20
    phase2_seed_per_family: int = 5
    walk_forward_select_per_fold: int = 30

    def validate(self) -> None:
        if self.fee < 0 or self.slippage < 0:
            raise ValueError("Fee và slippage không được âm.")
        if self.batch_size < 1:
            raise ValueError("batch_size phải >= 1.")
        if self.min_trades_ranking < 1:
            raise ValueError("min_trades_ranking phải >= 1.")
