from __future__ import annotations

from pathlib import Path
import pandas as pd

from btsearch.config import StrategyConfig


class CheckpointStore:
    def __init__(self, parquet_path: str | Path):
        self.parquet_path = Path(parquet_path)
        self.csv_path = self.parquet_path.with_suffix(".csv")

    def load(self) -> pd.DataFrame:
        if self.parquet_path.exists():
            return pd.read_parquet(self.parquet_path)
        if self.csv_path.exists():
            return pd.read_csv(self.csv_path)
        return pd.DataFrame()

    def completed_ids(self) -> set[str]:
        existing = self.load()
        if existing.empty or "config_id" not in existing.columns:
            return set()
        return set(existing["config_id"].astype(str))

    def pending(
        self, configs: list[StrategyConfig]
    ) -> list[StrategyConfig]:
        completed = self.completed_ids()
        return [cfg for cfg in configs if cfg.config_id not in completed]

    def append(self, rows: pd.DataFrame) -> pd.DataFrame:
        existing = self.load()
        combined = pd.concat([existing, rows], ignore_index=True)
        if "config_id" in combined.columns:
            combined = combined.drop_duplicates(
                subset=["config_id"], keep="last"
            )
        self.parquet_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(self.parquet_path, index=False)
        combined.to_csv(self.csv_path, index=False)
        return combined
