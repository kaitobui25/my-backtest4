from __future__ import annotations

import json
from pathlib import Path
import shutil
import time

import pandas as pd

from btsearch.config import RunSettings, StrategyConfig
from btsearch.engine import VectorBTBatchEngine
from btsearch.grids import (
    build_coarse_configs,
    build_full_configs,
    refine_configs,
)
from btsearch.ranking import (
    aggregate_walk_forward,
    main_ranking,
    rank_walk_forward_expectancy,
    rank_walk_forward_robust,
    top_by_family,
)
from btsearch.report import write_final_report, write_json
from btsearch.strategies import REGISTRY
from btsearch.walk_forward import run_walk_forward


def _rows_to_configs(rows: pd.DataFrame) -> list[StrategyConfig]:
    return [
        StrategyConfig(
            family=str(row["family"]),
            direction=str(row["direction"]),
            params=json.loads(row["params_json"]),
        )
        for _, row in rows.iterrows()
    ]


def run_pipeline(
    df: pd.DataFrame,
    output_dir: Path,
    mode: str,
    resume: bool,
    max_configs: int,
    settings: RunSettings,
) -> None:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not resume:
        stale_names = (
            "01_phase1_insufficient_sample.csv",
            "02_phase1_top_by_family.csv",
            "04_walk_forward_results.parquet",
            "04_walk_forward_results.csv",
            "05_top_expectancy.csv",
            "06_top_robust.csv",
            "BEST_EXPECTANCY_CONFIG.json",
            "BEST_ROBUST_CONFIG.json",
            "FINAL_REPORT.txt",
        )
        for name in stale_names:
            path = output_dir / name
            if path.exists():
                path.unlink()
        wf_dir = output_dir / "walk_forward_checkpoints"
        if wf_dir.exists():
            shutil.rmtree(wf_dir)

    has_volume = "volume" in df.columns
    configs = (
        build_coarse_configs(has_volume)
        if mode == "coarse"
        else build_full_configs(has_volume)
    )
    if max_configs > 0:
        configs = configs[:max_configs]

    # Phase 1 uses the first 60% of history only. Later periods remain for
    # expanding walk-forward selection and validation.
    phase1_end = int(len(df) * 0.60)
    phase1_df = df.iloc[:phase1_end]

    engine = VectorBTBatchEngine(settings)
    phase1 = engine.run(
        phase1_df,
        configs,
        output_dir / "01_phase1_all_results.parquet",
        resume,
        "PHASE1",
    )
    phase1.to_parquet(
        output_dir / "01_phase1_all_results.parquet", index=False
    )
    phase1.to_csv(
        output_dir / "01_phase1_all_results.csv", index=False
    )
    phase1.loc[phase1["trades"] < settings.min_trades_ranking].to_csv(
        output_dir / "01_phase1_insufficient_sample.csv", index=False
    )

    family_top = top_by_family(
        phase1,
        min_trades=settings.min_trades_ranking,
        per_family=settings.top_per_family,
    )
    family_top.to_csv(
        output_dir / "02_phase1_top_by_family.csv", index=False
    )

    seeds = (
        family_top.groupby("family", group_keys=False)
        .head(settings.phase2_seed_per_family)
        if not family_top.empty
        else main_ranking(phase1, 30).head(20)
    )
    seed_configs = _rows_to_configs(seeds)
    refined_configs = refine_configs(seed_configs)

    phase2 = engine.run(
        phase1_df,
        refined_configs,
        output_dir / "03_phase2_refined_results.parquet",
        resume,
        "PHASE2",
    )
    phase2.to_parquet(
        output_dir / "03_phase2_refined_results.parquet", index=False
    )
    phase2.to_csv(
        output_dir / "03_phase2_refined_results.csv", index=False
    )

    candidate_table = pd.concat(
        [
            top_by_family(
                phase1,
                settings.min_trades_ranking,
                settings.top_per_family,
            ),
            top_by_family(
                phase2,
                settings.min_trades_ranking,
                settings.top_per_family,
            ),
        ],
        ignore_index=True,
    ).drop_duplicates("config_id")

    if candidate_table.empty:
        candidate_table = main_ranking(phase1, 30).head(100)

    candidates = _rows_to_configs(candidate_table)
    first_validation_start = df.index[phase1_end]
    walk_forward = run_walk_forward(
        df,
        candidates,
        settings,
        output_dir,
        resume,
        first_validation_start=first_validation_start,
    )
    walk_forward.to_parquet(
        output_dir / "04_walk_forward_results.parquet", index=False
    )
    walk_forward.to_csv(
        output_dir / "04_walk_forward_results.csv", index=False
    )

    aggregate = aggregate_walk_forward(walk_forward)
    top_expectancy = rank_walk_forward_expectancy(
        aggregate, settings.min_trades_ranking
    )
    top_robust = rank_walk_forward_robust(
        aggregate, settings.min_trades_ranking
    )
    top_expectancy.to_csv(
        output_dir / "05_top_expectancy.csv", index=False
    )
    top_robust.to_csv(
        output_dir / "06_top_robust.csv", index=False
    )

    write_json(
        output_dir / "BEST_EXPECTANCY_CONFIG.json",
        None if top_expectancy.empty else top_expectancy.iloc[0],
    )
    write_json(
        output_dir / "BEST_ROBUST_CONFIG.json",
        None if top_robust.empty else top_robust.iloc[0],
    )

    runtime = time.perf_counter() - started
    write_final_report(
        output_dir / "FINAL_REPORT.txt",
        total_families=len(REGISTRY) - (0 if has_volume else 2),
        phase1_configs=len(phase1),
        phase2_configs=len(phase2),
        runtime_seconds=runtime,
        top_expectancy=top_expectancy,
        top_robust=top_robust,
        fee=settings.fee,
        slippage=settings.slippage,
        target_reference_r=settings.target_reference_r,
    )

    print("")
    print("=" * 72)
    print((output_dir / "FINAL_REPORT.txt").read_text(encoding="utf-8"))
