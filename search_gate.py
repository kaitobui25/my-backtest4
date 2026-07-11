from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

import search as core


MIN_TRAIN_TRADES = 300
MIN_VALIDATION_TRADES = 50
MIN_OOS_TRADES = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safe BTC 15m strategy screener using vectorbt."
    )
    parser.add_argument("--data", required=True, help="Path to OHLCV parquet file.")
    parser.add_argument("--mode", choices=["instant", "full"], default="instant")
    parser.add_argument("--fee", type=float, default=0.0005)
    parser.add_argument("--slippage", type=float, default=0.0002)
    parser.add_argument("--target-r", type=float, default=core.TARGET_R)
    parser.add_argument("--min-train-trades", type=int, default=MIN_TRAIN_TRADES)
    parser.add_argument(
        "--min-validation-trades", type=int, default=MIN_VALIDATION_TRADES
    )
    parser.add_argument("--min-oos-trades", type=int, default=MIN_OOS_TRADES)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default="output")
    return parser.parse_args()


def dedupe_configs(configs: list[core.Config]) -> list[core.Config]:
    """Remove configs that are behaviorally identical."""
    seen: set[tuple] = set()
    unique: list[core.Config] = []
    for cfg in configs:
        # Bollinger logic in search.py does not use EMA, so normalize it.
        ema_window = 100 if cfg.strategy == "bollinger_reentry" else cfg.ema_window
        key = (
            cfg.strategy,
            ema_window,
            cfg.signal_window,
            cfg.signal_level,
            cfg.atr_window,
            cfg.sl_atr,
            cfg.rr,
            cfg.direction,
        )
        if key in seen:
            continue
        seen.add(key)
        if ema_window != cfg.ema_window:
            cfg = core.Config(
                config_id=cfg.config_id,
                strategy=cfg.strategy,
                ema_window=ema_window,
                signal_window=cfg.signal_window,
                signal_level=cfg.signal_level,
                atr_window=cfg.atr_window,
                sl_atr=cfg.sl_atr,
                rr=cfg.rr,
                direction=cfg.direction,
            )
        unique.append(cfg)
    return unique


def add_flags(df: pd.DataFrame, target: float, min_trades: int) -> pd.DataFrame:
    result = df.copy()
    result["sample_ok"] = result["trades"] >= min_trades
    result["target_hit"] = result["expectancy_r"] >= target
    result["eligible"] = result["sample_ok"] & result["target_hit"]
    return result.sort_values(
        ["eligible", "sample_ok", "expectancy_r", "trades"],
        ascending=[False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)


def print_table(
    df: pd.DataFrame,
    title: str,
    target: float,
    min_trades: int,
    n: int = 10,
) -> None:
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)
    columns = [
        "config_id", "strategy", "direction", "ema_window", "signal_window",
        "signal_level", "sl_atr", "rr", "trades", "winrate",
        "avg_win_r", "avg_loss_r", "expectancy_r", "profit_factor_r",
        "max_drawdown_r",
    ]
    stable = df.loc[df["sample_ok"]]
    source = stable if not stable.empty else df
    show = source.head(n)[columns].copy()
    for column in [
        "winrate", "avg_win_r", "avg_loss_r", "expectancy_r",
        "profit_factor_r", "max_drawdown_r",
    ]:
        show[column] = show[column].map(
            lambda value: f"{value:.4f}" if pd.notna(value) else "nan"
        )
    print(show.to_string(index=False))
    print(
        f"\nExpectancy >= {target:+.3f}R: {int(df['target_hit'].sum())}/{len(df)} | "
        f"trades >= {min_trades}: {int(df['sample_ok'].sum())}/{len(df)} | "
        f"đạt cả hai: {int(df['eligible'].sum())}/{len(df)}"
    )


def clear_outputs(output_dir: Path) -> None:
    for name in [
        "01_train_all.parquet", "01_train_all.csv",
        "02_validation_shortlist.parquet", "02_validation_shortlist.csv",
        "03_final_oos.parquet", "03_final_oos.csv",
        "BEST_FROZEN_CONFIG.json", "FINAL_RESULT.txt",
    ]:
        (output_dir / name).unlink(missing_ok=True)


def stop_before_oos(
    output_dir: Path,
    stage: str,
    reason: str,
    args: argparse.Namespace,
    split_method: str,
) -> None:
    report = f"""RESULT: STOPPED BEFORE FINAL OOS

Stopped after: {stage}
Reason: {reason}

Target: {args.target_r:+.6f}R/trade
Minimum TRAIN trades: {args.min_train_trades}
Minimum VALIDATION trades: {args.min_validation_trades}
Fee per fill: {args.fee}
Slippage per fill: {args.slippage}
Split: {split_method}

FINAL OOS was not opened. This strategy family did not pass the screening gates.
"""
    (output_dir / "FINAL_RESULT.txt").write_text(report, encoding="utf-8")
    print("\n" + report)


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_outputs(output_dir)

    data_path = Path(args.data).expanduser().resolve()
    print(f"vectorbt: {getattr(core.vbt, '__version__', 'unknown')}")
    print(f"Data: {data_path}")
    print(f"Mode: {args.mode}")
    print(f"Fee/fill: {args.fee:.6f}; slippage/fill: {args.slippage:.6f}")
    print(f"Target: {args.target_r:+.3f}R/trade")
    print(
        f"Minimum trades: TRAIN={args.min_train_trades}, "
        f"VALIDATION={args.min_validation_trades}, OOS={args.min_oos_trades}"
    )

    data = core.load_ohlcv(data_path)
    configs = dedupe_configs(core.make_configs(args.mode))
    train, validation, oos, split_method = core.split_data(data)
    print(f"Candles: {len(data):,} | {data.index.min()} -> {data.index.max()}")
    print(f"Configs after dedupe: {len(configs):,}")
    print(split_method)

    train_result = core.run_configs(
        train, configs, args.fee, args.slippage, args.batch_size, "TRAIN"
    )
    train_result = add_flags(train_result, args.target_r, args.min_train_trades)
    train_result.to_parquet(output_dir / "01_train_all.parquet", index=False)
    train_result.to_csv(output_dir / "01_train_all.csv", index=False)
    print_table(
        train_result, "TRAIN — CONFIGS WITH ENOUGH TRADES",
        args.target_r, args.min_train_trades,
    )

    train_eligible = train_result.loc[train_result["eligible"]]
    if train_eligible.empty:
        stop_before_oos(
            output_dir,
            "TRAIN",
            "No config met both the expectancy target and minimum TRAIN trades.",
            args,
            split_method,
        )
        print(f"Total runtime: {time.perf_counter() - started:.2f}s")
        return 0

    lookup = core.config_lookup(configs)
    shortlist_ids = train_eligible.head(20)["config_id"].tolist()
    shortlist = [lookup[config_id] for config_id in shortlist_ids]
    validation_result = core.run_configs(
        validation, shortlist, args.fee, args.slippage,
        min(args.batch_size, 20), "VALIDATION",
    )
    validation_result = add_flags(
        validation_result, args.target_r, args.min_validation_trades
    )
    validation_result.to_parquet(
        output_dir / "02_validation_shortlist.parquet", index=False
    )
    validation_result.to_csv(
        output_dir / "02_validation_shortlist.csv", index=False
    )
    print_table(
        validation_result, "VALIDATION — TRAIN WINNERS ONLY",
        args.target_r, args.min_validation_trades,
    )

    validation_eligible = validation_result.loc[validation_result["eligible"]]
    if validation_eligible.empty:
        stop_before_oos(
            output_dir,
            "VALIDATION",
            "No TRAIN winner kept the target with enough VALIDATION trades.",
            args,
            split_method,
        )
        print(f"Total runtime: {time.perf_counter() - started:.2f}s")
        return 0

    final_ids = validation_eligible.head(5)["config_id"].tolist()
    final_configs = [lookup[config_id] for config_id in final_ids]
    oos_result = core.run_configs(
        oos, final_configs, args.fee, args.slippage,
        min(args.batch_size, 5), "FINAL_OOS",
    )
    oos_result = add_flags(oos_result, args.target_r, args.min_oos_trades)
    frozen_order = {config_id: index for index, config_id in enumerate(final_ids)}
    oos_result["_frozen_order"] = oos_result["config_id"].map(frozen_order)
    oos_result = oos_result.sort_values("_frozen_order").drop(columns="_frozen_order")
    oos_result.to_parquet(output_dir / "03_final_oos.parquet", index=False)
    oos_result.to_csv(output_dir / "03_final_oos.csv", index=False)
    print_table(
        oos_result, "FINAL OOS — FROZEN VALIDATION WINNERS",
        args.target_r, args.min_oos_trades, n=5,
    )

    best_id = final_ids[0]
    train_row = train_result.loc[train_result["config_id"] == best_id].iloc[0]
    val_row = validation_result.loc[validation_result["config_id"] == best_id].iloc[0]
    oos_row = oos_result.loc[oos_result["config_id"] == best_id].iloc[0]
    best = asdict(lookup[best_id])
    best.update({
        "train": {"trades": int(train_row["trades"]), "expectancy_r": float(train_row["expectancy_r"])},
        "validation": {"trades": int(val_row["trades"]), "expectancy_r": float(val_row["expectancy_r"])},
        "oos": {"trades": int(oos_row["trades"]), "expectancy_r": float(oos_row["expectancy_r"])},
        "target_r": args.target_r,
        "oos_target_hit": bool(oos_row["target_hit"]),
        "oos_sample_ok": bool(oos_row["sample_ok"]),
        "final_pass": bool(oos_row["eligible"]),
        "fee_per_fill": args.fee,
        "slippage_per_fill": args.slippage,
        "split_method": split_method,
    })
    (output_dir / "BEST_FROZEN_CONFIG.json").write_text(
        json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    status = (
        "ĐẠT TARGET TRÊN FINAL OOS"
        if best["final_pass"] else "CHƯA ĐẠT TARGET TRÊN FINAL OOS"
    )
    report = f"""RESULT: {status}

Frozen config selected by VALIDATION:
{json.dumps(asdict(lookup[best_id]), indent=2, ensure_ascii=False)}

TRAIN:      trades={int(train_row['trades'])}, expectancy={float(train_row['expectancy_r']):+.6f}R
VALIDATION: trades={int(val_row['trades'])}, expectancy={float(val_row['expectancy_r']):+.6f}R
FINAL OOS:  trades={int(oos_row['trades'])}, expectancy={float(oos_row['expectancy_r']):+.6f}R

Target: {args.target_r:+.6f}R/trade
Minimum trades: TRAIN={args.min_train_trades}, VALIDATION={args.min_validation_trades}, OOS={args.min_oos_trades}
Fee per fill: {args.fee}
Slippage per fill: {args.slippage}
"""
    (output_dir / "FINAL_RESULT.txt").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"Total runtime: {time.perf_counter() - started:.2f}s")
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
