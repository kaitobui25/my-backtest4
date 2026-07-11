from __future__ import annotations

import argparse
from pathlib import Path

from btsearch.config import RunSettings
from btsearch.data_loader import load_ohlcv
from btsearch.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fast multi-strategy vectorbt research engine."
    )
    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--mode", choices=("coarse", "full"), default="coarse"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--fee", type=float, default=0.0005)
    parser.add_argument("--slippage", type=float, default=0.0002)
    parser.add_argument("--min-trades-ranking", type=int, default=100)
    parser.add_argument("--max-configs", type=int, default=0)
    parser.add_argument("--output", default="output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = RunSettings(
        fee=args.fee,
        slippage=args.slippage,
        batch_size=args.batch_size,
        min_trades_ranking=args.min_trades_ranking,
    )
    df = load_ohlcv(args.data)
    print(
        f"Candles: {len(df):,} | "
        f"{df.index.min()} -> {df.index.max()}"
    )
    print(f"Volume: {'yes' if 'volume' in df.columns else 'no'}")
    run_pipeline(
        df=df,
        output_dir=Path(args.output).resolve(),
        mode=args.mode,
        resume=args.resume,
        max_configs=args.max_configs,
        settings=settings,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
