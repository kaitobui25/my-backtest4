from __future__ import annotations

import json
import sys
from pathlib import Path
import pandas as pd


def safe_print(text: str) -> None:
    """Print ``text`` to stdout, tolerating non-UTF-8 consoles.

    On consoles whose encoding cannot represent every character (e.g. cp932
    on Japanese Windows), ``print`` raises ``UnicodeEncodeError``. In that
    case the text is re-encoded with replacement characters so the report
    still reaches the user instead of crashing the run.
    """
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        safe_text = text.encode(encoding, errors="replace").decode(
            encoding, errors="replace"
        )
        print(safe_text)


def _safe_float(row: pd.Series, key: str) -> float:
    value = row.get(key)
    return float(value) if pd.notna(value) else float("nan")


def write_json(path: Path, row: pd.Series | None) -> None:
    if row is None:
        path.write_text("{}\n", encoding="utf-8")
        return
    data = row.to_dict()
    for key, value in list(data.items()):
        if hasattr(value, "item"):
            data[key] = value.item()
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def write_final_report(
    path: Path,
    total_families: int,
    phase1_configs: int,
    phase2_configs: int,
    runtime_seconds: float,
    top_expectancy: pd.DataFrame,
    top_robust: pd.DataFrame,
    fee: float,
    slippage: float,
    target_reference_r: float,
) -> None:
    best_exp = None if top_expectancy.empty else top_expectancy.iloc[0]
    best_rob = None if top_robust.empty else top_robust.iloc[0]

    lines = [
        "MY-BACKTEST4 — FINAL RESEARCH REPORT",
        "=" * 72,
        f"Strategy families tested: {total_families}",
        f"Phase 1 configs tested: {phase1_configs}",
        f"Phase 2 configs tested: {phase2_configs}",
        f"Total runtime seconds: {runtime_seconds:.2f}",
        f"Fee per fill: {fee}",
        f"Slippage per fill: {slippage}",
        f"Reference target: {target_reference_r:+.3f}R/trade",
        "",
    ]

    if best_exp is None:
        lines += [
            "BEST EXPECTANCY",
            "No walk-forward candidate had enough validation trades.",
            "",
        ]
    else:
        lines += [
            "BEST EXPECTANCY",
            f"Config: {best_exp['config_id']}",
            f"Family: {best_exp['family']}",
            f"Direction: {best_exp['direction']}",
            f"Mean validation expectancy: "
            f"{_safe_float(best_exp, 'macro_expectancy_r'):+.6f}R",
            f"Median validation expectancy: "
            f"{_safe_float(best_exp, 'median_expectancy_r'):+.6f}R",
            f"Worst validation fold: "
            f"{_safe_float(best_exp, 'worst_fold_expectancy_r'):+.6f}R",
            f"Positive fold ratio: "
            f"{_safe_float(best_exp, 'positive_fold_ratio'):.2%}",
            f"Validation trades: {int(best_exp['total_validation_trades'])}",
            f"Mean win rate: {_safe_float(best_exp, 'validation_mean_winrate'):.2%}",
            f"Mean avg win: {_safe_float(best_exp, 'validation_mean_avg_win_r'):+.4f}R",
            f"Mean avg loss: {_safe_float(best_exp, 'validation_mean_avg_loss_r'):+.4f}R",
            f"Mean profit factor: {_safe_float(best_exp, 'validation_mean_profit_factor_r'):.4f}",
            f"Max validation drawdown: {_safe_float(best_exp, 'validation_max_drawdown_r'):.4f}R",
            f"Modeled cost/trade: {_safe_float(best_exp, 'cost_r_per_trade'):.4f}R",
            f"Parameters: {best_exp['params_json']}",
            "",
        ]

    if best_rob is None:
        lines += [
            "BEST ROBUST",
            "No robust candidate had enough validation trades.",
            "",
        ]
    else:
        lines += [
            "BEST ROBUST",
            f"Config: {best_rob['config_id']}",
            f"Family: {best_rob['family']}",
            f"Direction: {best_rob['direction']}",
            f"Robust score: {_safe_float(best_rob, 'robust_score'):+.6f}",
            f"Mean validation expectancy: "
            f"{_safe_float(best_rob, 'macro_expectancy_r'):+.6f}R",
            f"Positive fold ratio: "
            f"{_safe_float(best_rob, 'positive_fold_ratio'):.2%}",
            f"Validation trades: {int(best_rob['total_validation_trades'])}",
            f"Mean win rate: {_safe_float(best_rob, 'validation_mean_winrate'):.2%}",
            f"Mean profit factor: {_safe_float(best_rob, 'validation_mean_profit_factor_r'):.4f}",
            f"Max validation drawdown: {_safe_float(best_rob, 'validation_max_drawdown_r'):.4f}R",
            f"Modeled cost/trade: {_safe_float(best_rob, 'cost_r_per_trade'):.4f}R",
            f"Parameters: {best_rob['params_json']}",
            "",
        ]

    lines.append("CONCLUSION")
    if best_exp is None:
        lines.append(
            "No statistically usable candidate was found in the tested space."
        )
    else:
        value = _safe_float(best_exp, "macro_expectancy_r")
        if value >= target_reference_r:
            lines.append(
                f"At least one candidate exceeded "
                f"{target_reference_r:+.3f}R/trade in mean walk-forward validation."
            )
        else:
            lines.append(
                f"Best mean walk-forward expectancy was {value:+.6f}R/trade; "
                f"no candidate reached {target_reference_r:+.3f}R/trade."
            )
    lines += [
        "",
        "Fast-engine candidates must still be rerun in the Exact Engine before live use.",
    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
