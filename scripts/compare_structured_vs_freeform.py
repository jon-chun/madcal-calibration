#!/usr/bin/env python3
"""Compare structured (adversarial) vs free-form debate calibration.

Computes BRD for both debate variants and reports which achieves
better calibration (lower BRD = closer to base rate).

Usage:
  python scripts/compare_structured_vs_freeform.py
  python scripts/compare_structured_vs_freeform.py --adv-dir results/ --freeform-dir results/
"""
import argparse
from pathlib import Path

import pandas as pd

NLSY97_BASE_RATE = 0.36
TARGET_COL = "y_arrestedafter2002"


def compute_brd(df: pd.DataFrame) -> float:
    """BRD = |YES% - base_rate|."""
    if "judge_prediction" in df.columns:
        yes_pct = (df["judge_prediction"] == "YES").mean()
    elif "prediction" in df.columns:
        yes_pct = (df["prediction"] == "YES").mean()
    else:
        raise ValueError(f"No prediction column in {df.columns.tolist()}")
    return abs(yes_pct - NLSY97_BASE_RATE)


def main():
    parser = argparse.ArgumentParser(
        description="Compare adversarial vs free-form debate calibration (BRD)")
    parser.add_argument("--adv-dir", default="results",
        help="Dir with adversarial aggregate CSVs")
    parser.add_argument("--freeform-dir", default="results",
        help="Dir with free-form aggregate CSVs")
    args = parser.parse_args()

    adv_base = Path(args.adv_dir)
    ff_base = Path(args.freeform_dir)

    print(f"{'Model':<45} {'Adv BRD':>8} {'FF BRD':>7}  {'Winner':<12}")
    print("-" * 78)

    adv_wins = 0
    ff_wins = 0
    total = 0

    # Find adversarial aggregate files for NLSY97
    adv_files = sorted(adv_base.glob("debate_aggregate_nlsy97_*.csv"))
    ff_files = sorted(ff_base.glob("freeform_aggregate_nlsy97_*.csv"))

    if not adv_files:
        print(f"  [SKIP] No adversarial aggregate CSVs in {adv_base}")
        return
    if not ff_files:
        print(f"  [SKIP] No free-form aggregate CSVs in {ff_base}")
        return

    # Load all into single frames
    adv_df = pd.concat([pd.read_csv(f) for f in adv_files], ignore_index=True)
    ff_df = pd.concat([pd.read_csv(f) for f in ff_files], ignore_index=True)

    for model in sorted(adv_df["model_name"].unique()):
        adv_sub = adv_df[adv_df["model_name"] == model]
        ff_sub = ff_df[ff_df["model_name"] == model]

        if ff_sub.empty:
            continue

        adv_brd = compute_brd(adv_sub)
        ff_brd = compute_brd(ff_sub)
        winner = "adversarial" if adv_brd < ff_brd else "free-form"
        if adv_brd < ff_brd:
            adv_wins += 1
        else:
            ff_wins += 1
        total += 1

        print(f"{model:<45} {adv_brd:>8.3f} {ff_brd:>7.3f}  {winner}")

    print(f"\nAdversarial wins: {adv_wins}/{total}")
    print(f"Free-form wins:   {ff_wins}/{total}")
    if total > 0 and adv_wins > ff_wins:
        print("\nConclusion: Adversarial role structure improves calibration")
        print("beyond generic multi-agent interaction.")
    elif total > 0:
        print("\nConclusion: Free-form debate achieves comparable calibration.")
        print("Role structure may not be the key variable.")


if __name__ == "__main__":
    main()
