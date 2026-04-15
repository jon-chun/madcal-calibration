#!/usr/bin/env python3
"""Compute sign test for debate calibration across model-dataset pairs.

Tests whether debate achieves lowest BRD (Base-Rate Deviation) across
model-dataset pairs more often than chance (H0: p=1/3 for 3 methods).

Usage:
  python scripts/compute_sign_test.py

  Fill in BRD values from experiment results before running.
"""
from scipy import stats

# ── BRD values per (model, dataset) ───────────────────
# Format: (model, dataset): {"zs": BRD, "sc": BRD, "debate": BRD}
# BRD = |YES% - base_rate|
#
# BASE RATES: nlsy97=0.36, compas=0.45, credit_default=0.22

RESULTS = {
    # ── Existing pairs (from results/comparison_{dataset}_100.csv) ──
    # BRD values verified against actual experiment data.
    # Gemini SC data is effective K~2 (96.4% JSON malformation) but
    # ZS and debate data are reliable.

    # NLSY97 (base rate = 0.36)
    ("GPT-4o Mini",   "NLSY97"):  {"zs": 0.470, "sc": 0.490, "debate": 0.023},
    ("GPT-4.1 Mini",  "NLSY97"):  {"zs": 0.360, "sc": 0.380, "debate": 0.303},
    ("Gemini 2.5",    "NLSY97"):  {"zs": 0.460, "sc": 0.460, "debate": 0.191},

    # COMPAS (base rate = 0.45)
    ("GPT-4o Mini",   "COMPAS"):  {"zs": 0.070, "sc": 0.080, "debate": 0.077},
    ("GPT-4.1 Mini",  "COMPAS"):  {"zs": 0.080, "sc": 0.060, "debate": 0.073},
    ("Gemini 2.5",    "COMPAS"):  {"zs": 0.110, "sc": 0.110, "debate": 0.098},

    # Credit Default (base rate = 0.22) — from public repo experiments
    ("GPT-4o Mini",   "Credit Default"):  {"zs": 0.020, "sc": 0.020, "debate": 0.090},
    ("GPT-4.1 Mini",  "Credit Default"):  {"zs": 0.040, "sc": 0.040, "debate": 0.013},

    # ── NEW OSS pairs from Option A (NLSY97 only) ──
    # ZS: from standardllm_metrics_nlsy97_20260227_193828.csv (T=0.0)
    # SC: majority vote K=59 from standardllm results (T=0.7)
    # Debate: from debate_aggregate_nlsy97_oss.csv (3 reps)
    # SC K=59 complete (20260228_011559), all values verified
    ("Llama 3.1 8B",  "NLSY97"):  {"zs": 0.527, "sc": 0.530, "debate": 0.303},
    ("Qwen 2.5 7B",   "NLSY97"):  {"zs": 0.090, "sc": 0.060, "debate": 0.250},
    ("Phi-4 14B",     "NLSY97"):  {"zs": 0.590, "sc": 0.590, "debate": 0.040},
    ("Gemma 2 9B",    "NLSY97"):  {"zs": 0.560, "sc": 0.560, "debate": 0.100},
}


def main():
    # Filter to pairs with complete data
    complete = {
        k: v for k, v in RESULTS.items()
        if all(val is not None for val in v.values())
    }

    n = len(complete)
    wins = sum(
        1 for brds in complete.values()
        if brds["debate"] < min(brds["zs"], brds["sc"])
    )
    ties = sum(
        1 for brds in complete.values()
        if brds["debate"] == min(brds["zs"], brds["sc"])
    )

    print(f"Complete pairs: {n}")
    print(f"Debate wins (lowest BRD): {wins}/{n}")
    print(f"Ties: {ties}/{n}")
    print()

    # One-sided binomial test: debate wins > chance (1/3)
    # Using exact binomial test
    p_binom = stats.binomtest(wins, n, 1/3, alternative="greater").pvalue
    print(f"Binomial test (H0: p=1/3): p = {p_binom:.4f}")

    # Also report simple sign test (debate < best-of-rest)
    p_sign = stats.binomtest(wins, n, 0.5, alternative="greater").pvalue
    print(f"Sign test (H0: p=0.5):     p = {p_sign:.4f}")

    # Per-pair detail
    print(f"\n{'Model':<20} {'Dataset':<18} {'ZS':>6} {'SC':>6} {'Debate':>6}  Winner")
    print("-" * 75)
    for (model, ds), brds in sorted(complete.items()):
        winner = min(brds, key=brds.get)
        marker = " <<<" if winner == "debate" else ""
        print(f"{model:<20} {ds:<18} {brds['zs']:>6.3f} {brds['sc']:>6.3f} {brds['debate']:>6.3f}  {winner}{marker}")

    # Summary
    print(f"\nWith all {n} pairs:")
    if p_binom < 0.05:
        print(f"  SIGNIFICANT at p<0.05 (p={p_binom:.4f})")
    else:
        print(f"  NOT significant at p<0.05 (p={p_binom:.4f})")
        remaining = 12 - n
        if remaining > 0:
            print(f"  {remaining} OSS model pairs still pending (fill in None values)")


if __name__ == "__main__":
    main()
