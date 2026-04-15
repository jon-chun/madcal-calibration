#!/usr/bin/env python3
"""
Aggregate ver26 debate JSON transcripts into a flat CSV.

Reads all transcript_row-{N}_ver-{R}.json files from a ver26 transcript directory
and produces a CSV with one row per debate, compatible with compare_debate_vs_sc.py.

Usage:
    python src/aggregate_ver26_transcripts.py \
        --transcript-dir transcripts_ver26_nlsy97_20260212-234350 \
        --dataset nlsy97 \
        --output results/debate_aggregate_nlsy97_100.csv
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd


DATASET_TARGET = {
    "nlsy97": "y_arrestedafter2002",
    "compas": "two_year_recid",
    "compas_nodecile": "two_year_recid",
    "credit_default": "default_payment_next_month",
}

# Reverse mapping: cleaned directory name -> original model ID.
# clean_model_name() is lossy (destroys /, -, .), so we maintain
# explicit mappings for models where heuristic reconstruction fails.
DIR_TO_MODEL = {
    # OpenRouter OSS models
    "openrouter_meta_llama_llama_3_1_8b_instruct": "openrouter/meta-llama/llama-3.1-8b-instruct",
    "openrouter_qwen_qwen_2_5_7b_instruct": "openrouter/qwen/qwen-2.5-7b-instruct",
    "openrouter_microsoft_phi_4": "openrouter/microsoft/phi-4",
    "openrouter_google_gemma_2_9b_it": "openrouter/google/gemma-2-9b-it",
    # xAI
    "xai_grok_4_1_fast_non_reasoning_latest": "xai/grok-4.1-fast-non-reasoning-latest",
}


def parse_transcript(json_path: Path, model_name: str, dataset: str) -> dict | None:
    """Parse a single transcript JSON and return a flat dict."""
    # Extract row and ver from filename: transcript_row-0_ver-1.json
    match = re.search(r"transcript_row-(\d+)_ver-(\d+)\.json", json_path.name)
    if not match:
        return None

    case_id = int(match.group(1))
    repeat_id = int(match.group(2))

    try:
        data = json.loads(json_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    case = data.get("case", {})
    ruling = data.get("final_ruling", {})

    target_col = DATASET_TARGET.get(dataset, "y_arrestedafter2002")
    ground_truth = case.get(target_col)

    prediction = ruling.get("prediction", "").strip().upper()
    confidence = ruling.get("confidence")

    # Judge opinion evolution length
    opinions = data.get("judge_opinion_evolution", [])
    n_turns = len(opinions)

    return {
        "model_name": model_name,
        "case_id": case_id,
        "repeat_id": repeat_id,
        "dataset": dataset,
        "prediction": prediction,
        "confidence": confidence,
        "ground_truth": ground_truth,
        "n_judge_opinions": n_turns,
    }


def main():
    parser = argparse.ArgumentParser(description="Aggregate ver26 debate transcripts")
    parser.add_argument("--transcript-dir", type=str, required=True,
                        help="Path to ver26 transcript directory")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=list(DATASET_TARGET.keys()))
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path")
    args = parser.parse_args()

    transcript_dir = Path(args.transcript_dir)
    if not transcript_dir.exists():
        raise FileNotFoundError(f"Transcript dir not found: {transcript_dir}")

    rows = []
    # Each model subdirectory contains transcript JSONs
    for model_dir in sorted(transcript_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name == "raw_api_responses":
            continue

        # Convert dir name back to model name (underscores -> slashes/dots)
        # e.g., gemini_gemini_2_5_flash -> gemini/gemini-2.5-flash
        # gpt_4o_mini -> gpt-4o-mini
        # gpt_4_1_mini -> gpt-4.1-mini
        dir_name = model_dir.name

        # Model name reconstruction: use explicit mapping first, then heuristics
        if dir_name in DIR_TO_MODEL:
            model_name = DIR_TO_MODEL[dir_name]
        elif dir_name == "gpt_4o_mini":
            model_name = "gpt-4o-mini"
        elif dir_name == "gpt_4_1_mini":
            model_name = "gpt-4.1-mini"
        elif dir_name == "gpt_5_mini":
            model_name = "gpt-5-mini"
        elif dir_name.startswith("gemini_"):
            # gemini_gemini_2_5_flash -> gemini/gemini-2.5-flash
            parts = dir_name.split("_", 1)  # ['gemini', 'gemini_2_5_flash']
            rest = parts[1]  # gemini_2_5_flash
            # Replace version-like patterns: 2_5 -> 2.5
            rest = re.sub(r"(\d+)_(\d+)", r"\1.\2", rest)
            rest = rest.replace("_", "-")
            model_name = f"{parts[0]}/{rest}"
        else:
            model_name = dir_name.replace("_", "-")

        json_files = sorted(model_dir.glob("transcript_row-*_ver-*.json"))
        print(f"  {model_name}: {len(json_files)} JSON files")

        for jf in json_files:
            row = parse_transcript(jf, model_name, args.dataset)
            if row:
                rows.append(row)

    df = pd.DataFrame(rows)
    print(f"\nTotal rows: {len(df)}")
    print(f"Models: {df['model_name'].unique().tolist()}")
    print(f"Cases: {df['case_id'].nunique()}")
    print(f"Repeats per model-case: {df.groupby(['model_name', 'case_id']).size().describe()}")

    # Compute accuracy
    valid = df[df["prediction"].isin(["YES", "NO"])]
    valid_bool_truth = valid["ground_truth"].astype(bool)
    correct = ((valid["prediction"] == "YES") & valid_bool_truth) | \
              ((valid["prediction"] == "NO") & ~valid_bool_truth)
    print(f"\nOverall accuracy: {correct.mean():.3f} ({correct.sum()}/{len(correct)})")

    for model in sorted(df["model_name"].unique()):
        m = valid[valid["model_name"] == model]
        mt = m["ground_truth"].astype(bool)
        mc = ((m["prediction"] == "YES") & mt) | ((m["prediction"] == "NO") & ~mt)
        yes_pct = (m["prediction"] == "YES").mean()
        print(f"  {model}: Acc={mc.mean():.3f} ({mc.sum()}/{len(mc)}) YES%={yes_pct:.3f}")

    # Save
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(f"results/debate_aggregate_{args.dataset}_100.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
