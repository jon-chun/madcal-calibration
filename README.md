# Adversarial Role Structure as a Calibration Mechanism in Multi-Agent LLM Systems

Replication package for the NeurIPS 2026 submission. This repository contains all code, data, experiment results, and reproducibility scripts needed to replicate the findings reported in the paper.

## Environment Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

API keys should be set via a `.env` file or environment variables:
`OPENAI_API_KEY`, `GOOGLE_API_KEY`, `XAI_API_KEY`, `OPENROUTER_API_KEY`, `FIREWORKS_API_KEY`.

## Reproducing Paper Results

All derived statistics, tables, and figures can be reproduced from the pre-computed experiment outputs included in `results/`:

```bash
bash scripts/reproduce_all.sh
```

Individual reproducibility scripts can also be run separately (see `scripts/README.md`).

## Running Experiments

### Multi-Agent Role-Structured Debates (Step 1)

```bash
python src/step1_ai-debators_ver26.py \
    --dataset nlsy97 --ensemble commercial \
    --cases 200 --repeats 5 --concurrency 30
```

### Standard LLM Baselines

```bash
python src/standardllm_evaluation.py \
    --dataset nlsy97 --models "gpt-4o-mini" \
    --prompts system1 cot cot-nshot --cases 150
```

Thinking/reasoning models (GPT-5-mini, o-series) are automatically detected and use `max_completion_tokens` instead of `temperature`/`max_tokens`.

### Self-Consistency Majority Voting

```bash
python src/sc_majority_vote.py
```

### Full Pipeline

```bash
# Step 2: Aggregate debate transcripts into CSV
python src/step2_aggregate_transcripts_ver6.py

# Step 3: Compute statistical summaries
python src/step3_statistical_analysis_ver12.py

# Step 5: Merge standard LLM and agentic results
python src/step5_merge_standard-agenticsim_ver3_o1.py

# Steps 4/6: Generate plots
python src/step4_visualize_model_statistics_ver5_FREEZE.py
python src/step6_visualize_performance_comparison_ver6.py
```

### Ablation Studies

```bash
# Random-flip Monte Carlo ablation (Appendix I)
python scripts/ablation_random_flip.py

# GPT-5-mini boundary condition analysis (Section 4.5)
python scripts/compare_bridge_gpt5mini.py

# Cooperative debate ablation (Appendix I)
python src/step1_ai-debators_ver26.py --dataset nlsy97 --ensemble commercial --cases 100 --repeats 3 --cooperative --concurrency 30

# Sign test across model-dataset pairs
python scripts/compute_sign_test.py

# Adversarial vs cooperative comparison
python scripts/compare_adversarial_vs_cooperative.py
```

### Regenerating Paper Figures

All four data-driven paper figures can be regenerated from included results:

```bash
# Generate all figures at once
python scripts/generate_all_figures.py

# Or individually:
python scripts/generate_brd_figure.py                    # BRD comparison (3 datasets)
python scripts/generate_commercial_comparison_figure.py   # Commercial LLM comparison
python scripts/generate_cooperative_figure.py             # Adversarial vs cooperative
python scripts/generate_random_flip_figure.py             # Random-flip ablation
```

Output: `figures/*.pdf`

## Directory Structure

```
madcal-calibration/
├── configs/              # Model ensemble YAML configurations
├── data/                 # Datasets and pre-computed feature subsets
│   ├── sample_vignettes.csv          # NLSY97 30-case base sample
│   ├── nlsy97_full.csv               # NLSY97 full dataset
│   ├── nlsy97_vignettes_100.csv      # NLSY97 100-case expanded
│   ├── nlsy97_vignettes_200.csv      # NLSY97 200-case expanded
│   ├── compas_vignettes.csv          # COMPAS 30-case base sample
│   ├── compas_full_filtered.csv      # COMPAS full filtered dataset
│   ├── compas_vignettes_100.csv      # COMPAS 100-case expanded
│   ├── compas_vignettes_200.csv      # COMPAS 200-case expanded
│   ├── credit_default_vignettes_100.csv  # Credit Default 100-case sample
│   └── topn_datasets/                # Pre-computed feature subsets (5 algos × 3 counts)
├── results/              # Pre-computed experiment outputs
│   ├── *.csv / *.txt                 # Aggregated debate results, statistics
│   ├── baselines*/                   # PyCaret ML baseline results
│   ├── bridge_gpt5mini_*/            # GPT-5-mini boundary condition results
│   └── standardllm/                  # Standard LLM prompting + SC results
├── figures/              # Generated paper figures (from scripts/)
├── scripts/              # Reproducibility scripts for paper tables/figures
│   ├── reproduce_all.sh              # Run all reproducibility scripts
│   ├── generate_all_figures.py       # Generate all paper figures
│   ├── generate_brd_figure.py        # BRD comparison bar chart
│   ├── generate_commercial_comparison_figure.py  # Commercial LLM comparison
│   ├── generate_cooperative_figure.py # Adversarial vs cooperative
│   ├── generate_random_flip_figure.py # Random-flip ablation
│   ├── ablation_random_flip.py       # Random-flip Monte Carlo ablation
│   └── compare_bridge_gpt5mini.py    # GPT-5-mini bridge analysis
├── src/                  # Source code
│   ├── llm_client.py                 # Dual-path LLM client (Ollama / LiteLLM)
│   ├── model_config.py               # Model ensemble management
│   ├── dataset_config.py             # Dataset configurations (NLSY97, COMPAS, Credit Default)
│   ├── step1_ai-debators_ver26.py    # Multi-agent debate engine
│   ├── standardllm_evaluation.py     # Standard prompting baselines
│   ├── sc_majority_vote.py           # Self-consistency majority voting
│   ├── compare_debate_vs_sc.py       # Debate vs SC comparison
│   └── step2-6 scripts               # Aggregation, statistics, visualization
├── docs/                 # Technical documentation
└── requirements.txt
```

## Datasets

- **NLSY97**: Rearrest prediction (3-year window) using 22 demographic/behavioral features from the National Longitudinal Survey of Youth 1997. The target variable is rearrest, not reconviction or reoffense. Base rate: 36%.
- **COMPAS**: Two-year recidivism prediction using 9 features from the ProPublica COMPAS dataset. A variant without the algorithmic `decile_score` is also evaluated. Base rate: 45%.
- **UCI Credit Default**: Next-month default prediction using 10 LOFO-selected features from the UCI Credit Card Clients dataset (Yeh & Lien, 2009; 30,000 cases). Base rate: 22%.

Only derived vignettes are included. Full NLSY97 data requires a license from the Bureau of Labor Statistics. COMPAS raw data can be re-downloaded via `python src/prepare_compas_dataset.py`. UCI Credit Default data is publicly available from the UCI Machine Learning Repository.

## Key Design Decisions

- **Thinking model support**: Automatic detection and routing of reasoning-class models (GPT-5-mini, o-series) that require `max_completion_tokens` and `reasoning_effort` instead of standard `temperature`/`max_tokens`.
- **Model routing**: Automatic detection of Ollama (local) vs. commercial API based on model ID format (`:` for Ollama, `/` for cloud).
- **JSON error recovery**: 3-layer fallback (stdlib JSON, `json_repair`, regex) for malformed LLM responses.
- **Parallelization**: Semaphore-bounded concurrency with fair distribution across API providers.
- **Two pipeline generations**: FREEZE versions (ver25) reproduce original paper results exactly; ver26 supports multi-dataset and commercial API experiments.

## License

MIT License
