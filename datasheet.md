# Datasheet for the MADCal Replication Package

This datasheet follows the *Datasheets for Datasets* template
(Gebru et al., "Datasheets for Datasets," CACM 2021) for the
replication package accompanying the NeurIPS 2026 paper *"Adversarial
Role Structure as a Calibration Mechanism in Multi-Agent LLM Systems:
A Sharp Boundary at Reasoning-Class Models."* The dataset is hosted at
`https://anonymous.4open.science/r/madcal-calibration-E8A2/` for
double-blind review and will be mirrored to GitHub + Zenodo upon
acceptance.

The package is a **research replication artifact**, not a deployable
predictor. It contains (i) natural-language *vignette* representations
of three publicly available tabular datasets, (ii) aggregated debate
and StandardLLM outcome CSVs (10,191 case-level debate-aggregate rows
across the 100/200-case headline experiments and the released
ablations; 230 K+ StandardLLM rows across the larger model ensemble),
and (iii) reproduction scripts that derive every figure, table, and
statistic in the paper from these outcomes. Note that the **raw
multi-turn debate transcript JSON files are not redistributed**:
they exceed practical distribution size and are excluded by
`.gitignore`. The aggregated outcomes are sufficient to reproduce
every quantitative claim in the paper.

## 1. Motivation

**For what purpose was the dataset created?**
The package was created to replicate the empirical findings of the
MADCal paper, which evaluates whether *adversarial role assignment* in
multi-agent LLM debate produces a prediction-distribution calibration
effect on three high-stakes tabular prediction tasks. The replication
package exists so that reviewers and other researchers can verify each
quantitative claim (sign-test result, BRD reductions, ablation
contrasts) without re-running the LLM API calls.

**Who created the dataset and on behalf of which entity?**
Creator information is withheld for double-blind review. Authorship
will be disclosed in the camera-ready version.

**Who funded the creation of the dataset?**
Funding information is withheld for double-blind review.

## 2. Composition

**What do the instances represent?**
The package contains four classes of instances:

1. **Source vignettes**: 1,412 NLSY97 records, 30/100/200 ProPublica
   COMPAS records, and 30/100 UCI Credit Default records, each
   serialized into a natural-language description following
   Hegselmann et al. (2023). The COMPAS vignettes retain the
   `decile_score` column; the no-decile sensitivity variant
   (Appendix K) is realized by filtering at evaluation time, not by a
   separate vignette file.
2. **Top-feature selections**: per-dataset LOFO (Leave-One-Feature-Out)
   importance results identifying the top-10 features used by each
   serialization, plus the 5-feature-selection-algorithm × 5/10-feature
   ablation grid for NLSY97.
3. **Per-debate outcome rows**: each row of `results/debate_aggregate_*.csv`
   records a single MADCal debate run (model, case, repetition,
   dataset, judge prediction, judge confidence, ground truth, number
   of judge opinion updates). The released aggregates total 10,191
   rows across 10 CSVs covering the headline 100/200-case experiments
   and the cooperative / free-form ablations (Appendix I, ablations
   C7 and C8).
4. **Per-model StandardLLM outcome rows**: equivalent rows for the
   zero-shot, CoT, and 30-shot CoT controls, including 236,702 rows
   in `results/standardllm/sc_nlsy97_merged.csv` covering the 16-model
   open-source ensemble plus commercial models.

In aggregate the package documents the data behind the paper's
"123,000+ inferences spanning 19 LLMs" claim. Per-debate
**multi-turn conversation transcripts** were generated locally during
research but are excluded from the public release for size reasons;
each released aggregate row encodes the judge's final prediction +
confidence + opinion-trajectory length, which is what the paper's
quantitative analyses use.

**How many instances are there in total?**
- **Vignette CSVs (12 files)**:
  - NLSY97: 1,412 (full) / 200 / 100 / 30-case sample.
  - COMPAS: 6,172 (full filtered) / 200 / 100 / 30-case sample.
  - UCI Credit Default: 30,000 (full filtered) / 100 / 30-case sample.
- **Top-feature CSVs (10 files in `data/topn_datasets/`)**: 5
  feature-selection algorithms × 5- and 10-feature variants for NLSY97.
- **MADCal debate aggregate rows (10 CSVs, 10,191 rows total)**:
  - `debate_aggregate_compas_100.csv` (900 rows; 3 commercial × 100 × 3),
  - `debate_aggregate_compas_200.csv` (1,697 rows),
  - `debate_aggregate_nlsy97_100.csv` (900),
  - `debate_aggregate_nlsy97_200.csv` (1,690),
  - `debate_aggregate_nlsy97_oss.csv` (1,200 rows; 4 open-source models × 100 × 3),
  - `debate_aggregate_credit_default_100.csv` (800),
  - `cooperative_aggregate_{compas,nlsy97,credit_default}_100.csv` (Appendix I, C7 ablation),
  - `freeform_aggregate_nlsy97_oss.csv` (Appendix I, C8 ablation).
- **StandardLLM merged CSVs**: large per-model log files in
  `results/standardllm/` (e.g., `sc_nlsy97_merged.csv` at 236,702 rows;
  `sc_compas_merged.csv` at comparable size); these cover the broader
  16-model open-source ensemble across zero-shot / CoT / 30-shot CoT
  prompts.
- **Auxiliary result CSVs**: BRD figure data, ablation random-flip
  outputs, judge-opinion-evolution data, accuracy/F1 leaderboards,
  and the GPT-5-mini bridging table.

**Does the dataset contain all possible instances, or is it a sample?**
Sample. The 100- and 200-case COMPAS / Credit Default / NLSY97
subsamples were drawn with random seed 42 from the publicly available
source datasets to keep API costs bounded; the full NLSY97 (1,412
cases), full filtered COMPAS (6,172 cases), and full filtered Credit
Default (30,000 cases) source CSVs are included for replication of the
sub-sampling step. The released aggregate CSVs document outcomes for
seven distinct LLMs (three commercial: `gpt-4.1-mini`, `gpt-4o-mini`,
`gemini-2.5-flash`; four open-source via OpenRouter: `llama-3.1-8b`,
`qwen-2.5-7b`, `phi-4-14b`, `gemma-2-9b`); other models named in the
paper appear in the StandardLLM merged CSVs but are not separately
debate-aggregated in the public release. The exhaustive 16-model
small-ensemble debates that yield the paper's "2,400+ debate runs"
phrasing exist as raw transcripts on the authors' compute hosts but
are not redistributed; the released aggregates already contain the
per-debate outcomes needed for every paper analysis.

**What data does each instance consist of?**

For vignette CSVs:
- `case_id`: integer index of the case within the source dataset.
- `vignette`: natural-language description of the top-10 LOFO features
  for that case.
- `target`: binary outcome (rearrest within 3 years for NLSY97;
  recidivism within 2 years for COMPAS; default-next-month for
  UCI Credit Default).
- For COMPAS only, additional columns including `decile_score`
  (the COMPAS algorithmic risk score; not used by the headline LLM
  evaluations, but retained for the App. K leakage-sensitivity analysis).

For debate / StandardLLM aggregate CSVs:
- `model_name` (provider/model identifier).
- `case_id` (matches the vignette case_id).
- `repeat_id` (1..N for repetitions of the same case).
- `dataset` (one of `nlsy97`, `compas`, `credit_default`).
- `prediction` (judge's final YES/NO).
- `confidence` (judge's reported 0–100 confidence).
- `ground_truth` (binary outcome).
- `n_judge_opinions` (count of judge belief updates; 6 for a complete
  MADCal debate, lower for runs that errored out).

**Are there labels or target values associated with each instance?**
Yes. Each vignette has the binary recidivism / default outcome from
the source dataset; each debate / StandardLLM row records the model's
predicted YES/NO and the corresponding ground truth.

**Is any information missing from individual instances?**
Cases where the judge agent failed to produce parseable JSON are
recorded as "No Decision" and excluded from headline metrics; these
appear in the aggregate CSVs with `prediction = NoDecision`.

**Are relationships between individual instances made explicit?**
Yes — within a (model, case, dataset) tuple, multiple `repeat_id`
rows correspond to repeated runs at the same temperature. Across
models, the `case_id` is shared so a researcher can compare model-pair
predictions on the same case.

**Are there recommended data splits?**
The paper uses a 60/20/20 train/val/test split for NLSY97 (test set
held out) and 100-case random subsamples for COMPAS and Credit
Default (seed 42). The released vignette CSVs are pre-split.

**Are there any errors, sources of noise, or redundancies?**
- Marco-o1 exhibits a degenerate behavior at $T = 0$ (App. I,
  Ablation C3): repetitions converge to identical outputs.
- Gemini-2.5-Flash returned malformed JSON in approximately 96% of
  single-turn calls; effective $K \approx 2$–$3$ for the
  $K = 59$ self-consistency comparison (App. L). Affected runs are
  flagged in `results/standardllm/sc_*_majority_vote.csv`.
- "No Decision" rows are present at non-trivial rates for some open-source
  models; per-model rates are reported in App. O.
- Open-source models are 4-bit quantized via Ollama; quantization
  noise is not separately characterized.

**Is the dataset self-contained, or does it link to external resources?**
The vignette CSVs are derived from publicly available source datasets:
- NLSY97 (Bureau of Labor Statistics, public-use file).
- ProPublica COMPAS recidivism release (public CSV release accompanying
  Angwin et al., 2016).
- UCI Credit Default dataset (Yeh and Lien, 2009).
The release does not contain raw NLSY97 / COMPAS / Credit Default
records that fall outside the 100/200/1,412-case subsamples used in
the paper.

**Does the dataset contain data that might be considered confidential or
contain personally identifiable information?**
No. All three source datasets are publicly available and de-identified
at the source; the released vignettes use the same de-identified
representations.

**Does the dataset contain data that might be considered sensitive in
nature?**
The source datasets contain demographic and criminal-justice features
(prior arrests, race, age) that are sensitive in operational use. The
release explicitly carries a non-deployment declaration (\S6 of the
paper) and is licensed for research use only.

## 3. Collection Process

**How was the data acquired?**
Source records were downloaded from the public NLSY97 / ProPublica /
UCI release pages between January 2026 and February 2026. LLM outputs
were generated by:
- Calling commercial provider APIs (OpenAI, Google, xAI, Anthropic,
  OpenRouter, Fireworks) between 2026-01 and 2026-02.
- Running 4-bit-quantized open-source models locally via Ollama 0.5.7
  on a Ryzen 9 / dual NVIDIA 3090 / CUDA 12.6 / Python 3.10 host.
All generated rows were logged together with API metadata
(timestamps, token counts, model version strings).

**Over what timeframe was the data collected?**
January 2026 through early March 2026.

**Was any ethical review process conducted?**
No human subjects were involved; LLMs only. IRB approval is therefore
not applicable.

**Did the individuals consent to the collection of their data?**
Not applicable: the source NLSY97, COMPAS, and UCI Credit Default
datasets are public-release files for which consent was obtained by
the original collectors. No new data was collected from human subjects.

## 4. Preprocessing / Cleaning / Labeling

**Was any preprocessing/cleaning/labeling of the data done?**
Yes:
- **Feature selection** via Leave-One-Feature-Out (LOFO) importance to
  retain the top 10 features per dataset.
- **Vignette serialization** following Hegselmann et al. (2023):
  each feature is rendered as `<feature> is <value>` and concatenated
  into a single natural-language description.
- **Output parsing** of LLM responses uses strict JSON parsing with a
  regex fallback; unrecoverable outputs are recorded as
  `prediction = NoDecision`.
- **Aggregation** rolls per-debate / per-prompt outputs into the
  aggregate CSVs in `results/`.

**Was the "raw" data saved in addition to the preprocessed data?**
The raw source CSVs (`*_full.csv` / `*_full_filtered.csv`) and the
derived vignette CSVs are both included.

**Is the software used to preprocess / clean / label the data
available?**
Yes — under `scripts/` and `src/`, all open-source.

## 5. Uses

**Has the dataset been used for any tasks already?**
Yes: it produces every figure, table, and statistic in the
NeurIPS 2026 MADCal paper, including the headline calibration claims,
the 12-pair sign test, the disparate-impact analysis, and the six
ablations enumerated in Appendix I.

**Is there a repository that links to any or all papers or systems
that use the dataset?**
The anonymous mirror serves as the repository during review; a
GitHub + Zenodo home will follow at acceptance.

**What other tasks could the dataset be used for?**
- Methodological work on multi-agent LLM debate orchestration,
  prompt design, or judge-confidence calibration.
- Comparative studies of test-time compute strategies (CoT,
  self-consistency, structured debate) on tabular tasks.
- Replication or extension of the disparate-impact analysis (App. K).

**Is there anything about the composition or collection / preprocessing
pipeline that might impact future uses?**
- Commercial-API model versions change continually; reproductions on
  newer provider versions will not match the released numbers
  byte-for-byte but should reproduce the directional findings.
- The 100-case COMPAS / Credit Default subsamples were drawn for cost
  containment and may not represent the full distributional
  characteristics of the source datasets.

**Are there tasks for which the dataset should not be used?**
- **Operational deployment**: the dataset is not a recidivism predictor,
  not a credit risk score, and must not be used as such. The paper's
  Broader Impact section makes this declaration explicit.
- **Demographic-attribute prediction**: the package contains demographic
  features in the source vignettes; downstream models trained to predict
  these attributes are out of scope and discouraged.
- **Training a calibration-bypass model**: the explicit goal is to
  evaluate, not to defeat, calibration mechanisms.

## 6. Distribution

**How will the dataset be distributed?**
- During review: anonymous mirror at
  `https://anonymous.4open.science/r/madcal-calibration-E8A2/`.
- At acceptance: GitHub + Zenodo (DOI to be assigned).

**When will the dataset be distributed?**
Immediately upon publication of the camera-ready paper.

**Will the dataset be distributed under a copyright or other
intellectual property (IP) license?**
- Code: MIT License (see `LICENSE`).
- Derived datasets (vignettes, aggregated outcomes): CC-BY-4.0
  (`https://creativecommons.org/licenses/by/4.0/`).
- Source datasets retain their original public-use licenses
  (NLSY97 BLS public-use terms; ProPublica's CC-BY-NC-SA for COMPAS;
  UCI's standard release terms for Credit Default).

**Have any third parties imposed IP-based or other restrictions on
the data associated with the instances?**
- **NLSY97**: BLS public-use file terms permit research and educational
  use.
- **COMPAS**: ProPublica's release is CC-BY-NC-SA (non-commercial,
  share-alike). Derived analyses inherit the share-alike requirement
  to the extent that the original COMPAS records are reproduced.
- **UCI Credit Default**: standard UCI ML Repository release terms
  (research use).

**Do any export controls or other regulatory restrictions apply?**
None known.

## 7. Maintenance

**Who will be supporting / hosting / maintaining the dataset?**
For the review period, the dataset is hosted via
`anonymous.4open.science`. Post-acceptance maintenance will be
performed by the (de-anonymized) author team via GitHub + Zenodo.

**How can the owner / curator / manager of the dataset be contacted?**
During review, via the OpenReview submission page (NeurIPS 2026 main
track). Post-acceptance, via the GitHub repository's issues tracker.

**Is there an erratum?**
None at submission; if errata are required they will be added to a
top-level `ERRATA.md` and linked from the README.

**Will the dataset be updated?**
Yes. Versioning policy: a new release tag for each substantive update;
updates that change quantitative results in the paper will be
documented in a `CHANGELOG.md`. The version corresponding to the
NeurIPS 2026 submission is `v1.0`.

**If others want to extend / augment / build on / contribute to the
dataset, is there a mechanism for them to do so?**
Yes. After acceptance, contributions will be accepted via GitHub pull
request; we anticipate accepting additional model evaluations,
additional ablations, and extensions to other tabular datasets
without changing the core protocol.
