# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

**Run the main classification pipeline (interactive menu):**
```bash
cd Code
python text_llms.py
```

**Run metrics analysis after classification:**
```bash
# From LLMS_analysis/ root (required — metrics_analysis.py imports utils via relative path)
.venv\Scripts\python.exe Code\metrics_analysis.py
```

**Run human IRR analysis:**
```bash
cd IRR_analysis
python human_irr.py
python plot_irr.py
```

**Install dependencies:**
```bash
pip install -r Code_for_all/requirements.txt
```

## Environment Setup

Place a `.env` file in `Code/` (and `Code_for_all/` if using that variant) with:
```
OAI_2=<openai-key>
GEMINI=<google-key>
CLAUDE=<anthropic-key>
```

`utils.py` loads these via `python-dotenv`. All path constants (`DATA_PATH`, `RESULTS_PATH`, `PROMPTS_PATH`) are resolved relative to `utils.py`'s own location, so the working directory must be set correctly when importing.

## Architecture

### Two parallel code trees

- **`Code/`** — active development version (3 files: `text_llms.py`, `utils.py`, `metrics_analysis.py`)
- **`Code_for_all/`** — alternate/experimental version with the same filenames plus `v.py`

Both share the same `Data/`, `Results/`, `prompts/`, and `Graphs/` directories via `utils.py` path constants.

### Core files

**`Code/text_llms.py`** (~900 lines) — Interactive menu-driven pipeline. Entry point for all classification runs. Supports:
- Three LLMs: OpenAI (ChatGPT), Google (Gemini), Anthropic (Claude)
- Four prompting strategies: Zero-Shot, Few-Shot, Zero-Shot CoT, Few-Shot CoT
- Three processing modes: line-by-line, group, Claude Batch API
- Checkpoint-based resumable execution (tracks processed `row_id`s in output CSV)
- Claude Batch API polling loop with `batch_status.json` tracking

**`Code/utils.py`** — Path constants and API key loading. All other modules import `from utils import *`. The `.env` file must be in the same directory as `utils.py`.

**`Code/metrics_analysis.py`** — Post-classification evaluation. Key functions:
- `krippendorff_alpha_nominal(y_true, y_pred)` — 2-rater alpha via `krippendorff` library; builds `(2, n_items)` reliability matrix
- `krippendorff_alpha_4raters(tags_df, merged_df, obs_keys, tag, y_pred_binary)` — 4-rater alpha (3 reconstructed human coders + 1 LLM) for Paper 1
- `_reconstruct_coders(avg_series)` — reconstructs 3 binary coder vectors from averaged fractions; assigns 1s to the last `k` coders (index-descending)
- `paper_evaluation()` — dispatches to paper-specific evaluation logic
- `get_results_and_visualize()` — scans `Results/` for CSV files and produces metric tables + PNG summaries

### Data layout

```
Data/{paper_name}/
    classify.csv        # input texts to classify
    real_answers.csv    # ground truth labels
    tags.csv            # Paper 1 only: raw coder fractions per conversation
```

Four papers: `managerial_leadership_Jordi_Cooper`, `strategic_environment_Ozkes_Hanaki`, `trust_promises_Ederer_Schneider`, `under_reporting_Ling_Kale_Imas`.

### Results layout

```
Results/{llm}/{paper_name}/{strategy}/
    results_line_temp{T}_mode{mode}.csv
    results_line_batch_temp{T}_mode{mode}.csv   # Claude batch API only
    results_group_temp{T}_mode{mode}.csv
    results_paper_{id}_temp{T}_mode{mode}_type{strategy}.csv   # metrics output
    results_paper_{id}_temp{T}_mode{mode}_type{strategy}.png   # visualization
```

### Prompts layout

```
prompts/{paper_name}/
    role.txt, context.txt, classificationTask.txt, format.txt, constraints.txt
    fewShot.txt, 0ShotCoT.txt, few-shotCoT.txt
```

### Krippendorff alpha usage

The `krippendorff` library expects a reliability data matrix of shape `(n_raters, n_items)` with `np.nan` for missing values. Both alpha functions build this matrix explicitly before calling `krippendorff.alpha(..., level_of_measurement="nominal")`. Do not compute expected disagreement manually — always delegate to the library.

### Paper 1 specifics (managerial_leadership)

Paper 1 uses `tags.csv` with averaged coder fractions instead of individual coder labels. `_reconstruct_coders()` expands these back into 3 binary rater vectors. The 4-rater alpha merges reconstructed human ratings with LLM predictions on `OBS_KEYS_P1 = ["session", "period", "group", "game"]`.

### Claude Batch API flow

1. Requests submitted → `batch_status.json` records `{paper: {strategy: {temp: batch_id}}}`
2. Polling loop checks status every 30 seconds
3. Results streamed and written to `results_line_batch_temp{T}_mode{mode}.csv`
4. Valid Claude temperatures: `0, 0.1, 0.5, 1` — `1.2` is auto-skipped
