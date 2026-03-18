# Synthetic-Prompt-Noise-in-a-3B-Chat-Model

## LLM Robustness Training and Evaluation

This repository contains three notebooks for training and evaluating QLoRA adapters under prompt corruption. The project studies how training on noisy instructions affects instruction-following robustness, JSON-format compliance, and task-grounded content quality.

The workflow has three stages:

1. Train adapters with different noise ratios.
2. Evaluate those adapters on a shared clean/noisy aggregate test set.
3. Evaluate the same adapters on a categorical corruption test set broken out by noise type.

All notebooks are written to run locally and save outputs to local directories.

## Repository contents

### 1. `Robustness_Training_local_submission.ipynb`
Trains separate QLoRA adapters on the Alpaca dataset using different levels of prompt corruption. The notebook:

- loads and cleans the source dataset,
- builds deterministic train/validation/test splits,
- generates noisy prompt variants,
- trains one adapter per noise condition,
- saves trained adapters and split files locally,
- writes training metadata for later analysis.

Main outputs:

- adapter folders under `trained_adapters/`,
- train/validation/test CSV splits under `data_splits/`,
- `training_variants_metadata.csv`.

### 2. `Robustness_Evaluation_local_submission.ipynb`
Runs aggregate evaluation on the trained adapters using a shared evaluation set with clean and noisy prompts. It measures both formatting compliance and content quality.

Main outputs:

- `strict_eval_set_tasknoise_answer_schema.csv`
- `all_adapters_raw_eval_taskgrounded_batch32.csv`
- `all_adapters_summary_with_ci_taskgrounded_batch32.csv`
- `all_adapters_summary_valid_json_only_taskgrounded_batch32.csv`
- `all_adapters_delta_vs_baseline_taskgrounded_batch32.csv`
- `all_adapters_summary_with_validation_loss_taskgrounded_batch32.csv`
- plots saved under `figures/`

### 3. `Robustness_Evaluation_Categorical_local_submission.ipynb`
Runs categorical evaluation where corruption type is separated into distinct conditions. This makes it easier to see which adapters are most robust to each corruption pattern.

The categorical conditions are:

- typos,
- punctuation removal,
- shorthand substitutions,
- word drop.

Main outputs:

- `categorical_eval_set_taskgrounded_answer_schema.csv`
- `categorical_noise_eval_taskgrounded_batch32.csv`
- `categorical_noise_summary_with_ci_taskgrounded_batch32.csv`
- `categorical_noise_summary_valid_json_only_taskgrounded_batch32.csv`
- `categorical_noise_delta_vs_mixed0_taskgrounded_batch32.csv`
- `categorical_noise_summary_with_validation_loss_taskgrounded_batch32.csv`
- heatmaps and other figures saved under `figures/`

## Recommended run order

Run the notebooks in this order:

1. `Robustness_Training_local_submission.ipynb`
2. `Robustness_Evaluation_local_submission.ipynb`
3. `Robustness_Evaluation_Categorical_local_submission.ipynb`

The evaluation notebooks assume the adapters already exist locally.

## Hardware requirements

These notebooks are designed for a local NVIDIA CUDA environment.

Important constraints:

- the model is loaded in 4-bit mode,
- training and evaluation use `bitsandbytes`,
- CPU-only execution is not supported,
- Apple Silicon is not the intended runtime target.

## Model and access requirements

The default base model is:

- `meta-llama/Llama-3.2-3B-Instruct`

You need:

- a Hugging Face account with access to the model,
- a Hugging Face token exported as an environment variable.

If judge-based metrics are enabled in the evaluation notebooks, you also need:

- an OpenAI API key.

## Environment setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
pip install --upgrade pip
pip install transformers datasets peft bitsandbytes trl accelerate pandas numpy scikit-learn nltk matplotlib jupyter ipykernel bert-score evaluate rouge-score openai
```

Register the environment as a notebook kernel if needed:

```bash
python -m ipykernel install --user --name robustness-env
```

## Environment variables

Set these before opening Jupyter:

```bash
export HF_TOKEN="your_huggingface_token"
export ROBUSTNESS_PROJECT_DIR="$HOME/llama_robustness_project"
export ADAPTER_ROOT="$ROBUSTNESS_PROJECT_DIR/trained_adapters"
export HF_HOME="$ROBUSTNESS_PROJECT_DIR/.cache/huggingface"
export MODEL_ID="meta-llama/Llama-3.2-3B-Instruct"
```

Optional variables for evaluation notebooks:

```bash
export OPENAI_API_KEY="your_openai_key"
export ENABLE_LLM_JUDGE="1"
export JUDGE_MODEL="gpt-4o-mini"
```

Notes:

- `HF_TOKEN` is required if the base model is gated and not already cached.
- `OPENAI_API_KEY` is only needed when judge-based metrics are enabled.
- Setting `ENABLE_LLM_JUDGE=0` disables LLM-judge scoring in the evaluation notebooks.

## Expected directory layout

A typical local directory layout looks like this:

```text
$ROBUSTNESS_PROJECT_DIR/
├── data_splits/
├── trained_adapters/
│   ├── llama3-3b-adapter-noise-0/
│   ├── llama3-3b-adapter-noise-25/
│   ├── llama3-3b-adapter-noise-50/
│   └── llama3-3b-adapter-noise-75/
├── figures/
├── training_variants_metadata.csv
└── ... evaluation CSV outputs ...
```

If your adapters are stored somewhere else, set `ADAPTER_ROOT` to that directory.

## How to run

Start Jupyter:

```bash
jupyter lab
```

Then:

1. Open the notebook you want to run.
2. Select the correct kernel.
3. Run the cells from top to bottom.

The notebooks are written so that generated files are saved locally under `ROBUSTNESS_PROJECT_DIR`.

## Evaluation design summary

The aggregate evaluation notebook compares clean and noisy prompting on a shared test set. It tracks formatting behavior and content quality separately.

The categorical evaluation notebook keeps the task fixed and isolates the corruption type so that robustness can be compared by error category.

Across the evaluation notebooks, the saved outputs include:

- row-level model predictions,
- summary tables with confidence intervals,
- valid-JSON-only summaries,
- delta tables relative to a baseline adapter,
- saved figures for analysis and reporting.

## Reproducibility notes

The notebooks use fixed seeds and deterministic data splits where possible. Output files are written to disk so that results can be inspected without rerunning every stage.

Before running a notebook, verify that:

- the environment variables are set in the same shell session used to launch Jupyter,
- the required adapters are present,
- the GPU is visible to PyTorch,
- the Hugging Face token and optional OpenAI key are available.

## Troubleshooting

### Model access fails
Make sure `HF_TOKEN` is set and that your Hugging Face account has access to `meta-llama/Llama-3.2-3B-Instruct`.

### CUDA or 4-bit loading fails
Check that you are running on a machine with an NVIDIA GPU, working CUDA drivers, and a compatible `bitsandbytes` installation.

### Judge-based metrics fail
Set `OPENAI_API_KEY`, or disable judge scoring with:

```bash
export ENABLE_LLM_JUDGE="0"
```

### Adapters are not found
Check the value of `ADAPTER_ROOT` and verify that the expected adapter directories exist.

