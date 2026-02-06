# Anomaly Dataset Benchmark — Quick Start

This repository provides a unified benchmarking framework for anomaly detection on cloud telemetry datasets.
This README documents the workflow for **a reconstruction-error–based training run**, followed by **optional likelihood tuning** as a post-processing step.

---

## 0) Environment setup

### Execution environment

All preprocessing, training, likelihood calibration, and evaluation were executed and validated across two environments—a Windows 10 desktop and a Linux server running a Docker-based setup (tf-docker)—to ensure reproducibility and consistency.

Windows 10 Desktop:
- **OS**: Windows 10
- **Python**: 3.12.7

Linux server:
- **OS**: Ubuntu 20.04.5 LTS
- **Python**: 3.8.10



### Create and activate a virtual environment from the repository root:

**Windows**

```bat
python -m venv .venv
. .venv\Scripts\activate
python --version
```

#### Install dependencies

With Windows-tested dependency set:

```bat
pip install -r requirements_winsows_desktop.txt
```

Verify key packages:

```bat
pip list | findstr "tensorflow numpy pandas scikit-learn scipy hydra-core"
```


**macOS/Linux**
```bat
python -m venv .venv
source .venv/bin/activate
```
    pip install -r requirements.txt


#### Install dependencies

With Linux server-tested dependency set:

```bat
pip install -r requirements.txt
```




Verify key packages:

```bat
pip list | findstr "tensorflow numpy pandas scikit-learn scipy hydra-core"
```

---

## 1) Folder expectations

Before running, ensure raw datasets are available locally and paths are correctly set in `config/prep.yaml`.

Typical expectations:

- Raw NAB CSVs:
  `config/prep.yaml -> nab.data_dir`
  (e.g., `data/NAB/data/...`)
- NAB labels JSON:
  `config/prep.yaml -> nab.label_file`
  (e.g., `data/NAB/labels/labels.json`)
- Raw Microsoft CSVs:
  `config/prep.yaml -> microsoft.data_dir`
  (e.g., `data/microsoft/...`)

Prepared data and dataset-level plots are written automatically under:

```
output/
├── prepared/
└── plots/
```

---

## 2) Execution pipeline

Running the training command triggers a **two-stage pipeline**.

### Stage 1 — Preparation

- Raw data are read from `data/`

```
data/
├── nab/
└── microsoft/
├── exathlon/
└── ibm/
```

- Dataset-specific preprocessing is applied
- Prepared CSVs and dataset-level plots are generated

Outputs:

```
output/prepared/<dataset>/
output/plots/<dataset>/
```

This stage may overwrite existing prepared files if enabled in `config/config.yaml`.

### Stage 2 — Training and evaluation

- Models are trained using prepared data
- Reconstruction error is computed
- Evaluation artifacts are generated

All model-related outputs are written under:

```
runs/
```
  (e.g., `runs/nab/2026-02-06_10-38-06__TF_GRU_AE__seq8_bs32_minmax__gru_nab/...`)

Preparation is invoked automatically based on configuration in `config/config.yaml`. Users do not need to run a separate preprocessing command.

---

## 3) Configuration 

The main experiment is controlled by `config/config.yaml`.

Key characteristics:

- Unified model width and depth across architectures
- Fixed 70/30 train–test split
- Reconstruction-error–based detection


Example (key fields):

```yaml
dataset:
  name: nab
  prepared_dir: output/prepared/${dataset.name}

steps:
  prep:
    enabled: true
    overwrite: false

output:
  dir: runs
  tag: ${dataset.name}_tf

split:
  train_ratio: 0.7
  seq_len: 8
  step: 1

scaler:
  kind: minmax

model:
  framework: tf
  name: gru_ae
  hidden: 32
  layers: 4
  batch_size: 32
  epochs: 30
  val_ratio: 0.1
```


---

## 4) Run preparation + training

Experiments were primarily executed using the provided wrapper script, which ensures consistent logging and background execution.

### Recommended (wrapper script)

From the repository root (Git Bash on Windows or Linux shell):

./run_bg.sh --mode train --config config/config.yaml --tag gru_nab

This command triggers a two-stage pipeline:

- prepares the dataset from raw files (if steps.prep.enabled: true)
- trains the selected model
- computes reconstruction error
- writes prepared artifacts and plots to `output/`
- writes all model outputs and logs to `runs/`

### Alternative (direct Python execution)

If you prefer to run without the wrapper script:

set PYTHONPATH=%CD%;%PYTHONPATH%
python -m src.main --config config\config.yaml --tag prep_train

This executes the same pipeline and produces identical artifacts under `output/` and `runs/`.

---



## 5) Switching models

Change the model by editing a single field in `config/config.yaml`:

```yaml
model:
  name: gru_ae
```

Supported models:

- `gru_ae`
- `tcn`
- `transformer`
- `tsmixer`
- `isolation_forest`

Isolation Forest serves as a non-neural baseline and follows the same training and evaluation flow.

---

## 6) Outputs and logs

All run artifacts are placed under:

```
runs/<DATASET>/<TIMESTAMP>__<MODEL>__seq<SEQ>_bs<BATCH>_<SCALER>[__<TAG>]/
```

Common contents (example):

- `checkpoints/`
- `series/`
- `_debug.log`
- `training_log.csv`

Quick sanity checks:

```bat
dir runs\nab
```

---

## 7) Likelihood tuning (post-processing)

Likelihood calibration parameters (short window, long window, threshold) are tuned **after model training**, using the per-series artifacts produced under `runs/`.

Directory for likelihood tuning in the project root:

```
likelihood_tuning/
```

This tuning workflow:
- reads `../runs/<dataset>/<run_id>/series/`
- constructs subgroups under `working_data/subgroups/<dataset>/`

Example usage:

```bat
cd likelihood_tuning
python prep_subgroups.py --series_dir "..\runs\nab\2026-02-06_10-38-06__TF_GRU_AE__seq8_bs32_minmax__gru_nab\series"
```

- runs W&B sweeps to select likelihood parameters on the training split only


Example sweep commands:

```bat
wandb sweep "$(pwd -W)/config/sweep_likelihood_tune.yaml"
wandb agent amirlab/hyper_tune_aws/<sweep_id>
```

**Example of `config/sweep_likelihood_tune.yaml`**
```yaml
program: "lik_sweep_runner.py" 
method: bayes 
project: hyper_tune_aws
entity: amirlab


run_cap: 100 #100        # <---- THIS controls number of runs; 

metric:
  name: dataset_raw_sum
  goal: maximize

parameters:
  # ----- Fixed context -----
  series_dir:
    value: "./working_data/subgroups/nab/artificialWithAnomaly"
  profile:
    value: standard
  split_mode:         
    value: train      # options: all, train, test

  likelihood.short_window:
    distribution: int_uniform
    min: 3
    max: 30
  likelihood.long_window:
    distribution: int_uniform
    min: 70
    max: 500
  likelihood.threshold:
    distribution: uniform
    min: 0.9 
    max: 0.9999
```

Once tuning completed on traing split, tuned parameters are used to test the test split.

**Example of `config/sweep_likelihood_test.yaml`**
```yaml
program: "lik_sweep_runner.py" 
method: grid
project: hyper_tune_aws
entity: amirlab


run_cap: 1         # <---- THIS controls number of runs

metric:
  name: dataset_raw_sum
  goal: maximize

parameters:
  series_dir:
    value: "./working_data/subgroups/nab/artificialWithAnomaly"
  profile:
    value: standard
  split_mode:
    value: test

  likelihood.long_window:
    value: 236
  likelihood.short_window:
    value: 28
  likelihood.threshold:
    value: 0.9950772703649864
```
---

## 8) Troubleshooting

- **Configs not found**
  Run from repo root and ensure `config\` exists.
- **Prepared data missing**
  Verify raw paths in `config/prep.yaml`.
- **Unexpected number of trained series**
  Check `output/prepared/<dataset>/manifest.json`.
- **Transformer errors**
  Ensure `model.hidden % transformer.heads == 0`.

---

