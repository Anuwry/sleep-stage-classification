# Sleep Stage Classification

Multichannel sleep-stage classification with a PyTorch CNN-BiLSTM-attention model.

## Project context

Super AI Engineering Season 6, 2026. Individual competition project.

## Approach

A 1D CNN, bidirectional LSTM and learned attention process multichannel wearable-sensor windows. Includes weighted loss, augmentation, early stopping and prediction-file generation.

## Competition result

Competition leaderboard **weighted F1: 0.43160**.

Leaderboard results are from the project author's competition records, not newly generated benchmark results.

## Source files

- notebooks/cnn_bilstm_attention.ipynb

## Setup and use

1. Create a Python environment compatible with the historical notebook and selected model.
2. Install relevant dependencies from requirements.txt. GPU frameworks and Unsloth may require platform-specific CUDA instructions.
3. Provide authorized competition sensor arrays and labels. Configure input paths in the notebook.
4. Start JupyterLab and run the selected notebook in order. Review experiment branches before a full training run.

    python -m venv .venv
    python -m pip install -r requirements.txt
    python -m jupyterlab

## Archive preparation

- Notebook outputs and execution metadata were cleared.
- Hardcoded credentials found in selected source were replaced with environment lookups.
- Datasets, personal records, submissions, model weights and local environments are excluded.
- Source was statically checked; training and external API calls were not rerun.

## Limitations

Historical validation uses random segments and fits the scaler before that internal split. It is not subject-held-out or leakage-free evaluation. For new experiments, fit preprocessing on training data only and use subject/session-aware splits where identifiers exist. The competition score is separate from internal validation logs.
