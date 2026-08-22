# Data-Driven Decision Intelligence with Uncertainty-Calibrated AI

## Proposed short communication title

**Data-Driven Decision Intelligence with Uncertainty-Calibrated AI: Learning When to Predict and When to Defer**

## Core research question

Can probability calibration improve an AI decision system's ability to determine
when a prediction is sufficiently reliable for automation and when a case should
be deferred for human review?

## Experiment

The project compares:

- Random Forest
- XGBoost
- Sigmoid-calibrated XGBoost

The main evaluation covers:

- Accuracy
- Balanced accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Brier score
- Log loss
- Expected Calibration Error (ECE)
- Coverage
- Deferral rate
- Selective accuracy
- Selective error
- Decision utility

## Dataset

UCI Bank Marketing dataset.

The target is whether a customer subscribes to a term deposit.

The `duration` feature is removed because it is not fully known before the
outcome of the call and could cause unrealistic leakage for pre-decision use.

## Predict-or-defer policy

For each observation:

```text
confidence = max(P(class=0), P(class=1))

if confidence >= threshold:
    PREDICT
else:
    DEFER TO HUMAN REVIEW
```

## Run on RunPod

### 1. Unzip and enter the repository

```bash
unzip data-driven-decision-intelligence.zip
cd data-driven-decision-intelligence
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download the dataset

```bash
python src/download_data.py
```

### 5. Run the experiment

```bash
python src/experiment.py
```

## Outputs

The experiment produces:

### Results

- `results/dataset_summary.csv`
- `results/model_metrics.csv`
- `results/calibration_metrics.csv`
- `results/selective_prediction.csv`
- `results/threshold_analysis.csv`
- `results/best_decision_policy.csv`

### Figures

- `figures/calibration_curve.png`
- `figures/coverage_risk_curve.png`
- `figures/coverage_accuracy_curve.png`
- `figures/threshold_utility.png`
- `figures/confusion_matrix.png`

## Main interpretation

Do not focus only on classification accuracy.

The intended research contribution is the comparison between:

1. uncalibrated AI confidence,
2. calibrated AI confidence,
3. automatic prediction coverage,
4. deferred cases,
5. error among automatically handled cases, and
6. decision utility.

The key question is whether calibration enables a better operating point between
automation and deferral.

## Important scientific wording

The uncertainty in this project is operational uncertainty represented through
calibrated predictive probabilities.

Do not claim that this experiment fully estimates epistemic uncertainty.

A defensible description is:

> The framework uses calibrated predictive probabilities to quantify operational
> uncertainty and determine whether an observation should be automatically
> classified or deferred for additional review.

## Reproducibility

Random seed: `42`
