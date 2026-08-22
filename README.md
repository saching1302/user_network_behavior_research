# Data-Driven Decision Intelligence with Uncertainty-Calibrated AI

A public research project exploring whether probability calibration can improve a decision system that is allowed to predict or defer to human review.

## Project goal

The project tests whether calibrated predictive uncertainty helps separate:

- cases that are reliable for automated prediction,
- cases that should be deferred for human review,
- and the trade-off between accuracy, coverage, and decision utility.

## Research question

Can probability calibration improve an AI decision system's ability to determine when a prediction is sufficiently reliable for automation and when a case should be deferred for human review?

## Methods compared

- Random Forest
- XGBoost
- Sigmoid-calibrated XGBoost

## Evaluation metrics

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

This project uses the UCI Bank Marketing dataset.

The target is whether a customer subscribes to a term deposit. The `duration` feature is intentionally removed because it is not fully known before the outcome of the call and could create unrealistic leakage in a pre-decision setting.

## Predict-or-defer policy

For each observation:

```text
confidence = max(P(class=0), P(class=1))

if confidence >= threshold:
    PREDICT
else:
    DEFER TO HUMAN REVIEW
```

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── run_all.bat
├── run_all.sh
├── data/
├── results/
├── figures/
├── src/
└── LICENSE
```

## Quick start

### 1. Clone or download the repository

```bash
git clone https://github.com/saching1302/user_network_behavior_research.git
cd user_network_behavior_research
```

### 2. Create a virtual environment

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

Or on Windows, simply run:

```bat
run_all.bat
```

## Generated outputs

### Results

- results/dataset_summary.csv
- results/model_metrics.csv
- results/calibration_metrics.csv
- results/selective_prediction.csv
- results/threshold_analysis.csv
- results/best_decision_policy.csv
- results/experiment_output.md

### Figures

- figures/calibration_curve.png
- figures/coverage_risk_curve.png
- figures/coverage_accuracy_curve.png
- figures/threshold_utility.png
- figures/confusion_matrix.png

## Main interpretation

The key contribution is not just classification accuracy, but the comparison between:

1. uncalibrated confidence,
2. calibrated confidence,
3. prediction coverage,
4. deferred cases,
5. error among automatically handled cases, and
6. decision utility.

This framing supports a practical operational view of uncertainty: calibrated predictive probabilities can be used to decide when it is appropriate to automate a decision and when a case should be escalated for human review.

## Reproducibility

Random seed: 42

## License

This project is licensed under the MIT License.
