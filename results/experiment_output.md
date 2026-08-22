# Experiment Output Summary

## Run status
The full project pipeline completed successfully.

## Best decision policy
- Method: Calibrated XGBoost
- Confidence threshold: 0.55
- Mean utility: 0.7100327749453751
- Coverage: 0.9617625637290604
- Deferral rate: 0.03823743627093955
- Selective accuracy: 0.9160671462829736
- Selective error: 0.08393285371702641

## Model metrics
- Random Forest
  - Accuracy: 0.8634377276037873
  - Balanced accuracy: 0.7640575616774377
  - Precision: 0.42846768336964414
  - Recall: 0.6357758620689655
  - F1: 0.5119305856832972
  - ROC-AUC: 0.8134467575121468
  - Brier score: 0.13963107855046913
  - Log loss: 0.45754048072730524
  - ECE: 0.2429496077230852

- XGBoost
  - Accuracy: 0.8485069191551348
  - Balanced accuracy: 0.7608187591395821
  - Precision: 0.39487516425755587
  - Recall: 0.6476293103448276
  - F1: 0.4906122448979592
  - ROC-AUC: 0.8107616367517336
  - Brier score: 0.1432216316461563
  - Log loss: 0.465639591217041
  - ECE: 0.24267886314630638

- XGBoost + Sigmoid Calibration
  - Accuracy: 0.9021607186210245
  - Balanced accuracy: 0.6113609427331478
  - Precision: 0.6930379746835443
  - Recall: 0.23599137931034483
  - F1: 0.3520900321543408
  - ROC-AUC: 0.8125547637860276
  - Brier score: 0.07489818904433787
  - Log loss: 0.2652038296751649
  - ECE: 0.014786273474615697

## Output files generated
- best_decision_policy.csv
- calibration_metrics.csv
- dataset_summary.csv
- model_metrics.csv
- selective_prediction.csv
- threshold_analysis.csv
- calibration_curve.png
- confusion_matrix.png
- coverage_accuracy_curve.png
- coverage_risk_curve.png
- threshold_utility.png

## Terminal result
The script reported: "Experiment complete."
