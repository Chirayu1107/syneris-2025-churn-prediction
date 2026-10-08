import os
import pandas as pd
import numpy as np

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report
)


# ============================================================
# CONFIGURATION
# ============================================================

FEATURE_DATA_PATH = (
    "data/processed/combined_purchase_cart_search_features.parquet"
)

PREDICTION_PATH = (
    "data/processed/customer_churn_predictions.parquet"
)

OUTPUT_DIR = "data/processed/evaluation"

VALIDATION_CSV = os.path.join(
    OUTPUT_DIR,
    "batch_prediction_validation.csv"
)

VALIDATION_REPORT = os.path.join(
    OUTPUT_DIR,
    "batch_prediction_validation.txt"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("LOADING DATA")
print("=" * 80)

features = pd.read_parquet(FEATURE_DATA_PATH)
predictions = pd.read_parquet(PREDICTION_PATH)

print("Feature dataset shape:", features.shape)
print("Prediction dataset shape:", predictions.shape)


# ============================================================
# BASIC VALIDATION
# ============================================================

print()
print("=" * 80)
print("VALIDATING DATA ALIGNMENT")
print("=" * 80)

if "client_id" not in features.columns:
    raise ValueError("client_id missing from feature dataset")

if "churn" not in features.columns:
    raise ValueError("churn column missing from feature dataset")

if "client_id" not in predictions.columns:
    raise ValueError("client_id missing from prediction dataset")

if "churn_prediction" not in predictions.columns:
    raise ValueError("churn_prediction missing from prediction dataset")

if "churn_probability" not in predictions.columns:
    raise ValueError("churn_probability missing from prediction dataset")


feature_ids = set(features["client_id"])
prediction_ids = set(predictions["client_id"])

missing_predictions = feature_ids - prediction_ids
extra_predictions = prediction_ids - feature_ids

print("Feature customers:", len(feature_ids))
print("Prediction customers:", len(prediction_ids))
print("Missing predictions:", len(missing_predictions))
print("Extra predictions:", len(extra_predictions))


if len(missing_predictions) > 0:
    print("WARNING: Some customers are missing predictions.")

if len(extra_predictions) > 0:
    print("WARNING: Prediction file contains extra customers.")


# ============================================================
# MERGE ACTUAL + PREDICTED
# ============================================================

print()
print("=" * 80)
print("MERGING ACTUAL LABELS WITH PREDICTIONS")
print("=" * 80)

actual = features[
    [
        "client_id",
        "churn"
    ]
].copy()

pred = predictions[
    [
        "client_id",
        "churn_probability",
        "churn_prediction"
    ]
].copy()

validation = actual.merge(
    pred,
    on="client_id",
    how="inner",
    validate="one_to_one"
)

print("Merged validation rows:", len(validation))


# ============================================================
# CHECK MERGE
# ============================================================

if len(validation) != len(actual):
    raise ValueError(
        "Validation merge lost customers. "
        "Check client_id alignment."
    )

if validation["client_id"].nunique() != len(validation):
    raise ValueError(
        "Duplicate client IDs found after merge."
    )

print("All customer IDs aligned successfully.")


# ============================================================
# CLEAN DATA
# ============================================================

validation = validation.dropna(
    subset=[
        "churn",
        "churn_probability",
        "churn_prediction"
    ]
).copy()

validation["churn"] = validation["churn"].astype(int)
validation["churn_prediction"] = validation["churn_prediction"].astype(int)

validation["churn_probability"] = (
    validation["churn_probability"].astype(float)
)


# ============================================================
# BASIC DISTRIBUTION
# ============================================================

print()
print("=" * 80)
print("LABEL DISTRIBUTION")
print("=" * 80)

actual_churn_count = validation["churn"].sum()
actual_retained_count = len(validation) - actual_churn_count

predicted_churn_count = validation["churn_prediction"].sum()
predicted_retained_count = (
    len(validation) - predicted_churn_count
)

print("Actual churn:", actual_churn_count)
print("Actual retained:", actual_retained_count)

print("Predicted churn:", predicted_churn_count)
print("Predicted retained:", predicted_retained_count)

print(
    "Actual churn rate: {:.2f}%".format(
        validation["churn"].mean() * 100
    )
)

print(
    "Predicted churn rate: {:.2f}%".format(
        validation["churn_prediction"].mean() * 100
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print()
print("=" * 80)
print("CONFUSION MATRIX")
print("=" * 80)

y_true = validation["churn"]
y_pred = validation["churn_prediction"]
y_prob = validation["churn_probability"]

cm = confusion_matrix(
    y_true,
    y_pred
)

tn, fp, fn, tp = cm.ravel()

print()
print("                 Predicted 0    Predicted 1")
print("Actual 0         {:12d}    {:12d}".format(tn, fp))
print("Actual 1         {:12d}    {:12d}".format(fn, tp))

print()
print("True Negatives :", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives :", tp)


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

print()
print("=" * 80)
print("CLASSIFICATION METRICS")
print("=" * 80)

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_true,
    y_prob
)

pr_auc = average_precision_score(
    y_true,
    y_prob
)

print("Accuracy :", round(accuracy, 6))
print("Precision:", round(precision, 6))
print("Recall   :", round(recall, 6))
print("F1 Score :", round(f1, 6))
print("ROC-AUC  :", round(roc_auc, 6))
print("PR-AUC   :", round(pr_auc, 6))


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("=" * 80)
print("CLASSIFICATION REPORT")
print("=" * 80)

report = classification_report(
    y_true,
    y_pred,
    target_names=[
        "Retained",
        "Churned"
    ],
    zero_division=0
)

print(report)


# ============================================================
# PREDICTION PROBABILITY CHECK
# ============================================================

print()
print("=" * 80)
print("PROBABILITY CHECK")
print("=" * 80)

print(
    "Minimum probability:",
    round(y_prob.min(), 6)
)

print(
    "Maximum probability:",
    round(y_prob.max(), 6)
)

print(
    "Mean probability:",
    round(y_prob.mean(), 6)
)

print(
    "Median probability:",
    round(y_prob.median(), 6)
)

invalid_probabilities = (
    (y_prob < 0) |
    (y_prob > 1)
).sum()

print(
    "Invalid probabilities:",
    invalid_probabilities
)

if invalid_probabilities > 0:
    raise ValueError(
        "Invalid churn probabilities detected."
    )


# ============================================================
# PREDICTION CONSISTENCY CHECK
# ============================================================

print()
print("=" * 80)
print("PREDICTION CONSISTENCY CHECK")
print("=" * 80)

# Final model threshold
FINAL_THRESHOLD = 0.38

recomputed_prediction = (
    y_prob >= FINAL_THRESHOLD
).astype(int)

prediction_matches = (
    recomputed_prediction.values ==
    y_pred.values
)

match_count = prediction_matches.sum()
mismatch_count = len(prediction_matches) - match_count

match_percentage = (
    match_count /
    len(prediction_matches)
) * 100

print(
    "Final threshold:",
    FINAL_THRESHOLD
)

print(
    "Matching predictions:",
    match_count
)

print(
    "Mismatching predictions:",
    mismatch_count
)

print(
    "Prediction consistency: {:.4f}%".format(
        match_percentage
    )
)

if mismatch_count == 0:
    print("Prediction consistency check PASSED.")
else:
    print("Prediction consistency check FAILED.")


# ============================================================
# SAVE VALIDATION DATA
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

validation.to_csv(
    VALIDATION_CSV,
    index=False
)


# ============================================================
# SAVE REPORT
# ============================================================

report_lines = []

report_lines.append("=" * 80)
report_lines.append("BATCH PREDICTION VALIDATION REPORT")
report_lines.append("=" * 80)
report_lines.append("")

report_lines.append(
    "Feature dataset rows: {}".format(len(features))
)

report_lines.append(
    "Prediction dataset rows: {}".format(len(predictions))
)

report_lines.append(
    "Validation rows: {}".format(len(validation))
)

report_lines.append(
    "Missing predictions: {}".format(len(missing_predictions))
)

report_lines.append(
    "Extra predictions: {}".format(len(extra_predictions))
)

report_lines.append("")

report_lines.append("ACTUAL LABELS")
report_lines.append(
    "Actual churn: {}".format(actual_churn_count)
)

report_lines.append(
    "Actual retained: {}".format(actual_retained_count)
)

report_lines.append(
    "Actual churn rate: {:.4f}%".format(
        validation["churn"].mean() * 100
    )
)

report_lines.append("")

report_lines.append("PREDICTIONS")
report_lines.append(
    "Predicted churn: {}".format(predicted_churn_count)
)

report_lines.append(
    "Predicted retained: {}".format(predicted_retained_count)
)

report_lines.append(
    "Predicted churn rate: {:.4f}%".format(
        validation["churn_prediction"].mean() * 100
    )
)

report_lines.append("")

report_lines.append("CONFUSION MATRIX")
report_lines.append(
    "TN: {}".format(tn)
)

report_lines.append(
    "FP: {}".format(fp)
)

report_lines.append(
    "FN: {}".format(fn)
)

report_lines.append(
    "TP: {}".format(tp)
)

report_lines.append("")

report_lines.append("METRICS")
report_lines.append(
    "Accuracy: {:.6f}".format(accuracy)
)

report_lines.append(
    "Precision: {:.6f}".format(precision)
)

report_lines.append(
    "Recall: {:.6f}".format(recall)
)

report_lines.append(
    "F1: {:.6f}".format(f1)
)

report_lines.append(
    "ROC-AUC: {:.6f}".format(roc_auc)
)

report_lines.append(
    "PR-AUC: {:.6f}".format(pr_auc)
)

report_lines.append("")

report_lines.append("PREDICTION CONSISTENCY")
report_lines.append(
    "Threshold: {:.2f}".format(FINAL_THRESHOLD)
)

report_lines.append(
    "Matching predictions: {}".format(match_count)
)

report_lines.append(
    "Mismatching predictions: {}".format(mismatch_count)
)

report_lines.append(
    "Consistency: {:.4f}%".format(
        match_percentage
    )
)

report_lines.append("")

if mismatch_count == 0:
    report_lines.append(
        "STATUS: PASSED"
    )
else:
    report_lines.append(
        "STATUS: FAILED"
    )


with open(
    VALIDATION_REPORT,
    "w",
    encoding="utf-8"
) as f:
    f.write("\n".join(report_lines))


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 80)
print("VALIDATION COMPLETE")
print("=" * 80)

print(
    "Validation CSV:",
    VALIDATION_CSV
)

print(
    "Validation report:",
    VALIDATION_REPORT
)

if (
    mismatch_count == 0
    and len(missing_predictions) == 0
    and len(extra_predictions) == 0
):
    print()
    print("FINAL STATUS: PASSED")
    print("Batch predictions are correctly aligned with the labels.")
else:
    print()
    print("FINAL STATUS: CHECK REQUIRED")
    print("Review the validation output before proceeding.")