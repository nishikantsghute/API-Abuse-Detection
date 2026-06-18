# API Abuse Detection - ML Pipeline

End-to-end notebook that classifies API traffic sessions as **normal** or **outlier** (abusive) using session-level behavioral statistics and API call-graph structure. The classifier is a Random Forest trained on SMOTE-balanced data, sharing its preprocessing/modeling code with a companion Streamlit app.

## Project Structure

This notebook is one piece of a larger project and imports shared helpers rather than defining everything inline. To run it, the following layout is expected relative to the notebook:

```
project-root/
├── api_abuse_detection.ipynb        ← this notebook
├── streamlit_app/
│   ├── utils/
│   │   ├── preprocessing.py         ← required — not included in this upload
│   │   └── modeling.py              ← required — not included in this upload
│   └── data/
│       └── sample_dataset.csv       ← required — not included in this upload
└── outputs/
    └── model_supervised_binary.joblib   ← created when the notebook runs
```

The notebook does `from utils import modeling as ml` and `from utils import preprocessing as pp`, with `streamlit_app/` added to `sys.path`. Only the `.ipynb` was provided here, so those three files need to be added alongside it before the notebook will execute top to bottom.

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
imbalanced-learn
```

`imbalanced-learn` isn't imported directly in the notebook, but the SMOTE step (`pp.balance_with_smote`) prints a message consistent with wrapping it internally — install it alongside the rest. Developed against Python 3.13.

## Running It

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib imbalanced-learn
jupyter notebook api_abuse_detection.ipynb
```

Run all cells in order. The trained model bundle lands at `outputs/model_supervised_binary.joblib`.

## Dataset

`sample_dataset.csv` is 1,699 rows × 26 columns. After cleaning (median imputation, duplicate removal) it's 1,665 rows with zero missing values and zero duplicates.

**Target:** `classification` — `normal` (1,106 rows) vs `outlier` (559 rows).

**Session / behavioral features:** `inter_api_access_duration(sec)`, `api_access_uniqueness`, `sequence_length(count)`, `vsession_duration(min)`, `ip_type`, `num_sessions`, `num_users`, `num_unique_apis`, `source`

**Call-graph features** (API call sequences modeled as a graph): `g_n_nodes`, `g_n_edges`, `g_n_unique_edges`, `g_edge_node_ratio`, `g_repeat_edge_ratio`, `g_self_loops`, `g_self_loop_ratio`, `g_max_out_degree`, `g_max_in_degree`, `g_mean_out_degree`, `g_mean_in_degree`, `g_density`, `g_n_weakly_conn_comp`, `g_largest_wcc_frac`, `g_n_strongly_conn_comp`, `g_has_cycle`

`engineer_features()` adds three more ratio features before training: `sessions_per_user`, `api_reuse_rate`, `edges_per_node`.

## Pipeline

The notebook is organized into 12 numbered sections:

0. **Setup** — imports, plot style, wires in `utils.preprocessing` / `utils.modeling` from `streamlit_app`.
1. **Data Loading** — reads the CSV, drops index/id-like columns.
2. **Data Cleaning** — median imputation, duplicate removal.
3. **EDA** — class balance plot, summary statistics, correlation heatmap, per-class histograms.
4–8. **Preprocessing** — outlier capping, feature engineering, categorical encoding, stratified 75/25 train/test split, standardization, then SMOTE (applied to the training set only, after the split, to avoid leakage).
9. **Train & Evaluate** — Random Forest (300 trees, balanced class weights), 5-fold cross-validation, full metric suite, confusion matrix, ROC curve, feature importances.
10. **Save Model** — bundles model + scaler + feature names + label classes + metrics into one `joblib` file.
11. **Predict** — builds a single synthetic sample from column medians and scores it as a smoke test.

## Results

From the run captured in this notebook:

| Metric | 5-fold CV | Test set |
|---|---|---|
| ROC-AUC | 1.0000 ± 0.0000 | 1.0000 |
| Accuracy | — | 1.0000 |
| Precision | — | 1.0000 |
| Recall | — | 1.0000 |
| F1 | — | 1.0000 |

Per-class on the held-out test set (n=417): both `normal` (support 277) and `outlier` (support 140) score 1.00 across precision, recall, and F1.

Top 5 features by importance: `num_users`, `num_sessions`, `api_reuse_rate`, `api_access_uniqueness`, `sessions_per_user`.

A caveat worth flagging: perfect scores across every metric, on both cross-validation and a held-out test set, isn't typical for a real-world classification task. Before relying on this model in production, it's worth checking for data leakage (e.g. a feature that's effectively a proxy for the label) or confirming whether `sample_dataset.csv` is synthetic or otherwise cleanly separable by construction.

## Output Artifact

`outputs/model_supervised_binary.joblib` is a dict bundle containing:

- `model` — the fitted `RandomForestClassifier`
- `scaler` — the fitted `StandardScaler`
- `feature_names` — column order expected at inference time
- `label_classes` — the `LabelEncoder` classes (`normal`, `outlier`)
- `metrics` — the test-set metrics above
- `task`, `model_name` — metadata strings

## Inference

The final cell scores one row built from dataset medians via `ml.predict_frame(bundle, sample_df)`, after running it through the same `engineer_features` → `encode_categoricals` → scaling steps used in training — on that demo row, the model predicts `normal` with ~92% probability. A comment in the notebook references a `predict.py --demo` script that does the same thing from the command line; that script isn't part of this upload, so you'd need to reuse this cell's logic if a standalone CLI is required.
