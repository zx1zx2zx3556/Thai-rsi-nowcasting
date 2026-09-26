import random
import warnings
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import lightning.pytorch as pl

from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import t
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import MAE

warnings.filterwarnings("ignore")

# =========================================================
# 0) REPRODUCIBILITY
# =========================================================
SEED = 42

def reset_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pl.seed_everything(seed, workers=True)

reset_seeds(SEED)

# =========================================================
# 1) LOAD DATA
# =========================================================
df_a = pd.read_csv(
    "/Users/methask./Retail_sales 2015 - 2023/scenario_a_full_df.csv",
    parse_dates=["date"]
)

df_a = df_a.sort_values("date").reset_index(drop=True)
df_a["group_id"] = "thai_rsi"
df_a["time_idx"] = np.arange(len(df_a))

# =========================================================
# 2) BEST PARAMS FROM ROUND 3 (Scenario A)
# =========================================================
best_params_a = {
    "learning_rate": 0.0002549438150777045,
    "hidden_size": 64,
    "attention_head_size": 4,
    "dropout": 0.18589201570671932,
    "lstm_layers": 2,
    "batch_size": 64,
    "hidden_continuous_size": 4,
    "max_epochs": 50
}

# =========================================================
# 3) FINAL SETTINGS FROM UPDATED ROUND 4
# =========================================================
BEST_LOOKBACK = 36
PREDICTION_LENGTH = 1
ENCODER_LENGTH_FIXED = 36

# =========================================================
# 4) METRICS
# =========================================================
def mape(y_true, y_pred, eps=1e-6):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100

def smape(y_true, y_pred, eps=1e-6):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    denom = np.maximum(denom, eps)
    return np.mean(np.abs(y_true - y_pred) / denom) * 100

def dm_test(e1, e2, h=1):
    """
    Harvey-Leybourne-Newbold (HLN) adjusted
    Diebold-Mariano (DM) test.

    e1 : forecast errors from Model 1
    e2 : forecast errors from Model 2
    h  : forecast horizon
         h=1 for one-step-ahead nowcasting
    """

    e1 = np.asarray(e1, dtype=float)
    e2 = np.asarray(e2, dtype=float)

    # Make sure both error series have the same length
    n = min(len(e1), len(e2))
    e1 = e1[:n]
    e2 = e2[:n]

    # Squared-error loss differential
    d = e1**2 - e2**2

    T = len(d)

    # Need at least 2 observations
    if T < 2:
        return np.nan, np.nan

    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)

    # Avoid division by zero / invalid variance
    if var_d == 0 or np.isnan(var_d):
        return np.nan, np.nan

    # Original Diebold-Mariano statistic
    dm_stat = mean_d / np.sqrt(var_d / T)

    # Harvey-Leybourne-Newbold small-sample correction
    hln_factor = np.sqrt(
        (T + 1 - 2*h + h*(h - 1)/T) / T
    )

    # HLN-adjusted DM statistic
    hln_stat = dm_stat * hln_factor

    # Two-sided p-value
    p_value = 2 * t.sf(
        np.abs(hln_stat),
        df=T - 1
    )

    return hln_stat, p_value

# =========================================================
# 5) RevIN HELPERS (Dynamic)
# =========================================================
def safe_sd(row):
    s = np.std(row, ddof=1)
    if np.isnan(s) or s == 0:
        return 1.0
    return s

def revin_group(df, cols, prefix):
    out = df.copy()
    if len(cols) == 0:
        return out

    mat = out[cols].to_numpy(dtype=float)
    row_means = np.nanmean(mat, axis=1)
    row_sds = np.array([safe_sd(row) for row in mat])

    norm_mat = (mat - row_means[:, None]) / row_sds[:, None]
    out.loc[:, cols] = norm_mat
    out[f"{prefix}_mean"] = row_means
    out[f"{prefix}_sd"] = row_sds
    return out

def apply_revin_dynamic(df, feature_cols):
    out = df.copy()

    rsi_cols = [c for c in ["RSI_lag1", "RSI_lag2", "RSI_lag3"] if c in out.columns]
    if len(rsi_cols) != 3:
        raise ValueError("RSI_lag1-3 are required for Round 5.")
    out = revin_group(out, rsi_cols, "RSI_hist")

    pci_cols = [c for c in ["PCI_lag1", "PCI_lag2", "PCI_lag3"] if c in feature_cols and c in out.columns]
    cpi_cols = [c for c in ["CPI_lag1", "CPI_lag2", "CPI_lag3"] if c in feature_cols and c in out.columns]
    cci_cols = [c for c in ["CCI_lag1", "CCI_lag2", "CCI_lag3"] if c in feature_cols and c in out.columns]

    if len(pci_cols) == 3:
        out = revin_group(out, pci_cols, "PCI_hist")
    if len(cpi_cols) == 3:
        out = revin_group(out, cpi_cols, "CPI_hist")
    if len(cci_cols) == 3:
        out = revin_group(out, cci_cols, "CCI_hist")

    digital_groups = {
        "Discount_hist":  ["Discount_rm3_lag1", "Discount_rm3_lag2", "Discount_rm3_lag3"],
        "Promotion_hist": ["Promotion_rm3_lag1", "Promotion_rm3_lag2", "Promotion_rm3_lag3"],
        "OnlineBuy_hist": ["OnlineBuy_rm3_lag1", "OnlineBuy_rm3_lag2", "OnlineBuy_rm3_lag3"],
        "Coupon_hist":    ["Coupon_rm3_lag1", "Coupon_rm3_lag2", "Coupon_rm3_lag3"],
        "Gift_hist":      ["Gift_rm3_lag1", "Gift_rm3_lag2", "Gift_rm3_lag3"]
    }

    for prefix, cols in digital_groups.items():
        existing_cols = [c for c in cols if c in feature_cols and c in out.columns]
        if len(existing_cols) == 3:
            out = revin_group(out, existing_cols, prefix)

    out["target_raw"] = out["RSI"]
    out["target_norm"] = (out["RSI"] - out["RSI_hist_mean"]) / out["RSI_hist_sd"]
    out["target_mean"] = out["RSI_hist_mean"]
    out["target_sd"] = out["RSI_hist_sd"]

    return out

# =========================================================
# 6) FEATURE GROUPS FOR ABLATION
# =========================================================
rsi_base_features = [
    "RSI_lag1", "RSI_lag2", "RSI_lag3"
]

official_features = [
    "PCI_lag1", "PCI_lag2", "PCI_lag3",
    "CPI_lag1", "CPI_lag2", "CPI_lag3"
]

cci_candidates = ["CCI_lag1", "CCI_lag2", "CCI_lag3"]
if all(col in df_a.columns for col in cci_candidates):
    official_features += cci_candidates
    print("CCI columns detected and included in official features.")
else:
    print("CCI columns not found in dataset. Official features will use PCI and CPI only.")

digital_features = [
    "Discount_rm3_lag1", "Discount_rm3_lag2", "Discount_rm3_lag3",
    "Promotion_rm3_lag1", "Promotion_rm3_lag2", "Promotion_rm3_lag3",
    "OnlineBuy_rm3_lag1", "OnlineBuy_rm3_lag2", "OnlineBuy_rm3_lag3",
    "Coupon_rm3_lag1", "Coupon_rm3_lag2", "Coupon_rm3_lag3",
    "Gift_rm3_lag1", "Gift_rm3_lag2", "Gift_rm3_lag3"
]

rsi_base_features = [c for c in rsi_base_features if c in df_a.columns]
official_features = [c for c in official_features if c in df_a.columns]
digital_features = [c for c in digital_features if c in df_a.columns]

feature_sets = {
    "Scenario C (Only Official)": rsi_base_features + official_features,
    "Scenario D (Only Digital)": rsi_base_features + digital_features,
    "Scenario E (Combined)": rsi_base_features + official_features + digital_features
}

print("\n=== Feature sets for Round 5 ===")
for name, cols in feature_sets.items():
    print(f"{name}: {len(cols)} features")
    print(cols)

# =========================================================
# 7) DATASET BUILDER
# =========================================================
def build_training_dataset(train_df, feature_cols, encoder_length):
    return TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="target_norm",
        group_ids=["group_id"],
        min_encoder_length=encoder_length,
        max_encoder_length=encoder_length,
        min_prediction_length=PREDICTION_LENGTH,
        max_prediction_length=PREDICTION_LENGTH,
        static_categoricals=["group_id"],
        time_varying_known_reals=["time_idx"] + feature_cols,
        time_varying_unknown_reals=["target_norm"],
        target_normalizer=None,
        add_relative_time_idx=True,
        add_target_scales=False,
        add_encoder_length=True,
        allow_missing_timesteps=False
    )

# =========================================================
# 8) SHARED VALID FOLDS
# =========================================================
def get_valid_splits(df, lookback, prediction_length):
    tscv = TimeSeriesSplit(n_splits=5)
    valid_splits = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(df), start=1):
        train_len = len(train_idx)
        required_len = lookback + prediction_length

        if train_len < required_len:
            print(f"Fold {fold} skipped globally: train length ({train_len}) < required ({required_len})")
            continue

        valid_splits.append((fold, train_idx, test_idx))

    if len(valid_splits) == 0:
        raise ValueError("No valid folds available under the current lookback setting.")

    print(f"\nValid folds for Round 5: {[fold for fold, _, _ in valid_splits]}")
    return valid_splits

VALID_SPLITS = get_valid_splits(df_a, BEST_LOOKBACK, PREDICTION_LENGTH)

# =========================================================
# 9) ROUND 5 ABLATION EXPERIMENT
# =========================================================
def run_tft_ablation(df, feature_cols, scenario_name, best_params, lookback, valid_splits):
    metrics_list = []
    preds_list = []

    for fold, train_idx, test_idx in valid_splits:
        try:
            reset_seeds(SEED)

            train_df = df.iloc[train_idx].copy().reset_index(drop=True)
            test_df = df.iloc[test_idx].copy().reset_index(drop=True)

            train_df = apply_revin_dynamic(train_df, feature_cols)
            test_df = apply_revin_dynamic(test_df, feature_cols)

            training = build_training_dataset(
                train_df=train_df,
                feature_cols=feature_cols,
                encoder_length=lookback
            )

            validation_source = pd.concat(
                [train_df.tail(lookback), test_df],
                axis=0
            ).reset_index(drop=True)

            validation = TimeSeriesDataSet.from_dataset(
                training,
                validation_source,
                predict=True,
                stop_randomization=True
            )

            train_loader = training.to_dataloader(
                train=True,
                batch_size=best_params["batch_size"],
                num_workers=0,
                shuffle=False
            )

            val_loader = validation.to_dataloader(
                train=False,
                batch_size=best_params["batch_size"],
                num_workers=0,
                shuffle=False
            )

            tft = TemporalFusionTransformer.from_dataset(
                training,
                learning_rate=best_params["learning_rate"],
                hidden_size=best_params["hidden_size"],
                attention_head_size=best_params["attention_head_size"],
                dropout=best_params["dropout"],
                hidden_continuous_size=best_params["hidden_continuous_size"],
                lstm_layers=best_params["lstm_layers"],
                loss=MAE(),
                output_size=1,
                log_interval=-1,
                reduce_on_plateau_patience=4
            )

            trainer = pl.Trainer(
                max_epochs=best_params["max_epochs"],
                accelerator="auto",
                devices=1,
                gradient_clip_val=0.1,
                logger=False,
                enable_checkpointing=False,
                enable_model_summary=False,
                deterministic=True
            )

            trainer.fit(tft, train_dataloaders=train_loader)

            pred_norm = tft.predict(val_loader)
            pred_norm = pred_norm.detach().cpu().numpy().reshape(-1)

            n = min(len(pred_norm), len(test_df))
            pred_norm = pred_norm[:n]
            eval_df = test_df.iloc[:n].copy()

            pred_raw = pred_norm * eval_df["target_sd"].values + eval_df["target_mean"].values
            true_raw = eval_df["target_raw"].values

            metrics_list.append({
                "scenario": scenario_name,
                "fold": fold,
                "train_size": len(train_df),
                "test_size": len(test_df),
                "MAE": np.mean(np.abs(true_raw - pred_raw)),
                "MAPE": mape(true_raw, pred_raw),
                "sMAPE": smape(true_raw, pred_raw)
            })

            preds_list.append(pd.DataFrame({
                "scenario": scenario_name,
                "fold": fold,
                "date": eval_df["date"].values,
                "actual": true_raw,
                "pred": pred_raw,
                "abs_error": np.abs(true_raw - pred_raw)
            }))

        except Exception as e:
            print(f"Error in {scenario_name}, Fold {fold}: {e}")
            continue

    if len(metrics_list) == 0:
        raise ValueError(f"No valid folds were generated for {scenario_name}.")

    return pd.DataFrame(metrics_list), pd.concat(preds_list, ignore_index=True)

# =========================================================
# 10) RUN ROUND 5
# =========================================================
all_metrics = []
all_preds = []

for scenario_name, feature_cols in feature_sets.items():
    metrics_df, preds_df = run_tft_ablation(
        df=df_a,
        feature_cols=feature_cols,
        scenario_name=scenario_name,
        best_params=best_params_a,
        lookback=BEST_LOOKBACK,
        valid_splits=VALID_SPLITS
    )
    all_metrics.append(metrics_df)
    all_preds.append(preds_df)

metrics_r5 = pd.concat(all_metrics, ignore_index=True)
preds_r5 = pd.concat(all_preds, ignore_index=True)

# =========================================================
# 11) SUMMARY TABLES
# =========================================================
summary_mean_r5 = metrics_r5.groupby("scenario")[["MAE", "MAPE", "sMAPE"]].mean().reset_index()
summary_std_r5 = metrics_r5.groupby("scenario")[["MAE", "MAPE", "sMAPE"]].std().reset_index()

print("\n=== TFT Round 5 Mean ===")
print(summary_mean_r5)

print("\n=== TFT Round 5 Std ===")
print(summary_std_r5)

metrics_r5.to_csv("tft_round5_metrics_all.csv", index=False)
summary_mean_r5.to_csv("tft_round5_mean.csv", index=False)
summary_std_r5.to_csv("tft_round5_std.csv", index=False)
preds_r5.to_csv("tft_round5_predictions_all.csv", index=False)

# =========================================================
# 12) ADDED VALUE TABLE (relative to Scenario C)
# =========================================================
mean_table = summary_mean_r5.set_index("scenario")

baseline_mae = mean_table.loc["Scenario C (Only Official)", "MAE"]
baseline_mape = mean_table.loc["Scenario C (Only Official)", "MAPE"]
baseline_smape = mean_table.loc["Scenario C (Only Official)", "sMAPE"]

added_value_rows = []
for scenario_name in summary_mean_r5["scenario"]:
    mae_val = mean_table.loc[scenario_name, "MAE"]
    mape_val = mean_table.loc[scenario_name, "MAPE"]
    smape_val = mean_table.loc[scenario_name, "sMAPE"]

    added_value_rows.append({
        "scenario": scenario_name,
        "MAE": mae_val,
        "MAPE": mape_val,
        "sMAPE": smape_val,
        "MAE_improvement_vs_C_pct": ((baseline_mae - mae_val) / baseline_mae) * 100,
        "MAPE_improvement_vs_C_pct": ((baseline_mape - mape_val) / baseline_mape) * 100,
        "sMAPE_improvement_vs_C_pct": ((baseline_smape - smape_val) / baseline_smape) * 100
    })

added_value_df = pd.DataFrame(added_value_rows)
print("\n=== Added Value Table (vs Scenario C) ===")
print(added_value_df)

added_value_df.to_csv("tft_round5_added_value_vs_c.csv", index=False)

# =========================================================
# 13) DM TESTS
# =========================================================
def run_pairwise_dm(preds_df, scenario_left, scenario_right):
    left = preds_df[
        preds_df["scenario"] == scenario_left
    ].copy()

    right = preds_df[
        preds_df["scenario"] == scenario_right
    ].copy()

    # Pool out-of-sample forecast errors
    # across all rolling-origin folds
    dm_df = left.merge(
        right,
        on=["date", "fold"],
        suffixes=("_left", "_right")
    ).sort_values(["fold", "date"])

    # Forecast errors
    err_left = (
        dm_df["actual_left"] -
        dm_df["pred_left"]
    )

    err_right = (
        dm_df["actual_right"] -
        dm_df["pred_right"]
    )

    # HLN-adjusted DM test
    dm_stat, p_value = dm_test(
        err_left.values,
        err_right.values,
        h=1
    )

    return {
        "scenario_left": scenario_left,
        "scenario_right": scenario_right,
        "DM_statistic": dm_stat,
        "p_value": p_value
    }

# ============================================================
# TFT ROUND 5: HLN-ADJUSTED DIEBOLD-MARIANO TEST
# Pooled out-of-sample forecast errors across 5 folds
# ============================================================

dm_results = pd.DataFrame([
    run_pairwise_dm(
        preds_r5,
        "Scenario C (Only Official)",
        "Scenario E (Combined)"
    ),
    run_pairwise_dm(
        preds_r5,
        "Scenario C (Only Official)",
        "Scenario D (Only Digital)"
    ),
    run_pairwise_dm(
        preds_r5,
        "Scenario D (Only Digital)",
        "Scenario E (Combined)"
    )
])

print("\n=== TFT Round 5 HLN-Adjusted Diebold-Mariano Tests ===")
print(dm_results)

dm_results.to_csv(
    "tft_round5_hln_dm_tests.csv",
    index=False
)

dm_results.to_csv("tft_round5_dm_tests.csv", index=False)

# =========================================================
# 14) PLOTS
# =========================================================
metric_names = ["MAE", "MAPE", "sMAPE"]
x = np.arange(len(metric_names))

plt.figure(figsize=(8, 5))
for scenario_name in summary_mean_r5["scenario"]:
    row = summary_mean_r5[summary_mean_r5["scenario"] == scenario_name].iloc[0]
    vals = [row["MAE"], row["MAPE"], row["sMAPE"]]
    plt.plot(x, vals, marker="o", linewidth=2, label=scenario_name)

plt.xticks(x, metric_names)
plt.ylabel("Mean Error Value")
plt.title("TFT Round 5: Mean Forecasting Error Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("tft_round5_mean_metrics_line.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, 5))
plt.bar(summary_mean_r5["scenario"], summary_mean_r5["MAPE"])
plt.ylabel("MAPE")
plt.title("TFT Round 5: MAPE Comparison")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("tft_round5_mape_bar.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, 5))
pair_labels = [
    f"{row['scenario_left']} vs\n{row['scenario_right']}"
    for _, row in dm_results.iterrows()
]
plt.bar(pair_labels, dm_results["p_value"])
plt.axhline(0.05, linestyle="--")
plt.ylabel("p-value")
plt.title("TFT Round 5: Diebold-Mariano Test p-values")
plt.tight_layout()
plt.savefig("tft_round5_dm_pvalues.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nTFT Round 5 finished successfully.")