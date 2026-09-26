import os
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
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
pl.seed_everything(SEED, workers=True)

# =========================================================
# 1) LOAD DATA
# =========================================================
df_a = pd.read_csv(
    "/Users/methask./Retail_sales 2015 - 2023/scenario_a_full_df.csv",
    parse_dates=["date"]
)
df_b = pd.read_csv(
    "/Users/methask./Retail_sales 2015 - 2023/scenario_b_full_df.csv",
    parse_dates=["date"]
)

df_a = df_a.sort_values("date").reset_index(drop=True)
df_b = df_b.sort_values("date").reset_index(drop=True)

# add fields for TFT
df_a["group_id"] = "thai_rsi"
df_b["group_id"] = "thai_rsi"
df_a["time_idx"] = np.arange(len(df_a))
df_b["time_idx"] = np.arange(len(df_b))

# =========================================================
# 2) METRICS
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

def dm_test(e1, e2):
    d = e1**2 - e2**2
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)
    if var_d == 0:
        return np.nan, np.nan
    dm_stat = mean_d / np.sqrt(var_d / len(d))
    p_value = 2 * (1 - t.cdf(np.abs(dm_stat), df=len(d)-1))
    return dm_stat, p_value

# =========================================================
# 3) RevIN HELPERS
# =========================================================
def safe_sd(row):
    s = np.std(row, ddof=1)
    if np.isnan(s) or s == 0:
        return 1.0
    return s

def revin_group(df, cols, prefix):
    out = df.copy()
    mat = out[cols].to_numpy(dtype=float)

    row_means = np.nanmean(mat, axis=1)
    row_sds = np.array([safe_sd(row) for row in mat])

    norm_mat = (mat - row_means[:, None]) / row_sds[:, None]
    out.loc[:, cols] = norm_mat
    out[f"{prefix}_mean"] = row_means
    out[f"{prefix}_sd"] = row_sds
    return out

def apply_revin_scenario_a(df):
    out = df.copy()

    out = revin_group(out, ["RSI_lag1", "RSI_lag2", "RSI_lag3"], "RSI_hist")
    out = revin_group(out, ["PCI_lag1", "PCI_lag2", "PCI_lag3"], "PCI_hist")
    out = revin_group(out, ["CPI_lag1", "CPI_lag2", "CPI_lag3"], "CPI_hist")

    monthly_groups = {
        "Discount_hist":  ["Discount_rm3_lag1", "Discount_rm3_lag2", "Discount_rm3_lag3"],
        "Promotion_hist": ["Promotion_rm3_lag1", "Promotion_rm3_lag2", "Promotion_rm3_lag3"],
        "OnlineBuy_hist": ["OnlineBuy_rm3_lag1", "OnlineBuy_rm3_lag2", "OnlineBuy_rm3_lag3"],
        "Coupon_hist":    ["Coupon_rm3_lag1", "Coupon_rm3_lag2", "Coupon_rm3_lag3"],
        "Gift_hist":      ["Gift_rm3_lag1", "Gift_rm3_lag2", "Gift_rm3_lag3"]
    }

    for prefix, cols in monthly_groups.items():
        out = revin_group(out, cols, prefix)

    out["target_raw"] = out["RSI"]
    out["target_norm"] = (out["RSI"] - out["RSI_hist_mean"]) / out["RSI_hist_sd"]
    out["target_mean"] = out["RSI_hist_mean"]
    out["target_sd"] = out["RSI_hist_sd"]
    return out

def apply_revin_scenario_b(df):
    out = df.copy()

    out = revin_group(out, ["RSI_lag1", "RSI_lag2", "RSI_lag3"], "RSI_hist")
    out = revin_group(out, ["PCI_lag1", "PCI_lag2", "PCI_lag3"], "PCI_hist")
    out = revin_group(out, ["CPI_lag1", "CPI_lag2", "CPI_lag3"], "CPI_hist")

    weekly_groups = {
        "Discount_hist":  ["Discount_lag4", "Discount_lag8", "Discount_lag12"],
        "Promotion_hist": ["Promotion_lag4", "Promotion_lag8", "Promotion_lag12"],
        "OnlineBuy_hist": ["OnlineBuy_lag4", "OnlineBuy_lag8", "OnlineBuy_lag12"],
        "Coupon_hist":    ["Coupon_lag4", "Coupon_lag8", "Coupon_lag12"],
        "Gift_hist":      ["Gift_lag4", "Gift_lag8", "Gift_lag12"]
    }

    for prefix, cols in weekly_groups.items():
        out = revin_group(out, cols, prefix)

    out["target_raw"] = out["RSI"]
    out["target_norm"] = (out["RSI"] - out["RSI_hist_mean"]) / out["RSI_hist_sd"]
    out["target_mean"] = out["RSI_hist_mean"]
    out["target_sd"] = out["RSI_hist_sd"]
    return out

# =========================================================
# 4) FEATURE COLUMNS
# =========================================================
feature_cols_a = [
    "RSI_lag1", "RSI_lag2", "RSI_lag3",
    "PCI_lag1", "PCI_lag2", "PCI_lag3",
    "CPI_lag1", "CPI_lag2", "CPI_lag3",
    "Discount_rm3_lag1", "Discount_rm3_lag2", "Discount_rm3_lag3",
    "Promotion_rm3_lag1", "Promotion_rm3_lag2", "Promotion_rm3_lag3",
    "OnlineBuy_rm3_lag1", "OnlineBuy_rm3_lag2", "OnlineBuy_rm3_lag3",
    "Coupon_rm3_lag1", "Coupon_rm3_lag2", "Coupon_rm3_lag3",
    "Gift_rm3_lag1", "Gift_rm3_lag2", "Gift_rm3_lag3"
]

feature_cols_b = [
    "RSI_lag1", "RSI_lag2", "RSI_lag3",
    "PCI_lag1", "PCI_lag2", "PCI_lag3",
    "CPI_lag1", "CPI_lag2", "CPI_lag3",
    "Discount_lag4", "Discount_lag8", "Discount_lag12",
    "Promotion_lag4", "Promotion_lag8", "Promotion_lag12",
    "OnlineBuy_lag4", "OnlineBuy_lag8", "OnlineBuy_lag12",
    "Coupon_lag4", "Coupon_lag8", "Coupon_lag12",
    "Gift_lag4", "Gift_lag8", "Gift_lag12"
]

# =========================================================
# 5) BASELINE TFT SETTINGS
# =========================================================
ENCODER_LENGTH = 12
PREDICTION_LENGTH = 1

BASELINE_PARAMS = {
    "learning_rate": 1e-3,
    "hidden_size": 16,
    "attention_head_size": 1,
    "dropout": 0.1,
    "hidden_continuous_size": 8,
    "batch_size": 32,
    "max_epochs": 30,
    "gradient_clip_val": 0.1
}

# =========================================================
# 6) ROUND 2: 5-FOLD TIME SERIES CV
# =========================================================
def run_tft_cv(df, feature_cols, scenario_name, revin_func):
    tscv = TimeSeriesSplit(n_splits=5)

    metrics_list = []
    preds_list = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(df), start=1):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()

        # apply RevIN inside each fold
        train_df = revin_func(train_df)
        test_df = revin_func(test_df)

        # TFT datasets
        training = TimeSeriesDataSet(
            train_df,
            time_idx="time_idx",
            target="target_norm",
            group_ids=["group_id"],
            min_encoder_length=ENCODER_LENGTH,
            max_encoder_length=ENCODER_LENGTH,
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

        validation = TimeSeriesDataSet.from_dataset(
            training,
            pd.concat([train_df.tail(ENCODER_LENGTH), test_df], axis=0),
            predict=True,
            stop_randomization=True
        )

        train_loader = training.to_dataloader(
            train=True,
            batch_size=BASELINE_PARAMS["batch_size"],
            num_workers=0
        )

        val_loader = validation.to_dataloader(
            train=False,
            batch_size=BASELINE_PARAMS["batch_size"],
            num_workers=0
        )

        # model
        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=BASELINE_PARAMS["learning_rate"],
            hidden_size=BASELINE_PARAMS["hidden_size"],
            attention_head_size=BASELINE_PARAMS["attention_head_size"],
            dropout=BASELINE_PARAMS["dropout"],
            hidden_continuous_size=BASELINE_PARAMS["hidden_continuous_size"],
            loss=MAE(),
            output_size=1,
            log_interval=-1,
            reduce_on_plateau_patience=4
        )

        trainer = pl.Trainer(
            max_epochs=BASELINE_PARAMS["max_epochs"],
            accelerator="auto",
            devices=1,
            gradient_clip_val=BASELINE_PARAMS["gradient_clip_val"],
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

    return pd.DataFrame(metrics_list), pd.concat(preds_list, ignore_index=True)

# =========================================================
# 7) RUN ROUND 2
# =========================================================
metrics_a_tft_r2, preds_a_tft_r2 = run_tft_cv(df_a, feature_cols_a, "Scenario A", apply_revin_scenario_a)
metrics_b_tft_r2, preds_b_tft_r2 = run_tft_cv(df_b, feature_cols_b, "Scenario B", apply_revin_scenario_b)

summary_r2 = pd.concat([metrics_a_tft_r2, metrics_b_tft_r2], ignore_index=True)
summary_mean_r2 = summary_r2.groupby("scenario")[["MAE", "MAPE", "sMAPE"]].mean()
summary_std_r2 = summary_r2.groupby("scenario")[["MAE", "MAPE", "sMAPE"]].std()

print("\n=== TFT Round 2 Metrics by Fold ===")
print(summary_r2)

print("\n=== TFT Round 2 Mean ===")
print(summary_mean_r2)

print("\n=== TFT Round 2 Std ===")
print(summary_std_r2)

# =========================================================
# 8) DIEBOLD-MARIANO TEST
# =========================================================
dm_df = preds_a_tft_r2.merge(
    preds_b_tft_r2,
    on="date",
    suffixes=("_a", "_b")
).sort_values("date")

err_a = dm_df["actual_a"] - dm_df["pred_a"]
err_b = dm_df["actual_b"] - dm_df["pred_b"]

dm_stat, p_value = dm_test(err_a.values, err_b.values)

print("\n=== TFT Round 2 Diebold-Mariano Test ===")
print("DM statistic:", dm_stat)
print("p-value:", p_value)

# =========================================================
# 9) SAVE RESULTS
# =========================================================
summary_r2.to_csv("tft_round2_metrics_all.csv", index=False)
summary_mean_r2.to_csv("tft_round2_mean.csv")
summary_std_r2.to_csv("tft_round2_std.csv")
preds_a_tft_r2.to_csv("tft_round2_predictions_scenario_a.csv", index=False)
preds_b_tft_r2.to_csv("tft_round2_predictions_scenario_b.csv", index=False)

# =========================================================
# 10) PLOTS
# =========================================================
def plot_metric_by_fold(metrics_df, metric_name):
    plt.figure(figsize=(8, 5))
    for scenario in metrics_df["scenario"].unique():
        tmp = metrics_df[metrics_df["scenario"] == scenario]
        plt.plot(tmp["fold"], tmp[metric_name], marker="o", linewidth=2, label=scenario)
    plt.title(f"TFT Round 2: {metric_name} by Fold")
    plt.xlabel("Fold")
    plt.ylabel(metric_name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"tft_round2_{metric_name.lower()}_by_fold.png", dpi=300, bbox_inches="tight")
    plt.show()

plot_metric_by_fold(summary_r2, "MAE")
plot_metric_by_fold(summary_r2, "MAPE")
plot_metric_by_fold(summary_r2, "sMAPE")

# mean comparison
mean_a = metrics_a_tft_r2[["MAE", "MAPE", "sMAPE"]].mean()
mean_b = metrics_b_tft_r2[["MAE", "MAPE", "sMAPE"]].mean()

metric_names = ["MAE", "MAPE", "sMAPE"]
scenario_a_mean = [mean_a["MAE"], mean_a["MAPE"], mean_a["sMAPE"]]
scenario_b_mean = [mean_b["MAE"], mean_b["MAPE"], mean_b["sMAPE"]]

x = np.arange(len(metric_names))
plt.figure(figsize=(8, 5))
plt.plot(x, scenario_a_mean, marker="o", linewidth=2, label="Scenario A")
plt.plot(x, scenario_b_mean, marker="o", linewidth=2, label="Scenario B")
plt.xticks(x, metric_names)
plt.ylabel("Mean Error Value")
plt.title("TFT Round 2: Mean Forecasting Error Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("tft_round2_mean_metrics_line.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nTFT Round 2 finished successfully.")