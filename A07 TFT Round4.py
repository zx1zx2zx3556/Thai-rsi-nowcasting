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
df_b = pd.read_csv(
    "/Users/methask./Retail_sales 2015 - 2023/scenario_b_full_df.csv",
    parse_dates=["date"]
)

df_a = df_a.sort_values("date").reset_index(drop=True)
df_b = df_b.sort_values("date").reset_index(drop=True)

df_a["group_id"] = "thai_rsi"
df_b["group_id"] = "thai_rsi"
df_a["time_idx"] = np.arange(len(df_a))
df_b["time_idx"] = np.arange(len(df_b))

# =========================================================
# 2) BEST PARAMS FROM ROUND 3
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

best_params_b = {
    "learning_rate": 0.00010218376758008094,
    "hidden_size": 8,
    "attention_head_size": 2,
    "dropout": 0.07224542260010827,
    "lstm_layers": 1,
    "batch_size": 32,
    "hidden_continuous_size": 4,
    "max_epochs": 40
}

# =========================================================
# 3) METRICS
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

    if var_d == 0 or np.isnan(var_d):
        return np.nan, np.nan

    dm_stat = mean_d / np.sqrt(var_d / len(d))
    p_value = 2 * (1 - t.cdf(np.abs(dm_stat), df=len(d) - 1))
    return dm_stat, p_value

# =========================================================
# 4) RevIN HELPERS
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
# 5) FEATURE COLUMNS
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
# 6) LOOK-BACK SETTINGS
# =========================================================
lookback_windows = [24, 36, 48, 60]
PREDICTION_LENGTH = 1

def build_training_dataset_lookback(train_df, feature_cols, encoder_length):
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
# 7) ROUND 4 LOOK-BACK EXPERIMENT
# =========================================================
def run_tft_lookback(df, feature_cols, scenario_name, revin_func, best_params, lookback_list):
    metrics_list = []
    preds_list = []

    for lookback in lookback_list:
        print(f"\n[{scenario_name}] Lookback = {lookback}")

        tscv = TimeSeriesSplit(n_splits=5)

        fold_counter = 0
        for fold, (train_idx, test_idx) in enumerate(tscv.split(df), start=1):
            train_df = df.iloc[train_idx].copy()
            test_df = df.iloc[test_idx].copy()

            required_len = lookback + PREDICTION_LENGTH
            if len(train_df) < required_len:
                print(f"[{scenario_name}] Fold {fold} skipped: train length ({len(train_df)}) < required ({required_len})")
                continue

            try:
                fold_counter += 1
                reset_seeds(SEED)

                train_df = revin_func(train_df)
                test_df = revin_func(test_df)

                training = build_training_dataset_lookback(
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
                    "lookback": lookback,
                    "fold": fold_counter,
                    "train_size": len(train_df),
                    "test_size": len(test_df),
                    "MAE": np.mean(np.abs(true_raw - pred_raw)),
                    "MAPE": mape(true_raw, pred_raw),
                    "sMAPE": smape(true_raw, pred_raw)
                })

                preds_list.append(pd.DataFrame({
                    "scenario": scenario_name,
                    "lookback": lookback,
                    "fold": fold_counter,
                    "date": eval_df["date"].values,
                    "actual": true_raw,
                    "pred": pred_raw,
                    "abs_error": np.abs(true_raw - pred_raw)
                }))

            except Exception as e:
                print(f"[{scenario_name}] Error in lookback={lookback}, fold={fold}: {e}")
                continue

    if len(metrics_list) == 0:
        raise ValueError("No valid results were generated in TFT Round 4.")

    return pd.DataFrame(metrics_list), pd.concat(preds_list, ignore_index=True)

# =========================================================
# 8) RUN ROUND 4
# =========================================================
metrics_a_tft_r4, preds_a_tft_r4 = run_tft_lookback(
    df=df_a,
    feature_cols=feature_cols_a,
    scenario_name="Scenario A",
    revin_func=apply_revin_scenario_a,
    best_params=best_params_a,
    lookback_list=lookback_windows
)

metrics_b_tft_r4, preds_b_tft_r4 = run_tft_lookback(
    df=df_b,
    feature_cols=feature_cols_b,
    scenario_name="Scenario B",
    revin_func=apply_revin_scenario_b,
    best_params=best_params_b,
    lookback_list=lookback_windows
)

# =========================================================
# 9) SUMMARY TABLES
# =========================================================
summary_r4 = pd.concat([metrics_a_tft_r4, metrics_b_tft_r4], ignore_index=True)

summary_mean_r4 = summary_r4.groupby(["scenario", "lookback"])[["MAE", "MAPE", "sMAPE"]].mean().reset_index()
summary_std_r4 = summary_r4.groupby(["scenario", "lookback"])[["MAE", "MAPE", "sMAPE"]].std().reset_index()

print("\n=== TFT Round 4 Mean by Look-back ===")
print(summary_mean_r4)

print("\n=== TFT Round 4 Std by Look-back ===")
print(summary_std_r4)

summary_r4.to_csv("tft_round4_metrics_all.csv", index=False)
summary_mean_r4.to_csv("tft_round4_mean_by_lookback.csv", index=False)
summary_std_r4.to_csv("tft_round4_std_by_lookback.csv", index=False)

# =========================================================
# 10) BEST LOOK-BACK TABLE
# =========================================================
best_lookback_table = summary_mean_r4.loc[
    summary_mean_r4.groupby("scenario")["MAPE"].idxmin()
].reset_index(drop=True)

print("\n=== TFT Round 4 Best Look-back by Scenario ===")
print(best_lookback_table)

best_lookback_table.to_csv("tft_round4_best_lookback_table.csv", index=False)

# =========================================================
# 11) DM TEST BY LOOK-BACK
# =========================================================
dm_results = []

valid_lookbacks = sorted(summary_mean_r4["lookback"].unique().tolist())

for lb in valid_lookbacks:
    tmp_a = preds_a_tft_r4[preds_a_tft_r4["lookback"] == lb].copy()
    tmp_b = preds_b_tft_r4[preds_b_tft_r4["lookback"] == lb].copy()

    dm_df = tmp_a.merge(
        tmp_b,
        on="date",
        suffixes=("_a", "_b")
    ).sort_values("date")

    err_a = dm_df["actual_a"] - dm_df["pred_a"]
    err_b = dm_df["actual_b"] - dm_df["pred_b"]

    dm_stat, p_value = dm_test(err_a.values, err_b.values)

    dm_results.append({
        "lookback": lb,
        "DM_statistic": dm_stat,
        "p_value": p_value
    })

dm_results_df = pd.DataFrame(dm_results)

print("\n=== TFT Round 4 Diebold-Mariano Test by Look-back ===")
print(dm_results_df)

dm_results_df.to_csv("tft_round4_dm_test_by_lookback.csv", index=False)

# =========================================================
# 12) PLOTS
# =========================================================
plt.figure(figsize=(8, 5))
for scenario in summary_mean_r4["scenario"].unique():
    tmp = summary_mean_r4[summary_mean_r4["scenario"] == scenario]
    plt.plot(tmp["lookback"], tmp["MAE"], marker="o", linewidth=2, label=scenario)
plt.title("TFT Round 4: MAE by Look-back Window")
plt.xlabel("Look-back Window (Months)")
plt.ylabel("MAE")
plt.legend()
plt.tight_layout()
plt.savefig("tft_round4_mae_by_lookback.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, 5))
for scenario in summary_mean_r4["scenario"].unique():
    tmp = summary_mean_r4[summary_mean_r4["scenario"] == scenario]
    plt.plot(tmp["lookback"], tmp["MAPE"], marker="o", linewidth=2, label=scenario)
plt.title("TFT Round 4: MAPE by Look-back Window")
plt.xlabel("Look-back Window (Months)")
plt.ylabel("MAPE")
plt.legend()
plt.tight_layout()
plt.savefig("tft_round4_mape_by_lookback.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, 5))
for scenario in summary_mean_r4["scenario"].unique():
    tmp = summary_mean_r4[summary_mean_r4["scenario"] == scenario]
    plt.plot(tmp["lookback"], tmp["sMAPE"], marker="o", linewidth=2, label=scenario)
plt.title("TFT Round 4: sMAPE by Look-back Window")
plt.xlabel("Look-back Window (Months)")
plt.ylabel("sMAPE")
plt.legend()
plt.tight_layout()
plt.savefig("tft_round4_smape_by_lookback.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(9, 5))
for metric in ["MAE", "MAPE", "sMAPE"]:
    pivot_df = summary_mean_r4.pivot(index="lookback", columns="scenario", values=metric)
    norm_df = (pivot_df - pivot_df.min()) / (pivot_df.max() - pivot_df.min() + 1e-12)
    plt.plot(norm_df.index, norm_df["Scenario A"], marker="o", linestyle="-", label=f"{metric} - Scenario A")
    plt.plot(norm_df.index, norm_df["Scenario B"], marker="o", linestyle="--", label=f"{metric} - Scenario B")
plt.title("TFT Round 4: Normalized Forecast Errors by Look-back Window")
plt.xlabel("Look-back Window (Months)")
plt.ylabel("Normalized Error (0-1)")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig("tft_round4_normalized_all_metrics.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, 5))
for scenario in summary_mean_r4["scenario"].unique():
    mean_tmp = summary_mean_r4[summary_mean_r4["scenario"] == scenario]
    std_tmp = summary_std_r4[summary_std_r4["scenario"] == scenario]
    plt.errorbar(
        mean_tmp["lookback"],
        mean_tmp["MAPE"],
        yerr=std_tmp["MAPE"],
        marker="o",
        linewidth=2,
        capsize=5,
        label=scenario
    )
plt.title("TFT Round 4: Mean ± Std of MAPE by Look-back")
plt.xlabel("Look-back Window (Months)")
plt.ylabel("MAPE")
plt.legend()
plt.tight_layout()
plt.savefig("tft_round4_mape_mean_std.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(dm_results_df["lookback"], dm_results_df["p_value"], marker="o", linewidth=2)
plt.axhline(0.05, linestyle="--")
plt.title("TFT Round 4: Diebold-Mariano Test p-value by Look-back")
plt.xlabel("Look-back Window (Months)")
plt.ylabel("p-value")
plt.tight_layout()
plt.savefig("tft_round4_dm_pvalue_by_lookback.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, 5))
plt.boxplot(
    [preds_a_tft_r4["abs_error"], preds_b_tft_r4["abs_error"]],
    tick_labels=["Scenario A", "Scenario B"]
)
plt.title("TFT Round 4: Absolute Error Distribution")
plt.ylabel("Absolute Error")
plt.tight_layout()
plt.savefig("tft_round4_abs_error_boxplot.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nTFT Round 4 finished successfully.")

# ดูว่า best look-back ของแต่ละ scenario คืออะไร
print(best_lookback_table)

# ดึง best look-back ของแต่ละ scenario
best_lb_a = best_lookback_table.loc[
    best_lookback_table["scenario"] == "Scenario A", "lookback"
].iloc[0]

best_lb_b = best_lookback_table.loc[
    best_lookback_table["scenario"] == "Scenario B", "lookback"
].iloc[0]

# ดึงค่าพยากรณ์ RSI ของ best model
best_preds_a = preds_a_tft_r4[preds_a_tft_r4["lookback"] == best_lb_a].copy()
best_preds_b = preds_b_tft_r4[preds_b_tft_r4["lookback"] == best_lb_b].copy()

print("\n=== Best TFT Predictions: Scenario A ===")
print(best_preds_a[["date", "actual", "pred", "abs_error"]])

print("\n=== Best TFT Predictions: Scenario B ===")
print(best_preds_b[["date", "actual", "pred", "abs_error"]])
