import os
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error
from scipy.stats import t

from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST

warnings.filterwarnings("ignore")

# =========================================================
# 0) REPRODUCIBILITY
# =========================================================
SEED = 42

def reset_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)

reset_seeds(SEED)
print(f"Seed set to {SEED}")

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

    if var_d == 0 or np.isnan(var_d):
        return np.nan, np.nan

    dm_stat = mean_d / np.sqrt(var_d / len(d))
    p_value = 2 * (1 - t.cdf(np.abs(dm_stat), df=len(d)-1))
    return dm_stat, p_value

# =========================================================
# 3) DEFINE CHANNELS FOR EACH SCENARIO
# ใช้ multivariate-channel style
# =========================================================
scenario_a_channels = [
    "RSI",
    "PCI_lag1", "PCI_lag2", "PCI_lag3",
    "CPI_lag1", "CPI_lag2", "CPI_lag3",
    "Discount_rm3_lag1", "Discount_rm3_lag2", "Discount_rm3_lag3",
    "Promotion_rm3_lag1", "Promotion_rm3_lag2", "Promotion_rm3_lag3",
    "OnlineBuy_rm3_lag1", "OnlineBuy_rm3_lag2", "OnlineBuy_rm3_lag3",
    "Coupon_rm3_lag1", "Coupon_rm3_lag2", "Coupon_rm3_lag3",
    "Gift_rm3_lag1", "Gift_rm3_lag2", "Gift_rm3_lag3"
]

scenario_b_channels = [
    "RSI",
    "PCI_lag1", "PCI_lag2", "PCI_lag3",
    "CPI_lag1", "CPI_lag2", "CPI_lag3",
    "Discount_lag4", "Discount_lag8", "Discount_lag12",
    "Promotion_lag4", "Promotion_lag8", "Promotion_lag12",
    "OnlineBuy_lag4", "OnlineBuy_lag8", "OnlineBuy_lag12",
    "Coupon_lag4", "Coupon_lag8", "Coupon_lag12",
    "Gift_lag4", "Gift_lag8", "Gift_lag12"
]

# =========================================================
# 4) PREPARE PANEL DATA FOR PATCHTST
# แปลง wide -> long โดยให้แต่ละตัวแปรเป็น 1 channel
# =========================================================
def prepare_patchtst_panel(df, channels):
    panel_list = []

    for col in channels:
        tmp = df[["date", col]].copy()
        tmp["unique_id"] = col
        tmp["ds"] = pd.to_datetime(tmp["date"])
        tmp["y"] = tmp[col]
        tmp = tmp[["unique_id", "ds", "y"]]
        panel_list.append(tmp)

    panel_df = pd.concat(panel_list, ignore_index=True)
    panel_df = panel_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    return panel_df

panel_a = prepare_patchtst_panel(df_a, scenario_a_channels)
panel_b = prepare_patchtst_panel(df_b, scenario_b_channels)

# =========================================================
# 5) TRAIN / TEST SPLIT
# ใช้ holdout ช่วงสุดท้าย 16 เดือน
# =========================================================
H = 16
INPUT_SIZE = 24

def split_panel(panel_df, horizon=16):
    train_parts = []
    test_parts = []

    for uid in panel_df["unique_id"].unique():
        tmp = panel_df[panel_df["unique_id"] == uid].copy().sort_values("ds")
        train_parts.append(tmp.iloc[:-horizon])
        test_parts.append(tmp.iloc[-horizon:])

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    return train_df, test_df

train_a, test_a = split_panel(panel_a, H)
train_b, test_b = split_panel(panel_b, H)

# =========================================================
# 6) BUILD BASELINE PATCHTST MODEL
# ใช้ multivariate panel แต่ไม่ส่ง exogenous list
# =========================================================
def build_patchtst_model():
    model = PatchTST(
        h=H,
        input_size=INPUT_SIZE,
        encoder_layers=3,
        n_heads=4,
        hidden_size=64,
        linear_hidden_size=128,
        dropout=0.2,
        fc_dropout=0.2,
        head_dropout=0.0,
        attn_dropout=0.0,
        patch_len=8,
        stride=4,
        revin=True,
        revin_affine=False,
        revin_subtract_last=True,
        activation="gelu",
        batch_normalization=False,
        learning_rate=1e-3,
        max_steps=500,
        val_check_steps=50,
        early_stop_patience_steps=3,
        batch_size=32,
        random_seed=SEED
    )
    return model

# =========================================================
# 7) FIT + PREDICT
# =========================================================
def fit_predict_patchtst(train_df, test_df, scenario_name):
    reset_seeds(SEED)

    model = build_patchtst_model()
    nf = NeuralForecast(models=[model], freq="MS")

    nf.fit(df=train_df, val_size=H)
    forecasts = nf.predict(futr_df=test_df)

    pred_col = [c for c in forecasts.columns if c not in ["unique_id", "ds"]][0]

    result = test_df[["unique_id", "ds", "y"]].copy()
    result["pred"] = forecasts[pred_col].values
    result["scenario"] = scenario_name
    result.rename(columns={"y": "actual", "ds": "date"}, inplace=True)
    result["abs_error"] = np.abs(result["actual"] - result["pred"])

    # evaluate only RSI channel
    rsi_result = result[result["unique_id"] == "RSI"].copy().sort_values("date")

    metrics = {
        "scenario": scenario_name,
        "MAE": mean_absolute_error(rsi_result["actual"], rsi_result["pred"]),
        "MAPE": mape(rsi_result["actual"], rsi_result["pred"]),
        "sMAPE": smape(rsi_result["actual"], rsi_result["pred"])
    }

    return pd.DataFrame([metrics]), rsi_result, result

metrics_a_patch_r1, preds_a_patch_r1, preds_all_a_patch_r1 = fit_predict_patchtst(
    train_df=train_a,
    test_df=test_a,
    scenario_name="Scenario A"
)

metrics_b_patch_r1, preds_b_patch_r1, preds_all_b_patch_r1 = fit_predict_patchtst(
    train_df=train_b,
    test_df=test_b,
    scenario_name="Scenario B"
)

# =========================================================
# 8) SUMMARY
# =========================================================
summary_r1 = pd.concat([metrics_a_patch_r1, metrics_b_patch_r1], ignore_index=True)

print("\n=== PatchTST Round 1 Mean ===")
print(summary_r1)

# =========================================================
# 9) DIEBOLD-MARIANO TEST
# compare only RSI channel
# =========================================================
dm_df = preds_a_patch_r1.merge(
    preds_b_patch_r1,
    on="date",
    suffixes=("_a", "_b")
).sort_values("date")

err_a = dm_df["actual_a"] - dm_df["pred_a"]
err_b = dm_df["actual_b"] - dm_df["pred_b"]

dm_stat, p_value = dm_test(err_a.values, err_b.values)

print("\n=== PatchTST Round 1 DM Test ===")
print("DM statistic:", dm_stat)
print("p-value:", p_value)

# =========================================================
# 10) SAVE
# =========================================================
summary_r1.to_csv("patchtst_round1_metrics_all.csv", index=False)
preds_a_patch_r1.to_csv("patchtst_round1_predictions_scenario_a_rsi.csv", index=False)
preds_b_patch_r1.to_csv("patchtst_round1_predictions_scenario_b_rsi.csv", index=False)
preds_all_a_patch_r1.to_csv("patchtst_round1_predictions_scenario_a_all_channels.csv", index=False)
preds_all_b_patch_r1.to_csv("patchtst_round1_predictions_scenario_b_all_channels.csv", index=False)

# =========================================================
# 11) PLOTS
# =========================================================

# 11.1 Bar chart of metrics
metric_names = ["MAE", "MAPE", "sMAPE"]
a_vals = metrics_a_patch_r1.loc[0, metric_names].values.astype(float)
b_vals = metrics_b_patch_r1.loc[0, metric_names].values.astype(float)

x = np.arange(len(metric_names))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - width/2, a_vals, width=width, label="Scenario A")
plt.bar(x + width/2, b_vals, width=width, label="Scenario B")
plt.xticks(x, metric_names)
plt.ylabel("Error Value")
plt.title("PatchTST Round 1: Forecast Error Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("patchtst_round1_bar_metrics.png", dpi=300, bbox_inches="tight")
plt.show()

# 11.2 Actual vs Predicted (RSI only)
def plot_actual_vs_pred(pred_df, scenario_name):
    tmp = pred_df.copy().sort_values("date")

    plt.figure(figsize=(10, 5))
    plt.plot(tmp["date"], tmp["actual"], linewidth=2, label="Actual RSI")
    plt.plot(tmp["date"], tmp["pred"], linewidth=2, label="Predicted RSI")
    plt.title(f"PatchTST Round 1: {scenario_name} Actual vs Predicted (RSI)")
    plt.xlabel("Date")
    plt.ylabel("RSI")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(
        f"patchtst_round1_actual_vs_pred_{scenario_name.lower().replace(' ', '_')}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

plot_actual_vs_pred(preds_a_patch_r1, "Scenario A")
plot_actual_vs_pred(preds_b_patch_r1, "Scenario B")

# 11.3 Absolute error boxplot
plt.figure(figsize=(8, 5))
plt.boxplot(
    [preds_a_patch_r1["abs_error"], preds_b_patch_r1["abs_error"]],
    tick_labels=["Scenario A", "Scenario B"]
)
plt.title("PatchTST Round 1: Absolute Error Distribution (RSI)")
plt.ylabel("Absolute Error")
plt.tight_layout()
plt.savefig("patchtst_round1_abs_error_boxplot.png", dpi=300, bbox_inches="tight")
plt.show()

# 11.4 Loss differential
loss_diff = err_a**2 - err_b**2
cum_diff = np.cumsum(loss_diff)

plt.figure(figsize=(10, 5))
plt.plot(dm_df["date"], loss_diff, marker="o")
plt.axhline(0, linestyle="--")
plt.title("PatchTST Round 1: Loss Differential (Scenario A vs Scenario B)")
plt.xlabel("Date")
plt.ylabel("Squared Error Difference")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("patchtst_round1_loss_differential.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(dm_df["date"], cum_diff)
plt.axhline(0, linestyle="--")
plt.title("PatchTST Round 1: Cumulative Loss Differential")
plt.xlabel("Date")
plt.ylabel("Cumulative Difference")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("patchtst_round1_cumulative_loss_differential.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nPatchTST Round 1 finished successfully.")