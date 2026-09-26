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

    if len(d) < 2 or var_d == 0 or np.isnan(var_d):
        return np.nan, np.nan

    dm_stat = mean_d / np.sqrt(var_d / len(d))
    p_value = 2 * (1 - t.cdf(np.abs(dm_stat), df=len(d)-1))
    return dm_stat, p_value

# =========================================================
# 3) DEFINE CHANNELS
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
# 4) WIDE -> PANEL
# each variable = one channel
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
# 5) PATCHTST SETTINGS
# Round 2 = 16-month test window, 5 windows
# =========================================================
H = 16
INPUT_SIZE = 24
N_WINDOWS = 5
STEP_SIZE = 16   # non-overlapping windows
VAL_SIZE = 16    # must be >= h for NeuralForecast temporal CV

def build_patchtst_model():
    return PatchTST(
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
        start_padding_enabled=True,
        random_seed=SEED
    )

# =========================================================
# 6) RUN PATCHTST ROUND 2 WITH OFFICIAL CROSS_VALIDATION
# =========================================================
def run_patchtst_cv(panel_df, scenario_name):
    reset_seeds(SEED)

    model = build_patchtst_model()
    nf = NeuralForecast(models=[model], freq="MS")

    # official temporal cross-validation
    cv_df = nf.cross_validation(
        df=panel_df,
        n_windows=N_WINDOWS,
        step_size=STEP_SIZE,
        val_size=VAL_SIZE,
        refit=True,
        verbose=0
    )

    pred_col = [c for c in cv_df.columns if c not in ["unique_id", "ds", "cutoff", "y"]][0]

    # keep only RSI channel for evaluation
    rsi_df = cv_df[cv_df["unique_id"] == "RSI"].copy().sort_values(["cutoff", "ds"])
    rsi_df.rename(columns={"ds": "date", "y": "actual", pred_col: "pred"}, inplace=True)
    rsi_df["scenario"] = scenario_name
    rsi_df["abs_error"] = np.abs(rsi_df["actual"] - rsi_df["pred"])

    # create fold id from cutoff order
    cutoff_order = {c: i+1 for i, c in enumerate(sorted(rsi_df["cutoff"].unique()))}
    rsi_df["fold"] = rsi_df["cutoff"].map(cutoff_order)

    metrics_list = []
    for fold in sorted(rsi_df["fold"].unique()):
        tmp = rsi_df[rsi_df["fold"] == fold].copy()

        metrics_list.append({
            "scenario": scenario_name,
            "fold": fold,
            "test_size": len(tmp),
            "MAE": mean_absolute_error(tmp["actual"], tmp["pred"]),
            "MAPE": mape(tmp["actual"], tmp["pred"]),
            "sMAPE": smape(tmp["actual"], tmp["pred"])
        })

    metrics_df = pd.DataFrame(metrics_list)
    return metrics_df, rsi_df, cv_df

metrics_a_patch_r2, preds_a_patch_r2, preds_all_a_patch_r2 = run_patchtst_cv(
    panel_df=panel_a,
    scenario_name="Scenario A"
)

metrics_b_patch_r2, preds_b_patch_r2, preds_all_b_patch_r2 = run_patchtst_cv(
    panel_df=panel_b,
    scenario_name="Scenario B"
)

# =========================================================
# 7) SUMMARY
# =========================================================
summary_r2 = pd.concat([metrics_a_patch_r2, metrics_b_patch_r2], ignore_index=True)

summary_mean_r2 = summary_r2.groupby("scenario")[["MAE", "MAPE", "sMAPE"]].mean()
summary_std_r2 = summary_r2.groupby("scenario")[["MAE", "MAPE", "sMAPE"]].std()

print("\n=== PatchTST Round 2 Metrics by Fold ===")
print(summary_r2)

print("\n=== PatchTST Round 2 Mean ===")
print(summary_mean_r2)

print("\n=== PatchTST Round 2 Std ===")
print(summary_std_r2)

# =========================================================
# 8) DIEBOLD-MARIANO TEST
# align by date + fold
# =========================================================
dm_df = preds_a_patch_r2.merge(
    preds_b_patch_r2,
    on=["date", "fold"],
    suffixes=("_a", "_b")
).sort_values(["fold", "date"])

err_a = dm_df["actual_a"] - dm_df["pred_a"]
err_b = dm_df["actual_b"] - dm_df["pred_b"]

dm_stat, p_value = dm_test(err_a.values, err_b.values)

print("\n=== PatchTST Round 2 DM Test ===")
print("DM statistic:", dm_stat)
print("p-value:", p_value)

# =========================================================
# 9) SAVE
# =========================================================
summary_r2.to_csv("patchtst_round2_metrics_all.csv", index=False)
summary_mean_r2.to_csv("patchtst_round2_mean.csv")
summary_std_r2.to_csv("patchtst_round2_std.csv")
preds_a_patch_r2.to_csv("patchtst_round2_predictions_scenario_a_rsi.csv", index=False)
preds_b_patch_r2.to_csv("patchtst_round2_predictions_scenario_b_rsi.csv", index=False)
preds_all_a_patch_r2.to_csv("patchtst_round2_predictions_scenario_a_all_channels.csv", index=False)
preds_all_b_patch_r2.to_csv("patchtst_round2_predictions_scenario_b_all_channels.csv", index=False)

# =========================================================
# 10) PLOTS
# =========================================================
def plot_metric_by_fold(metrics_df, metric_name):
    plt.figure(figsize=(8, 5))
    for scenario in metrics_df["scenario"].unique():
        tmp = metrics_df[metrics_df["scenario"] == scenario].copy()
        plt.plot(tmp["fold"], tmp[metric_name], marker="o", linewidth=2, label=scenario)
    plt.title(f"PatchTST Round 2: {metric_name} by Fold")
    plt.xlabel("Fold")
    plt.ylabel(metric_name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"patchtst_round2_{metric_name.lower()}_by_fold.png", dpi=300, bbox_inches="tight")
    plt.show()

plot_metric_by_fold(summary_r2, "MAE")
plot_metric_by_fold(summary_r2, "MAPE")
plot_metric_by_fold(summary_r2, "sMAPE")

mean_a = metrics_a_patch_r2[["MAE", "MAPE", "sMAPE"]].mean()
mean_b = metrics_b_patch_r2[["MAE", "MAPE", "sMAPE"]].mean()

metric_names = ["MAE", "MAPE", "sMAPE"]
x = np.arange(len(metric_names))

scenario_a_mean = [mean_a["MAE"], mean_a["MAPE"], mean_a["sMAPE"]]
scenario_b_mean = [mean_b["MAE"], mean_b["MAPE"], mean_b["sMAPE"]]

plt.figure(figsize=(8, 5))
plt.plot(x, scenario_a_mean, marker="o", linewidth=2, label="Scenario A")
plt.plot(x, scenario_b_mean, marker="o", linewidth=2, label="Scenario B")
plt.xticks(x, metric_names)
plt.ylabel("Mean Error Value")
plt.title("PatchTST Round 2: Mean Forecasting Error Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("patchtst_round2_mean_metrics_line.png", dpi=300, bbox_inches="tight")
plt.show()

std_a = metrics_a_patch_r2[["MAE", "MAPE", "sMAPE"]].std()
std_b = metrics_b_patch_r2[["MAE", "MAPE", "sMAPE"]].std()

scenario_a_std = [std_a["MAE"], std_a["MAPE"], std_a["sMAPE"]]
scenario_b_std = [std_b["MAE"], std_b["MAPE"], std_b["sMAPE"]]

plt.figure(figsize=(8, 5))
plt.errorbar(x, scenario_a_mean, yerr=scenario_a_std, marker="o", linewidth=2, capsize=5, label="Scenario A")
plt.errorbar(x, scenario_b_mean, yerr=scenario_b_std, marker="o", linewidth=2, capsize=5, label="Scenario B")
plt.xticks(x, metric_names)
plt.ylabel("Mean Error Value")
plt.title("PatchTST Round 2: Mean ± Standard Deviation of Forecast Errors")
plt.legend()
plt.tight_layout()
plt.savefig("patchtst_round2_mean_std_metrics_line.png", dpi=300, bbox_inches="tight")
plt.show()

def plot_actual_vs_pred_average(pred_df, scenario_name):
    tmp = pred_df.copy()
    avg_df = tmp.groupby("date")[["actual", "pred"]].mean().reset_index()
    plt.figure(figsize=(10, 5))
    plt.plot(avg_df["date"], avg_df["actual"], linewidth=2, label="Actual RSI")
    plt.plot(avg_df["date"], avg_df["pred"], linewidth=2, label="Predicted RSI")
    plt.title(f"PatchTST Round 2: {scenario_name} Actual vs Predicted (Average by Date)")
    plt.xlabel("Date")
    plt.ylabel("RSI")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(
        f"patchtst_round2_actual_vs_pred_avg_{scenario_name.lower().replace(' ', '_')}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

plot_actual_vs_pred_average(preds_a_patch_r2, "Scenario A")
plot_actual_vs_pred_average(preds_b_patch_r2, "Scenario B")

plt.figure(figsize=(8, 5))
plt.boxplot(
    [preds_a_patch_r2["abs_error"], preds_b_patch_r2["abs_error"]],
    tick_labels=["Scenario A", "Scenario B"]
)
plt.title("PatchTST Round 2: Absolute Error Distribution (RSI)")
plt.ylabel("Absolute Error")
plt.tight_layout()
plt.savefig("patchtst_round2_abs_error_boxplot.png", dpi=300, bbox_inches="tight")
plt.show()

loss_diff = err_a**2 - err_b**2
cum_diff = np.cumsum(loss_diff)

plt.figure(figsize=(10, 5))
plt.plot(dm_df["date"], loss_diff, marker="o")
plt.axhline(0, linestyle="--")
plt.title("PatchTST Round 2: Loss Differential (Scenario A vs Scenario B)")
plt.xlabel("Date")
plt.ylabel("Squared Error Difference")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("patchtst_round2_loss_differential.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(dm_df["date"], cum_diff)
plt.axhline(0, linestyle="--")
plt.title("PatchTST Round 2: Cumulative Loss Differential")
plt.xlabel("Date")
plt.ylabel("Cumulative Difference")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("patchtst_round2_cumulative_loss_differential.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nPatchTST Round 2 finished successfully.")