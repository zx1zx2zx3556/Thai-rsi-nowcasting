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
# 5) ROUND 3 BEST PARAMS
# ใส่ค่าที่ดีที่สุดจาก Round 3 ของคุณตรงนี้
# =========================================================
best_params_a = {
    "patch_len": 16,
    "stride": 8,
    "hidden_size": 128,
    "n_heads": 4,
    "encoder_layers": 3,
    "learning_rate": 0.0005325732706437209,
    "dropout": 0.05889669436043332,
    "batch_size": 64
}

best_params_b = {
    "patch_len": 8,
    "stride": 8,
    "hidden_size": 64,
    "n_heads": 4,
    "encoder_layers": 2,
    "learning_rate": 0.0008689017469597926,
    "dropout": 0.05596027608265379,
    "batch_size": 32
}

# ถ้าคุณมี best params จริงจาก Round 3 ให้แทนค่า 2 ชุดข้างบนด้วยค่าจริง

# =========================================================
# 6) ROUND 4 SETTINGS
# =========================================================
H = 16
N_WINDOWS = 5
STEP_SIZE = 16
VAL_SIZE = 16
LOOKBACK_LIST = [24, 36, 48, 60]

# =========================================================
# 7) BUILD MODEL
# input_size = lookback
# =========================================================
def build_patchtst_model(best_params, lookback):
    patch_len = min(best_params["patch_len"], lookback)
    stride = min(best_params["stride"], patch_len)

    return PatchTST(
        h=H,
        input_size=lookback,
        encoder_layers=best_params["encoder_layers"],
        n_heads=best_params["n_heads"],
        hidden_size=best_params["hidden_size"],
        linear_hidden_size=best_params["hidden_size"] * 2,
        dropout=best_params["dropout"],
        fc_dropout=best_params["dropout"],
        head_dropout=0.0,
        attn_dropout=0.0,
        patch_len=patch_len,
        stride=stride,
        revin=True,
        revin_affine=False,
        revin_subtract_last=True,
        activation="gelu",
        batch_normalization=False,
        learning_rate=best_params["learning_rate"],
        max_steps=500,
        val_check_steps=50,
        early_stop_patience_steps=3,
        batch_size=best_params["batch_size"],
        start_padding_enabled=True,
        random_seed=SEED
    )

# =========================================================
# 8) RUN LOOK-BACK SEARCH
# =========================================================
def run_patchtst_lookback(panel_df, scenario_name, best_params):
    all_metrics = []
    all_preds = []

    for lookback in LOOKBACK_LIST:
        reset_seeds(SEED)

        model = build_patchtst_model(best_params, lookback)
        nf = NeuralForecast(models=[model], freq="MS")

        cv_df = nf.cross_validation(
            df=panel_df,
            n_windows=N_WINDOWS,
            step_size=STEP_SIZE,
            val_size=VAL_SIZE,
            refit=True,
            verbose=0
        )

        pred_col = [c for c in cv_df.columns if c not in ["unique_id", "ds", "cutoff", "y"]][0]

        rsi_df = cv_df[cv_df["unique_id"] == "RSI"].copy().sort_values(["cutoff", "ds"])
        rsi_df.rename(columns={"ds": "date", "y": "actual", pred_col: "pred"}, inplace=True)
        rsi_df["scenario"] = scenario_name
        rsi_df["lookback"] = lookback
        rsi_df["abs_error"] = np.abs(rsi_df["actual"] - rsi_df["pred"])

        cutoff_order = {c: i+1 for i, c in enumerate(sorted(rsi_df["cutoff"].unique()))}
        rsi_df["fold"] = rsi_df["cutoff"].map(cutoff_order)

        metrics_by_fold = []
        for fold in sorted(rsi_df["fold"].unique()):
            tmp = rsi_df[rsi_df["fold"] == fold].copy()

            metrics_by_fold.append({
                "scenario": scenario_name,
                "lookback": lookback,
                "fold": fold,
                "test_size": len(tmp),
                "MAE": mean_absolute_error(tmp["actual"], tmp["pred"]),
                "MAPE": mape(tmp["actual"], tmp["pred"]),
                "sMAPE": smape(tmp["actual"], tmp["pred"])
            })

        all_metrics.append(pd.DataFrame(metrics_by_fold))
        all_preds.append(rsi_df)

    return pd.concat(all_metrics, ignore_index=True), pd.concat(all_preds, ignore_index=True)

metrics_a_patch_r4, preds_a_patch_r4 = run_patchtst_lookback(
    panel_df=panel_a,
    scenario_name="Scenario A",
    best_params=best_params_a
)

metrics_b_patch_r4, preds_b_patch_r4 = run_patchtst_lookback(
    panel_df=panel_b,
    scenario_name="Scenario B",
    best_params=best_params_b
)

# =========================================================
# 9) SUMMARY
# =========================================================
summary_r4 = pd.concat([metrics_a_patch_r4, metrics_b_patch_r4], ignore_index=True)

summary_mean_r4 = (
    summary_r4.groupby(["scenario", "lookback"])[["MAE", "MAPE", "sMAPE"]]
    .mean()
    .reset_index()
)

summary_std_r4 = (
    summary_r4.groupby(["scenario", "lookback"])[["MAE", "MAPE", "sMAPE"]]
    .std()
    .reset_index()
)

print("\n=== PatchTST Round 4 Mean by Look-back ===")
print(summary_mean_r4)

print("\n=== PatchTST Round 4 Std by Look-back ===")
print(summary_std_r4)

# =========================================================
# 10) BEST LOOKBACK TABLE
# ใช้ MAPE ต่ำสุดเป็นเกณฑ์หลัก
# =========================================================
best_lookback_table = summary_mean_r4.loc[
    summary_mean_r4.groupby("scenario")["MAPE"].idxmin()
].reset_index(drop=True)

print("\n=== PatchTST Round 4 Best Look-back by Scenario ===")
print(best_lookback_table)

# =========================================================
# 11) DM TEST BY LOOKBACK
# =========================================================
dm_results = []

for lb in LOOKBACK_LIST:
    tmp_a = preds_a_patch_r4[preds_a_patch_r4["lookback"] == lb].copy()
    tmp_b = preds_b_patch_r4[preds_b_patch_r4["lookback"] == lb].copy()

    dm_df = tmp_a.merge(
        tmp_b,
        on=["date", "fold"],
        suffixes=("_a", "_b")
    ).sort_values(["fold", "date"])

    err_a = dm_df["actual_a"] - dm_df["pred_a"]
    err_b = dm_df["actual_b"] - dm_df["pred_b"]

    dm_stat, p_value = dm_test(err_a.values, err_b.values)

    dm_results.append({
        "lookback": lb,
        "DM_statistic": dm_stat,
        "p_value": p_value
    })

dm_results_df = pd.DataFrame(dm_results)

print("\n=== PatchTST Round 4 DM Test by Look-back ===")
print(dm_results_df)

# =========================================================
# 12) SAVE
# =========================================================
summary_r4.to_csv("patchtst_round4_metrics_all.csv", index=False)
summary_mean_r4.to_csv("patchtst_round4_mean_by_lookback.csv", index=False)
summary_std_r4.to_csv("patchtst_round4_std_by_lookback.csv", index=False)
best_lookback_table.to_csv("patchtst_round4_best_lookback_table.csv", index=False)
dm_results_df.to_csv("patchtst_round4_dm_test_by_lookback.csv", index=False)

preds_a_patch_r4.to_csv("patchtst_round4_predictions_scenario_a_rsi.csv", index=False)
preds_b_patch_r4.to_csv("patchtst_round4_predictions_scenario_b_rsi.csv", index=False)

# =========================================================
# 13) PLOTS
# =========================================================

# 13.1 MAE by lookback
plt.figure(figsize=(8, 5))
for scenario in summary_mean_r4["scenario"].unique():
    tmp = summary_mean_r4[summary_mean_r4["scenario"] == scenario]
    plt.plot(tmp["lookback"], tmp["MAE"], marker="o", linewidth=2, label=scenario)

plt.title("PatchTST Round 4: MAE by Look-back Window")
plt.xlabel("Look-back Window (Months)")
plt.ylabel("MAE")
plt.legend()
plt.tight_layout()
plt.savefig("patchtst_round4_mae_by_lookback.png", dpi=300, bbox_inches="tight")
plt.show()

# 13.2 MAPE by lookback
plt.figure(figsize=(8, 5))
for scenario in summary_mean_r4["scenario"].unique():
    tmp = summary_mean_r4[summary_mean_r4["scenario"] == scenario]
    plt.plot(tmp["lookback"], tmp["MAPE"], marker="o", linewidth=2, label=scenario)

plt.title("PatchTST Round 4: MAPE by Look-back Window")
plt.xlabel("Look-back Window (Months)")
plt.ylabel("MAPE")
plt.legend()
plt.tight_layout()
plt.savefig("patchtst_round4_mape_by_lookback.png", dpi=300, bbox_inches="tight")
plt.show()

# 13.3 sMAPE by lookback
plt.figure(figsize=(8, 5))
for scenario in summary_mean_r4["scenario"].unique():
    tmp = summary_mean_r4[summary_mean_r4["scenario"] == scenario]
    plt.plot(tmp["lookback"], tmp["sMAPE"], marker="o", linewidth=2, label=scenario)

plt.title("PatchTST Round 4: sMAPE by Look-back Window")
plt.xlabel("Look-back Window (Months)")
plt.ylabel("sMAPE")
plt.legend()
plt.tight_layout()
plt.savefig("patchtst_round4_smape_by_lookback.png", dpi=300, bbox_inches="tight")
plt.show()

# 13.4 Combined normalized comparison
plt.figure(figsize=(9, 5))
for metric in ["MAE", "MAPE", "sMAPE"]:
    pivot_df = summary_mean_r4.pivot(index="lookback", columns="scenario", values=metric)
    norm_df = (pivot_df - pivot_df.min()) / (pivot_df.max() - pivot_df.min() + 1e-12)

    plt.plot(norm_df.index, norm_df["Scenario A"], marker="o", linestyle="-", label=f"{metric} - Scenario A")
    plt.plot(norm_df.index, norm_df["Scenario B"], marker="o", linestyle="--", label=f"{metric} - Scenario B")

plt.title("PatchTST Round 4: Normalized Forecast Errors by Look-back")
plt.xlabel("Look-back Window (Months)")
plt.ylabel("Normalized Error (0-1)")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig("patchtst_round4_normalized_all_metrics.png", dpi=300, bbox_inches="tight")
plt.show()

# 13.5 Mean ± std of MAPE
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

plt.title("PatchTST Round 4: Mean ± Std of MAPE by Look-back")
plt.xlabel("Look-back Window (Months)")
plt.ylabel("MAPE")
plt.legend()
plt.tight_layout()
plt.savefig("patchtst_round4_mape_mean_std.png", dpi=300, bbox_inches="tight")
plt.show()

# 13.6 DM p-value by lookback
plt.figure(figsize=(8, 5))
plt.plot(dm_results_df["lookback"], dm_results_df["p_value"], marker="o", linewidth=2)
plt.axhline(0.05, linestyle="--")
plt.title("PatchTST Round 4: Diebold-Mariano Test p-value by Look-back")
plt.xlabel("Look-back Window (Months)")
plt.ylabel("p-value")
plt.tight_layout()
plt.savefig("patchtst_round4_dm_pvalue_by_lookback.png", dpi=300, bbox_inches="tight")
plt.show()

# 13.7 Absolute error distribution
plt.figure(figsize=(8, 5))
plt.boxplot(
    [preds_a_patch_r4["abs_error"], preds_b_patch_r4["abs_error"]],
    tick_labels=["Scenario A", "Scenario B"]
)
plt.title("PatchTST Round 4: Absolute Error Distribution")
plt.ylabel("Absolute Error")
plt.tight_layout()
plt.savefig("patchtst_round4_abs_error_boxplot.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nPatchTST Round 4 finished successfully.")