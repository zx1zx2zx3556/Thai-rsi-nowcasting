#########################################################################################   Round 2 #################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from scipy.stats import t

# =========================================================
# 1) LOAD DATA
# =========================================================
df_a = pd.read_csv(
    '/Users/methask./Retail_sales 2015 - 2023/scenario_a_full_df.csv',
    parse_dates=["date"]
)
df_b = pd.read_csv(
    '/Users/methask./Retail_sales 2015 - 2023/scenario_b_full_df.csv',
    parse_dates=["date"]
)

df_a = df_a.sort_values("date").reset_index(drop=True)
df_b = df_b.sort_values("date").reset_index(drop=True)

print("Scenario A shape:", df_a.shape)
print("Scenario B shape:", df_b.shape)


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


# =========================================================
# 4) APPLY RevIN PER SCENARIO
# =========================================================
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
# 6) RUN 5-FOLD TIME SERIES CV
# =========================================================
def run_time_series_cv(df, feature_cols, scenario_name, revin_func, alpha=1.0, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)

    metrics = []
    pred_list = []
    coef_list = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(df), start=1):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()

        # RevIN per fold
        train_df = revin_func(train_df)
        test_df = revin_func(test_df)

        X_train = train_df[feature_cols]
        y_train = train_df["target_norm"]

        X_test = test_df[feature_cols]
        y_test_raw = test_df["target_raw"].values

        # Stable linear model
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)

        y_pred_norm = model.predict(X_test)

        # inverse RevIN
        y_pred_raw = (
            y_pred_norm * test_df["target_sd"].values
            + test_df["target_mean"].values
        )

        mae_val = mean_absolute_error(y_test_raw, y_pred_raw)
        mape_val = mape(y_test_raw, y_pred_raw)
        smape_val = smape(y_test_raw, y_pred_raw)

        metrics.append({
            "scenario": scenario_name,
            "fold": fold,
            "train_size": len(train_df),
            "test_size": len(test_df),
            "MAE": mae_val,
            "MAPE": mape_val,
            "sMAPE": smape_val
        })

        pred_df = pd.DataFrame({
            "scenario": scenario_name,
            "fold": fold,
            "date": test_df["date"].values,
            "actual": y_test_raw,
            "pred": y_pred_raw,
            "abs_error": np.abs(y_test_raw - y_pred_raw)
        })
        pred_list.append(pred_df)

        coef_df = pd.DataFrame({
            "scenario": scenario_name,
            "fold": fold,
            "feature": feature_cols,
            "coefficient": model.coef_
        })
        coef_list.append(coef_df)

    metrics_df = pd.DataFrame(metrics)
    preds_df = pd.concat(pred_list, ignore_index=True)
    coef_df = pd.concat(coef_list, ignore_index=True)

    return metrics_df, preds_df, coef_df


# =========================================================
# 7) RUN ROUND 2
# =========================================================
metrics_a_cv, preds_a_cv, coef_a = run_time_series_cv(
    df=df_a,
    feature_cols=feature_cols_a,
    scenario_name="Scenario A",
    revin_func=apply_revin_scenario_a,
    alpha=1.0,
    n_splits=5
)

metrics_b_cv, preds_b_cv, coef_b = run_time_series_cv(
    df=df_b,
    feature_cols=feature_cols_b,
    scenario_name="Scenario B",
    revin_func=apply_revin_scenario_b,
    alpha=1.0,
    n_splits=5
)

summary_cv = pd.concat([metrics_a_cv, metrics_b_cv], ignore_index=True)
summary_mean = summary_cv.groupby("scenario")[["MAE", "MAPE", "sMAPE"]].mean()
summary_std = summary_cv.groupby("scenario")[["MAE", "MAPE", "sMAPE"]].std()

print("=== Round 2: 5-Fold Time Series Cross-Validation ===")
print(summary_cv)
print("\n=== Mean ===")
print(summary_mean)
print("\n=== Std ===")
print(summary_std)


# =========================================================
# 8) DIEBOLD–MARIANO TEST
# =========================================================
def dm_test(e1, e2):
    d = e1**2 - e2**2
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)
    dm_stat = mean_d / np.sqrt(var_d / len(d))
    p_value = 2 * (1 - t.cdf(np.abs(dm_stat), df=len(d)-1))
    return dm_stat, p_value

dm_df = preds_a_cv.merge(
    preds_b_cv,
    on="date",
    suffixes=("_a", "_b")
).sort_values("date")

err_a = dm_df["actual_a"] - dm_df["pred_a"]
err_b = dm_df["actual_b"] - dm_df["pred_b"]

dm_stat, p_value = dm_test(err_a.values, err_b.values)

print("\n=== Diebold-Mariano Test (Round 2) ===")
print("DM statistic:", dm_stat)
if p_value < 0.001:
    print("p-value: < 0.001")
else:
    print("p-value:", p_value)


# =========================================================
# 9) SAVE RESULTS
# =========================================================
metrics_a_cv.to_csv("round2_metrics_scenario_a.csv", index=False)
metrics_b_cv.to_csv("round2_metrics_scenario_b.csv", index=False)
preds_a_cv.to_csv("round2_predictions_scenario_a.csv", index=False)
preds_b_cv.to_csv("round2_predictions_scenario_b.csv", index=False)
summary_cv.to_csv("round2_metrics_all.csv", index=False)
summary_mean.to_csv("round2_summary_mean.csv")
summary_std.to_csv("round2_summary_std.csv")
coef_a.to_csv("round2_coefficients_scenario_a.csv", index=False)
coef_b.to_csv("round2_coefficients_scenario_b.csv", index=False)


# =========================================================
# 10) PLOTS
# =========================================================

# 10.1 Metrics by fold
def plot_metric_by_fold(metrics_df, metric_name):
    plt.figure(figsize=(8, 5))

    for scenario in metrics_df["scenario"].unique():
        tmp = metrics_df[metrics_df["scenario"] == scenario]
        plt.plot(tmp["fold"], tmp[metric_name], marker="o", label=scenario)

    plt.title(f"{metric_name} by Fold (Round 2)")
    plt.xlabel("Fold")
    plt.ylabel(metric_name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"round2_{metric_name.lower()}_by_fold.png", dpi=300, bbox_inches="tight")
    plt.show()

plot_metric_by_fold(summary_cv, "MAE")
plot_metric_by_fold(summary_cv, "MAPE")
plot_metric_by_fold(summary_cv, "sMAPE")


# 10.2 Combined bar chart (normalized for same chart)
mean_a = metrics_a_cv[["MAE", "MAPE", "sMAPE"]].mean()
mean_b = metrics_b_cv[["MAE", "MAPE", "sMAPE"]].mean()

metrics = ["MAE", "MAPE", "sMAPE"]
vals_a = np.array([mean_a["MAE"], mean_a["MAPE"], mean_a["sMAPE"]], dtype=float)
vals_b = np.array([mean_b["MAE"], mean_b["MAPE"], mean_b["sMAPE"]], dtype=float)

all_vals = np.vstack([vals_a, vals_b])
norm_vals = (all_vals - all_vals.min(axis=0)) / (all_vals.max(axis=0) - all_vals.min(axis=0) + 1e-12)

norm_a = norm_vals[0]
norm_b = norm_vals[1]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(8, 5))
bars1 = plt.bar(x - width/2, norm_a, width, label="Scenario A")
bars2 = plt.bar(x + width/2, norm_b, width, label="Scenario B")

plt.xticks(x, metrics)
plt.ylabel("Normalized Error (0–1)")
plt.title("Combined Forecast Error Comparison (Round 2)")
plt.legend()

for bar in list(bars1) + list(bars2):
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, h, f"{h:.2f}", ha="center", va="bottom")

plt.tight_layout()
plt.savefig("round2_combined_normalized_metrics.png", dpi=300, bbox_inches="tight")
plt.show()


# 10.3 Actual vs Predicted
def plot_actual_vs_pred(pred_df, scenario_name):
    tmp = pred_df[pred_df["scenario"] == scenario_name].copy()

    plt.figure(figsize=(10, 5))
    plt.plot(tmp["date"], tmp["actual"], label="Actual RSI")
    plt.plot(tmp["date"], tmp["pred"], label="Predicted RSI")
    plt.title(f"{scenario_name}: Actual vs Predicted (Round 2)")
    plt.xlabel("Date")
    plt.ylabel("RSI")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"round2_actual_vs_pred_{scenario_name.lower().replace(' ', '_')}.png", dpi=300, bbox_inches="tight")
    plt.show()

plot_actual_vs_pred(preds_a_cv, "Scenario A")
plot_actual_vs_pred(preds_b_cv, "Scenario B")


# 10.4 Boxplot of absolute errors
plt.figure(figsize=(8, 5))
plt.boxplot(
    [
        preds_a_cv["abs_error"],
        preds_b_cv["abs_error"]
    ],
    labels=["Scenario A", "Scenario B"]
)
plt.title("Absolute Error Distribution (Round 2)")
plt.ylabel("Absolute Error")
plt.tight_layout()
plt.savefig("round2_abs_error_boxplot.png", dpi=300, bbox_inches="tight")
plt.show()





print("\nRound 2 finished successfully.")