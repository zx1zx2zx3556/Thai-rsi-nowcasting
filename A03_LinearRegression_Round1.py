import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
import random
import os

from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error
from scipy.stats import t

# =========================================================
# 0) GLOBAL SEED
# =========================================================
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

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
# 6) OPTUNA OBJECTIVE
# =========================================================
def objective_factory(df, feature_cols, revin_func):
    def objective(trial):
        alpha = trial.suggest_float("alpha", 1e-4, 1e3, log=True)
        l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
        fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])
        max_iter = trial.suggest_int("max_iter", 2000, 10000, step=2000)

        tscv = TimeSeriesSplit(n_splits=5)
        mae_scores = []

        for train_idx, test_idx in tscv.split(df):
            train_df = revin_func(df.iloc[train_idx].copy())
            test_df = revin_func(df.iloc[test_idx].copy())

            X_train = train_df[feature_cols]
            y_train = train_df["target_norm"]
            X_test = test_df[feature_cols]
            y_test_raw = test_df["target_raw"].values

            model = ElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                fit_intercept=fit_intercept,
                max_iter=max_iter,
                random_state=SEED,
                selection="cyclic"   # deterministic
            )
            model.fit(X_train, y_train)

            y_pred_norm = model.predict(X_test)
            y_pred_raw = y_pred_norm * test_df["target_sd"].values + test_df["target_mean"].values

            mae_scores.append(mean_absolute_error(y_test_raw, y_pred_raw))

        return float(np.mean(mae_scores))
    return objective

# =========================================================
# 7) RUN OPTUNA  <-- สำคัญ: ใช้ sampler ที่ fix seed จริง
# =========================================================
sampler_a = optuna.samplers.TPESampler(seed=SEED)
sampler_b = optuna.samplers.TPESampler(seed=SEED)

study_a = optuna.create_study(direction="minimize", sampler=sampler_a)
study_a.optimize(
    objective_factory(df_a, feature_cols_a, apply_revin_scenario_a),
    n_trials=50
)

study_b = optuna.create_study(direction="minimize", sampler=sampler_b)
study_b.optimize(
    objective_factory(df_b, feature_cols_b, apply_revin_scenario_b),
    n_trials=50
)

def run_best_model(df, feature_cols, scenario_name, revin_func, best_params):

    tscv = TimeSeriesSplit(n_splits=5)

    metrics = []
    pred_list = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(df), start=1):

        train_df = revin_func(df.iloc[train_idx].copy())
        test_df = revin_func(df.iloc[test_idx].copy())

        X_train = train_df[feature_cols]
        y_train = train_df["target_norm"]

        X_test = test_df[feature_cols]
        y_test_raw = test_df["target_raw"].values

        model = ElasticNet(**best_params, random_state=42)

        model.fit(X_train, y_train)

        y_pred_norm = model.predict(X_test)
        y_pred_raw = (
            y_pred_norm * test_df["target_sd"].values
            + test_df["target_mean"].values
        )

        metrics.append({
            "scenario": scenario_name,
            "fold": fold,
            "MAE": mean_absolute_error(y_test_raw, y_pred_raw),
            "MAPE": mape(y_test_raw, y_pred_raw),
            "sMAPE": smape(y_test_raw, y_pred_raw)
        })

        pred_list.append(pd.DataFrame({
            "scenario": scenario_name,
            "fold": fold,
            "date": test_df["date"].values,
            "actual": y_test_raw,
            "pred": y_pred_raw,
            "abs_error": np.abs(y_test_raw - y_pred_raw)
        }))

    return pd.DataFrame(metrics), pd.concat(pred_list, ignore_index=True)


metrics_a_opt, preds_a_opt = run_best_model(
    df_a, feature_cols_a, "Scenario A", apply_revin_scenario_a, study_a.best_params
)

metrics_b_opt, preds_b_opt = run_best_model(
    df_b, feature_cols_b, "Scenario B", apply_revin_scenario_b, study_b.best_params
)
# =========================================================
# PLOTS (Run after metrics_a_opt, metrics_b_opt, preds_a_opt, preds_b_opt)
# =========================================================
import numpy as np
import matplotlib.pyplot as plt

# กัน error ถ้าตัวแปรยังไม่ถูกสร้าง
#rint("metrics_a_opt shape:", metrics_a_opt.shape)
#print("metrics_b_opt shape:", metrics_b_opt.shape)
#print("preds_a_opt shape:", preds_a_opt.shape)
#print("preds_b_opt shape:", preds_b_opt.shape)

# ---------------------------------------------------------
# 1) Summary table
# ---------------------------------------------------------
summary_opt = pd.concat([metrics_a_opt, metrics_b_opt], ignore_index=True)

summary_mean_opt = summary_opt.groupby("scenario")[["MAE", "MAPE", "sMAPE"]].mean()
summary_std_opt = summary_opt.groupby("scenario")[["MAE", "MAPE", "sMAPE"]].std()

print("\n=== Round 3 Mean ===")
print(summary_mean_opt)

print("\n=== Round 3 Std ===")
print(summary_std_opt)

summary_mean_opt.to_csv("round3_mean_metrics.csv")
summary_std_opt.to_csv("round3_std_metrics.csv")
summary_opt.to_csv("round3_fold_results.csv", index=False)

# ---------------------------------------------------------
# 2) Line plot: mean comparison
# ---------------------------------------------------------
mean_a = metrics_a_opt[["MAE", "MAPE", "sMAPE"]].mean()
mean_b = metrics_b_opt[["MAE", "MAPE", "sMAPE"]].mean()

metric_names = ["MAE", "MAPE", "sMAPE"]
x = np.arange(len(metric_names))

scenario_a_mean = [mean_a["MAE"], mean_a["MAPE"], mean_a["sMAPE"]]
scenario_b_mean = [mean_b["MAE"], mean_b["MAPE"], mean_b["sMAPE"]]

plt.figure(figsize=(8, 5))
plt.plot(x, scenario_a_mean, marker="o", linewidth=2, label="Scenario A")
plt.plot(x, scenario_b_mean, marker="o", linewidth=2, label="Scenario B")

for i, v in enumerate(scenario_a_mean):
    plt.text(i, v, f"{v:.2f}", ha="center", va="bottom")

for i, v in enumerate(scenario_b_mean):
    plt.text(i, v, f"{v:.2f}", ha="center", va="bottom")

plt.xticks(x, metric_names)
plt.ylabel("Mean Error Value")
plt.title("Mean Forecasting Error Comparison Between Scenario A and Scenario B")
plt.legend()
plt.tight_layout()
plt.savefig("round3_mean_metrics_line.png", dpi=300, bbox_inches="tight")
#plt.show()

# ---------------------------------------------------------
# 3) Line plot: mean ± std
# ---------------------------------------------------------
std_a = metrics_a_opt[["MAE", "MAPE", "sMAPE"]].std()
std_b = metrics_b_opt[["MAE", "MAPE", "sMAPE"]].std()

scenario_a_std = [std_a["MAE"], std_a["MAPE"], std_a["sMAPE"]]
scenario_b_std = [std_b["MAE"], std_b["MAPE"], std_b["sMAPE"]]

plt.figure(figsize=(8, 5))
plt.errorbar(
    x, scenario_a_mean,
    yerr=scenario_a_std,
    marker="o", linewidth=2, capsize=5, label="Scenario A"
)
plt.errorbar(
    x, scenario_b_mean,
    yerr=scenario_b_std,
    marker="o", linewidth=2, capsize=5, label="Scenario B"
)

plt.xticks(x, metric_names)
plt.ylabel("Mean Error Value")
plt.title("Mean ± Standard Deviation of Forecasting Errors")
plt.legend()
plt.tight_layout()
plt.savefig("round3_mean_std_metrics_line.png", dpi=300, bbox_inches="tight")
#plt.show()

# ---------------------------------------------------------
# 4) MAE / MAPE / sMAPE by fold
# ---------------------------------------------------------
def plot_metric_by_fold(metrics_df, metric_name):
    plt.figure(figsize=(8, 5))

    for scenario in metrics_df["scenario"].unique():
        tmp = metrics_df[metrics_df["scenario"] == scenario]
        plt.plot(tmp["fold"], tmp[metric_name], marker="o", linewidth=2, label=scenario)

        for _, row in tmp.iterrows():
            plt.text(row["fold"], row[metric_name], f'{row[metric_name]:.2f}', ha="center", va="bottom")

    plt.title(f"{metric_name} by Fold (Round 3)")
    plt.xlabel("Fold")
    plt.ylabel(metric_name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"round3_{metric_name.lower()}_by_fold.png", dpi=300, bbox_inches="tight")
    #plt.show()

#plot_metric_by_fold(summary_opt, "MAE")
#plot_metric_by_fold(summary_opt, "MAPE")
#plot_metric_by_fold(summary_opt, "sMAPE")

# ---------------------------------------------------------
# 5) Actual vs Predicted
# ---------------------------------------------------------
def plot_actual_vs_pred(pred_df, scenario_name):
    tmp = pred_df[pred_df["scenario"] == scenario_name].copy()

    plt.figure(figsize=(10, 5))
    plt.plot(tmp["date"], tmp["actual"], linewidth=2, label="Actual RSI")
    plt.plot(tmp["date"], tmp["pred"], linewidth=2, label="Predicted RSI")

    plt.title(f"{scenario_name}: Actual vs Predicted (Round 3)")
    plt.xlabel("Date")
    plt.ylabel("RSI")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"round3_actual_vs_pred_{scenario_name.lower().replace(' ', '_')}.png", dpi=300, bbox_inches="tight")
    #plt.show()

plot_actual_vs_pred(preds_a_opt, "Scenario A")
plot_actual_vs_pred(preds_b_opt, "Scenario B")

# ---------------------------------------------------------
# 6) Boxplot: absolute error distribution
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.boxplot(
    [preds_a_opt["abs_error"], preds_b_opt["abs_error"]],
    labels=["Scenario A", "Scenario B"]
)
plt.title("Absolute Error Distribution (Round 3)")
plt.ylabel("Absolute Error")
plt.tight_layout()
plt.savefig("round3_abs_error_boxplot.png", dpi=300, bbox_inches="tight")
#plt.show()

# ---------------------------------------------------------
# 7) Diebold-Mariano related plots
# ---------------------------------------------------------

def dm_test(e1, e2):

    d = e1**2 - e2**2
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)

    DM = mean_d / np.sqrt(var_d/len(d))

    p_value = 2 * (1 - t.cdf(np.abs(DM), df=len(d)-1))

    return DM, p_value

dm_df = preds_a_opt.merge(
    preds_b_opt,
    on="date",
    suffixes=("_a", "_b")
).sort_values("date")

err_a = dm_df["actual_a"] - dm_df["pred_a"]
err_b = dm_df["actual_b"] - dm_df["pred_b"]

loss_diff = err_a**2 - err_b**2
cum_diff = np.cumsum(loss_diff)
DM_stat, p_val = dm_test(err_a.values, err_b.values)
#print("\nDiebold-Mariano Test")
#print("DM statistic:", DM_stat)
#print("p-value:", p_val)


plt.figure(figsize=(10, 5))
plt.plot(dm_df["date"], loss_diff, marker="o")
plt.axhline(0, linestyle="--")
plt.title("Loss Differential (Scenario A vs Scenario B) - Round 3")
plt.xlabel("Date")
plt.ylabel("Squared Error Difference")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("round3_loss_differential.png", dpi=300, bbox_inches="tight")
#plt.show()

plt.figure(figsize=(10, 5))
plt.plot(dm_df["date"], cum_diff)
plt.axhline(0, linestyle="--")
plt.title("Cumulative Loss Differential - Round 3")
plt.xlabel("Date")
plt.ylabel("Cumulative Difference")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("round3_cumulative_loss_differential.png", dpi=300, bbox_inches="tight")
#plt.show()

#print("\nAll Round 3 plots finished successfully.")
#print("Best params Scenario A:", study_a.best_params)
#print("Best value Scenario A:", study_a.best_value)

#print("Best params Scenario B:", study_b.best_params)
#print("Best value Scenario B:", study_b.best_value)
#print('Diebold–Mariano Test', dm_df)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error

# =========================================================
# 1) BEST PARAMS FROM ROUND 3
# =========================================================
best_params_a = study_a.best_params
best_params_b = study_b.best_params

# ถ้าไม่มี study_a / study_b ให้ใส่เองแบบนี้
# best_params_a = {"alpha": 31.3495, "l1_ratio": 0.1996, "fit_intercept": False, "max_iter": 2000}
# best_params_b = {"alpha": 4.3416, "l1_ratio": 0.3117, "fit_intercept": False, "max_iter": 2000}

# =========================================================
# 2) LOOK-BACK WINDOWS
# =========================================================
lookback_windows = [24, 36, 48, 60]

# =========================================================
# 3) RUN LOOK-BACK SEARCH
# =========================================================
def run_lookback_experiment(df, feature_cols, scenario_name, revin_func, best_params, lookback_list):
    metrics_list = []
    preds_list = []

    for lookback in lookback_list:
        tscv = TimeSeriesSplit(n_splits=5, max_train_size=lookback)

        fold_counter = 0
        for fold, (train_idx, test_idx) in enumerate(tscv.split(df), start=1):
            train_df = df.iloc[train_idx].copy()
            test_df = df.iloc[test_idx].copy()

            # ข้ามกรณี train window ยังสั้นเกิน lookback มากเกินไป
            if len(train_df) < lookback:
                continue

            fold_counter += 1

            train_df = revin_func(train_df)
            test_df = revin_func(test_df)

            X_train = train_df[feature_cols]
            y_train = train_df["target_norm"]

            X_test = test_df[feature_cols]
            y_test_raw = test_df["target_raw"].values

            model = ElasticNet(
                **best_params,
                random_state=42,
                selection="cyclic"
            )
            model.fit(X_train, y_train)

            y_pred_norm = model.predict(X_test)
            y_pred_raw = y_pred_norm * test_df["target_sd"].values + test_df["target_mean"].values

            metrics_list.append({
                "scenario": scenario_name,
                "lookback": lookback,
                "fold": fold_counter,
                "train_size": len(train_df),
                "test_size": len(test_df),
                "MAE": mean_absolute_error(y_test_raw, y_pred_raw),
                "MAPE": mape(y_test_raw, y_pred_raw),
                "sMAPE": smape(y_test_raw, y_pred_raw)
            })

            preds_list.append(pd.DataFrame({
                "scenario": scenario_name,
                "lookback": lookback,
                "fold": fold_counter,
                "date": test_df["date"].values,
                "actual": y_test_raw,
                "pred": y_pred_raw,
                "abs_error": np.abs(y_test_raw - y_pred_raw)
            }))

    metrics_df = pd.DataFrame(metrics_list)
    preds_df = pd.concat(preds_list, ignore_index=True)

    return metrics_df, preds_df


metrics_a_lb, preds_a_lb = run_lookback_experiment(
    df=df_a,
    feature_cols=feature_cols_a,
    scenario_name="Scenario A",
    revin_func=apply_revin_scenario_a,
    best_params=best_params_a,
    lookback_list=lookback_windows
)

metrics_b_lb, preds_b_lb = run_lookback_experiment(
    df=df_b,
    feature_cols=feature_cols_b,
    scenario_name="Scenario B",
    revin_func=apply_revin_scenario_b,
    best_params=best_params_b,
    lookback_list=lookback_windows
)

# =========================================================
# 4) SUMMARY TABLES
# =========================================================
summary_lb = pd.concat([metrics_a_lb, metrics_b_lb], ignore_index=True)

summary_mean_lb = summary_lb.groupby(["scenario", "lookback"])[["MAE", "MAPE", "sMAPE"]].mean().reset_index()
summary_std_lb = summary_lb.groupby(["scenario", "lookback"])[["MAE", "MAPE", "sMAPE"]].std().reset_index()

print("\n=== Round 4 Mean by Look-back ===")
print(summary_mean_lb)

print("\n=== Round 4 Std by Look-back ===")
print(summary_std_lb)

summary_lb.to_csv("round4_lookback_all_folds.csv", index=False)
summary_mean_lb.to_csv("round4_lookback_mean.csv", index=False)
summary_std_lb.to_csv("round4_lookback_std.csv", index=False)

# =========================================================
# 5) DM TEST FOR EACH LOOK-BACK
# =========================================================
dm_results = []

for lb in lookback_windows:
    tmp_a = preds_a_lb[preds_a_lb["lookback"] == lb].copy()
    tmp_b = preds_b_lb[preds_b_lb["lookback"] == lb].copy()

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

print("\n=== Round 4 Diebold-Mariano Test by Look-back ===")
print(dm_results_df)

dm_results_df.to_csv("round4_dm_test_by_lookback.csv", index=False)

# =========================================================
# 6) PLOTS
# =========================================================

# 6.1 MAE by look-back
plt.figure(figsize=(8, 5))
for scenario in summary_mean_lb["scenario"].unique():
    tmp = summary_mean_lb[summary_mean_lb["scenario"] == scenario]
    plt.plot(tmp["lookback"], tmp["MAE"], marker="o", linewidth=2, label=scenario)

plt.title("MAE by Look-back Window (Round 4)")
plt.xlabel("Look-back Window (Months)")
plt.ylabel("MAE")
plt.legend()
plt.tight_layout()
plt.savefig("round4_mae_by_lookback.png", dpi=300, bbox_inches="tight")
plt.show()

# 6.2 MAPE by look-back
plt.figure(figsize=(8, 5))
for scenario in summary_mean_lb["scenario"].unique():
    tmp = summary_mean_lb[summary_mean_lb["scenario"] == scenario]
    plt.plot(tmp["lookback"], tmp["MAPE"], marker="o", linewidth=2, label=scenario)

plt.title("MAPE by Look-back Window (Round 4)")
plt.xlabel("Look-back Window (Months)")
plt.ylabel("MAPE")
plt.legend()
plt.tight_layout()
plt.savefig("round4_mape_by_lookback.png", dpi=300, bbox_inches="tight")
plt.show()

# 6.3 sMAPE by look-back
plt.figure(figsize=(8, 5))
for scenario in summary_mean_lb["scenario"].unique():
    tmp = summary_mean_lb[summary_mean_lb["scenario"] == scenario]
    plt.plot(tmp["lookback"], tmp["sMAPE"], marker="o", linewidth=2, label=scenario)

plt.title("sMAPE by Look-back Window (Round 4)")
plt.xlabel("Look-back Window (Months)")
plt.ylabel("sMAPE")
plt.legend()
plt.tight_layout()
plt.savefig("round4_smape_by_lookback.png", dpi=300, bbox_inches="tight")
plt.show()

# 6.4 Combined normalized line plot
plt.figure(figsize=(9, 5))

for metric in ["MAE", "MAPE", "sMAPE"]:
    pivot_df = summary_mean_lb.pivot(index="lookback", columns="scenario", values=metric)
    norm_df = (pivot_df - pivot_df.min()) / (pivot_df.max() - pivot_df.min() + 1e-12)

    plt.plot(norm_df.index, norm_df["Scenario A"], marker="o", linestyle="-", label=f"{metric} - Scenario A")
    plt.plot(norm_df.index, norm_df["Scenario B"], marker="o", linestyle="--", label=f"{metric} - Scenario B")

plt.title("Normalized Forecast Errors by Look-back Window (Round 4)")
plt.xlabel("Look-back Window (Months)")
plt.ylabel("Normalized Error (0-1)")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig("round4_normalized_all_metrics.png", dpi=300, bbox_inches="tight")
plt.show()

# 6.5 DM test p-value by look-back
plt.figure(figsize=(8, 5))
plt.plot(dm_results_df["lookback"], dm_results_df["p_value"], marker="o", linewidth=2)
plt.axhline(0.05, linestyle="--")
plt.title("Diebold-Mariano Test p-value by Look-back Window")
plt.xlabel("Look-back Window (Months)")
plt.ylabel("p-value")
plt.tight_layout()
plt.savefig("round4_dm_pvalue_by_lookback.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nRound 4 finished successfully.")