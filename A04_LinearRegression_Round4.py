import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error

# =========================================================
# 1) BEST PARAMS FROM ROUND 3
# =========================================================

best_params_a = {"alpha": 31.3495, "l1_ratio": 0.1996, "fit_intercept": False, "max_iter": 2000}
best_params_b = {"alpha": 4.3416, "l1_ratio": 0.3117, "fit_intercept": False, "max_iter": 2000}

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
