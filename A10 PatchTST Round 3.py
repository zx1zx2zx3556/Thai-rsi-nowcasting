import os
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna

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
# 5) ROUND 3 SETTINGS
# =========================================================
H = 16
INPUT_SIZE = 24
N_WINDOWS = 5
STEP_SIZE = 16
VAL_SIZE = 16

# =========================================================
# 6) BUILD MODEL FROM TRIAL PARAMS
# =========================================================
def build_patchtst_model(trial=None, best_params=None):
    if trial is not None:
        patch_len = trial.suggest_categorical("patch_len", [8, 16])
        stride = trial.suggest_categorical("stride", [4, 8])
        hidden_size = trial.suggest_categorical("hidden_size", [16, 32, 64, 128])
        n_heads = trial.suggest_categorical("n_heads", [2, 4, 8])
        encoder_layers = trial.suggest_int("encoder_layers", 2, 4)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
        dropout = trial.suggest_float("dropout", 0.05, 0.40)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    else:
        patch_len = best_params["patch_len"]
        stride = best_params["stride"]
        hidden_size = best_params["hidden_size"]
        n_heads = best_params["n_heads"]
        encoder_layers = best_params["encoder_layers"]
        learning_rate = best_params["learning_rate"]
        dropout = best_params["dropout"]
        batch_size = best_params["batch_size"]

    # linear_hidden_size ตั้งให้สัมพันธ์กับ hidden_size
    linear_hidden_size = hidden_size * 2

    model = PatchTST(
        h=H,
        input_size=INPUT_SIZE,
        encoder_layers=encoder_layers,
        n_heads=n_heads,
        hidden_size=hidden_size,
        linear_hidden_size=linear_hidden_size,
        dropout=dropout,
        fc_dropout=dropout,
        head_dropout=0.0,
        attn_dropout=0.0,
        patch_len=patch_len,
        stride=stride,
        revin=True,
        revin_affine=False,
        revin_subtract_last=True,
        activation="gelu",
        batch_normalization=False,
        learning_rate=learning_rate,
        max_steps=500,
        val_check_steps=50,
        early_stop_patience_steps=3,
        batch_size=batch_size,
        start_padding_enabled=True,
        random_seed=SEED
    )
    return model

# =========================================================
# 7) OBJECTIVE FUNCTION
# optimize mean MAE of RSI across 5 windows
# =========================================================
def objective_factory(panel_df, scenario_name):
    def objective(trial):
        reset_seeds(SEED)

        try:
            model = build_patchtst_model(trial=trial)
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
            rsi_df.rename(columns={"y": "actual", pred_col: "pred"}, inplace=True)

            mae_score = mean_absolute_error(rsi_df["actual"], rsi_df["pred"])
            return float(mae_score)

        except Exception as e:
            print(f"[{scenario_name}] Trial failed:", e)
            return 1e10

    return objective

# =========================================================
# 8) RUN OPTUNA
# =========================================================
sampler_a = optuna.samplers.TPESampler(seed=SEED)
sampler_b = optuna.samplers.TPESampler(seed=SEED)

study_a_patch = optuna.create_study(direction="minimize", sampler=sampler_a)
study_b_patch = optuna.create_study(direction="minimize", sampler=sampler_b)

study_a_patch.optimize(
    objective_factory(panel_a, "Scenario A"),
    n_trials=30
)

study_b_patch.optimize(
    objective_factory(panel_b, "Scenario B"),
    n_trials=30
)

print("\n=== PatchTST Round 3 Best Params Scenario A ===")
print(study_a_patch.best_params)
print("Best MAE:", study_a_patch.best_value)

print("\n=== PatchTST Round 3 Best Params Scenario B ===")
print(study_b_patch.best_params)
print("Best MAE:", study_b_patch.best_value)

best_params_a = study_a_patch.best_params
best_params_b = study_b_patch.best_params

# =========================================================
# 9) FINAL EVALUATION WITH BEST PARAMS
# =========================================================
def run_best_patchtst_cv(panel_df, scenario_name, best_params):
    reset_seeds(SEED)

    model = build_patchtst_model(best_params=best_params)
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

    # keep only RSI channel for evaluation
    rsi_df = cv_df[cv_df["unique_id"] == "RSI"].copy().sort_values(["cutoff", "ds"])
    rsi_df.rename(columns={"ds": "date", "y": "actual", pred_col: "pred"}, inplace=True)
    rsi_df["scenario"] = scenario_name
    rsi_df["abs_error"] = np.abs(rsi_df["actual"] - rsi_df["pred"])

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

metrics_a_patch_r3, preds_a_patch_r3, preds_all_a_patch_r3 = run_best_patchtst_cv(
    panel_df=panel_a,
    scenario_name="Scenario A",
    best_params=best_params_a
)

metrics_b_patch_r3, preds_b_patch_r3, preds_all_b_patch_r3 = run_best_patchtst_cv(
    panel_df=panel_b,
    scenario_name="Scenario B",
    best_params=best_params_b
)

# =========================================================
# 10) SUMMARY
# =========================================================
summary_r3 = pd.concat([metrics_a_patch_r3, metrics_b_patch_r3], ignore_index=True)

summary_mean_r3 = summary_r3.groupby("scenario")[["MAE", "MAPE", "sMAPE"]].mean()
summary_std_r3 = summary_r3.groupby("scenario")[["MAE", "MAPE", "sMAPE"]].std()

print("\n=== PatchTST Round 3 Metrics by Fold ===")
print(summary_r3)

print("\n=== PatchTST Round 3 Mean ===")
print(summary_mean_r3)

print("\n=== PatchTST Round 3 Std ===")
print(summary_std_r3)

# =========================================================
# 11) DIEBOLD-MARIANO TEST
# =========================================================
dm_df = preds_a_patch_r3.merge(
    preds_b_patch_r3,
    on=["date", "fold"],
    suffixes=("_a", "_b")
).sort_values(["fold", "date"])

err_a = dm_df["actual_a"] - dm_df["pred_a"]
err_b = dm_df["actual_b"] - dm_df["pred_b"]

dm_stat, p_value = dm_test(err_a.values, err_b.values)

print("\n=== PatchTST Round 3 DM Test ===")
print("DM statistic:", dm_stat)
print("p-value:", p_value)

# =========================================================
# 12) SAVE
# =========================================================
summary_r3.to_csv("patchtst_round3_metrics_all.csv", index=False)
summary_mean_r3.to_csv("patchtst_round3_mean.csv")
summary_std_r3.to_csv("patchtst_round3_std.csv")
preds_a_patch_r3.to_csv("patchtst_round3_predictions_scenario_a_rsi.csv", index=False)
preds_b_patch_r3.to_csv("patchtst_round3_predictions_scenario_b_rsi.csv", index=False)
preds_all_a_patch_r3.to_csv("patchtst_round3_predictions_scenario_a_all_channels.csv", index=False)
preds_all_b_patch_r3.to_csv("patchtst_round3_predictions_scenario_b_all_channels.csv", index=False)
study_a_patch.trials_dataframe().to_csv("patchtst_round3_optuna_trials_scenario_a.csv", index=False)
study_b_patch.trials_dataframe().to_csv("patchtst_round3_optuna_trials_scenario_b.csv", index=False)

# =========================================================
# 13) PLOTS
# =========================================================
def plot_metric_by_fold(metrics_df, metric_name):
    plt.figure(figsize=(8, 5))
    for scenario in metrics_df["scenario"].unique():
        tmp = metrics_df[metrics_df["scenario"] == scenario].copy()
        plt.plot(tmp["fold"], tmp[metric_name], marker="o", linewidth=2, label=scenario)
    plt.title(f"PatchTST Round 3: {metric_name} by Fold")
    plt.xlabel("Fold")
    plt.ylabel(metric_name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"patchtst_round3_{metric_name.lower()}_by_fold.png", dpi=300, bbox_inches="tight")
    plt.show()

plot_metric_by_fold(summary_r3, "MAE")
plot_metric_by_fold(summary_r3, "MAPE")
plot_metric_by_fold(summary_r3, "sMAPE")

mean_a = metrics_a_patch_r3[["MAE", "MAPE", "sMAPE"]].mean()
mean_b = metrics_b_patch_r3[["MAE", "MAPE", "sMAPE"]].mean()

metric_names = ["MAE", "MAPE", "sMAPE"]
x = np.arange(len(metric_names))

scenario_a_mean = [mean_a["MAE"], mean_a["MAPE"], mean_a["sMAPE"]]
scenario_b_mean = [mean_b["MAE"], mean_b["MAPE"], mean_b["sMAPE"]]

plt.figure(figsize=(8, 5))
plt.plot(x, scenario_a_mean, marker="o", linewidth=2, label="Scenario A")
plt.plot(x, scenario_b_mean, marker="o", linewidth=2, label="Scenario B")
plt.xticks(x, metric_names)
plt.ylabel("Mean Error Value")
plt.title("PatchTST Round 3: Mean Forecasting Error Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("patchtst_round3_mean_metrics_line.png", dpi=300, bbox_inches="tight")
plt.show()

std_a = metrics_a_patch_r3[["MAE", "MAPE", "sMAPE"]].std()
std_b = metrics_b_patch_r3[["MAE", "MAPE", "sMAPE"]].std()

scenario_a_std = [std_a["MAE"], std_a["MAPE"], std_a["sMAPE"]]
scenario_b_std = [std_b["MAE"], std_b["MAPE"], std_b["sMAPE"]]

plt.figure(figsize=(8, 5))
plt.errorbar(x, scenario_a_mean, yerr=scenario_a_std, marker="o", linewidth=2, capsize=5, label="Scenario A")
plt.errorbar(x, scenario_b_mean, yerr=scenario_b_std, marker="o", linewidth=2, capsize=5, label="Scenario B")
plt.xticks(x, metric_names)
plt.ylabel("Mean Error Value")
plt.title("PatchTST Round 3: Mean ± Standard Deviation of Forecast Errors")
plt.legend()
plt.tight_layout()
plt.savefig("patchtst_round3_mean_std_metrics_line.png", dpi=300, bbox_inches="tight")
plt.show()

def plot_actual_vs_pred_average(pred_df, scenario_name):
    tmp = pred_df.copy()
    avg_df = tmp.groupby("date")[["actual", "pred"]].mean().reset_index()

    plt.figure(figsize=(10, 5))
    plt.plot(avg_df["date"], avg_df["actual"], linewidth=2, label="Actual RSI")
    plt.plot(avg_df["date"], avg_df["pred"], linewidth=2, label="Predicted RSI")
    plt.title(f"PatchTST Round 3: {scenario_name} Actual vs Predicted (Average by Date)")
    plt.xlabel("Date")
    plt.ylabel("RSI")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(
        f"patchtst_round3_actual_vs_pred_avg_{scenario_name.lower().replace(' ', '_')}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

plot_actual_vs_pred_average(preds_a_patch_r3, "Scenario A")
plot_actual_vs_pred_average(preds_b_patch_r3, "Scenario B")

plt.figure(figsize=(8, 5))
plt.boxplot(
    [preds_a_patch_r3["abs_error"], preds_b_patch_r3["abs_error"]],
    tick_labels=["Scenario A", "Scenario B"]
)
plt.title("PatchTST Round 3: Absolute Error Distribution (RSI)")
plt.ylabel("Absolute Error")
plt.tight_layout()
plt.savefig("patchtst_round3_abs_error_boxplot.png", dpi=300, bbox_inches="tight")
plt.show()

loss_diff = err_a**2 - err_b**2
cum_diff = np.cumsum(loss_diff)

plt.figure(figsize=(10, 5))
plt.plot(dm_df["date"], loss_diff, marker="o")
plt.axhline(0, linestyle="--")
plt.title("PatchTST Round 3: Loss Differential (Scenario A vs Scenario B)")
plt.xlabel("Date")
plt.ylabel("Squared Error Difference")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("patchtst_round3_loss_differential.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(dm_df["date"], cum_diff)
plt.axhline(0, linestyle="--")
plt.title("PatchTST Round 3: Cumulative Loss Differential")
plt.xlabel("Date")
plt.ylabel("Cumulative Difference")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("patchtst_round3_cumulative_loss_differential.png", dpi=300, bbox_inches="tight")
plt.show()

# Optional: plot Optuna trial values
try:
    plt.figure(figsize=(8, 5))
    plt.plot(
        range(1, len(study_a_patch.trials) + 1),
        [t.value for t in study_a_patch.trials],
        marker="o",
        linewidth=1.5
    )
    plt.title("PatchTST Round 3: Optuna Trial Values - Scenario A")
    plt.xlabel("Trial")
    plt.ylabel("Objective (MAE)")
    plt.tight_layout()
    plt.savefig("patchtst_round3_optuna_trials_scenario_a.png", dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(
        range(1, len(study_b_patch.trials) + 1),
        [t.value for t in study_b_patch.trials],
        marker="o",
        linewidth=1.5
    )
    plt.title("PatchTST Round 3: Optuna Trial Values - Scenario B")
    plt.xlabel("Trial")
    plt.ylabel("Objective (MAE)")
    plt.tight_layout()
    plt.savefig("patchtst_round3_optuna_trials_scenario_b.png", dpi=300, bbox_inches="tight")
    plt.show()
except Exception as e:
    print("Optuna trial plot skipped:", e)

print("\nPatchTST Round 3 finished successfully.")