import os
import random
import warnings
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import optuna
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
# เพิ่มใหม่: reset seeds function
# =========================================================
def reset_seeds(seed=42):
    import random
    import numpy as np
    import torch
    import lightning.pytorch as pl

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pl.seed_everything(seed, workers=True)

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
# 5) SEARCH SPACE
# =========================================================
ENCODER_LENGTH = 12
PREDICTION_LENGTH = 1

def build_training_dataset(train_df, feature_cols):
    return TimeSeriesDataSet(
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

# =========================================================
# 6) OPTUNA OBJECTIVE
# =========================================================
def objective_factory(df, feature_cols, revin_func, scenario_name):
    def objective(trial):
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
        hidden_size = trial.suggest_categorical("hidden_size", [8, 16, 24, 32, 48, 64])
        attention_head_size = trial.suggest_categorical("attention_head_size", [1, 2, 4, 8])
        dropout = trial.suggest_float("dropout", 0.05, 0.4)
        lstm_layers = trial.suggest_int("lstm_layers", 1, 3)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        hidden_continuous_size = trial.suggest_categorical("hidden_continuous_size", [4, 8, 16, 32])
        max_epochs = trial.suggest_categorical("max_epochs", [20, 30, 40, 50])

        tscv = TimeSeriesSplit(n_splits=5)
        fold_maes = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(df), start=1):
            reset_seeds(SEED)  # เพิ่มใหม่

            train_df = df.iloc[train_idx].copy()
            test_df = df.iloc[test_idx].copy()

            train_df = revin_func(train_df)
            test_df = revin_func(test_df)

            training = build_training_dataset(train_df, feature_cols)

            validation = TimeSeriesDataSet.from_dataset(
                training,
                pd.concat([train_df.tail(ENCODER_LENGTH), test_df], axis=0),
                predict=True,
                stop_randomization=True
            )

            train_loader = training.to_dataloader(
                train=True,
                batch_size=batch_size,
                num_workers=0,
                shuffle=False  # เพิ่มใหม่
            )

            val_loader = validation.to_dataloader(
                train=False,
                batch_size=batch_size,
                num_workers=0,
                shuffle=False  # เพิ่มใหม่
            )

            tft = TemporalFusionTransformer.from_dataset(
                training,
                learning_rate=learning_rate,
                hidden_size=hidden_size,
                attention_head_size=attention_head_size,
                dropout=dropout,
                hidden_continuous_size=hidden_continuous_size,
                lstm_layers=lstm_layers,
                loss=MAE(),
                output_size=1,
                log_interval=-1,
                reduce_on_plateau_patience=4,
            )

            trainer = pl.Trainer(
                max_epochs=max_epochs,
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

            mae_fold = np.mean(np.abs(true_raw - pred_raw))
            fold_maes.append(mae_fold)

            trial.report(np.mean(fold_maes), step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_maes))
    return objective

# =========================================================
# 7) RUN OPTUNA
# =========================================================
sampler_a = optuna.samplers.TPESampler(seed=SEED)
sampler_b = optuna.samplers.TPESampler(seed=SEED)

study_a_tft = optuna.create_study(direction="minimize", sampler=sampler_a)
study_b_tft = optuna.create_study(direction="minimize", sampler=sampler_b)

study_a_tft.optimize(
    objective_factory(df_a, feature_cols_a, apply_revin_scenario_a, "Scenario A"),
    n_trials=30
)
study_b_tft.optimize(
    objective_factory(df_b, feature_cols_b, apply_revin_scenario_b, "Scenario B"),
    n_trials=30
)

print("\n=== TFT Round 3 Best Params Scenario A ===")
print(study_a_tft.best_params)
print("Best MAE:", study_a_tft.best_value)

print("\n=== TFT Round 3 Best Params Scenario B ===")
print(study_b_tft.best_params)
print("Best MAE:", study_b_tft.best_value)

# =========================================================
# 8) FINAL EVALUATION WITH BEST PARAMS
# =========================================================
def run_tft_best_model(df, feature_cols, scenario_name, revin_func, best_params):
    tscv = TimeSeriesSplit(n_splits=5)

    metrics_list = []
    preds_list = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(df), start=1):
        reset_seeds(SEED)  # เพิ่มใหม่

        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()

        train_df = revin_func(train_df)
        test_df = revin_func(test_df)

        training = build_training_dataset(train_df, feature_cols)

        validation = TimeSeriesDataSet.from_dataset(
            training,
            pd.concat([train_df.tail(ENCODER_LENGTH), test_df], axis=0),
            predict=True,
            stop_randomization=True
        )

        train_loader = training.to_dataloader(
            train=True,
            batch_size=best_params["batch_size"],
            num_workers=0,
            shuffle=False  # เพิ่มใหม่
        )

        val_loader = validation.to_dataloader(
            train=False,
            batch_size=best_params["batch_size"],
            num_workers=0,
            shuffle=False  # เพิ่มใหม่
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
            reduce_on_plateau_patience=4,
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

metrics_a_tft_r3, preds_a_tft_r3 = run_tft_best_model(
    df_a, feature_cols_a, "Scenario A", apply_revin_scenario_a, study_a_tft.best_params
)
metrics_b_tft_r3, preds_b_tft_r3 = run_tft_best_model(
    df_b, feature_cols_b, "Scenario B", apply_revin_scenario_b, study_b_tft.best_params
)

summary_r3 = pd.concat([metrics_a_tft_r3, metrics_b_tft_r3], ignore_index=True)
summary_mean_r3 = summary_r3.groupby("scenario")[["MAE", "MAPE", "sMAPE"]].mean()
summary_std_r3 = summary_r3.groupby("scenario")[["MAE", "MAPE", "sMAPE"]].std()

print("\n=== TFT Round 3 Mean ===")
print(summary_mean_r3)

print("\n=== TFT Round 3 Std ===")
print(summary_std_r3)

# =========================================================
# 9) DM TEST
# =========================================================
dm_df = preds_a_tft_r3.merge(
    preds_b_tft_r3,
    on="date",
    suffixes=("_a", "_b")
).sort_values("date")

err_a = dm_df["actual_a"] - dm_df["pred_a"]
err_b = dm_df["actual_b"] - dm_df["pred_b"]

dm_stat, p_value = dm_test(err_a.values, err_b.values)

print("\n=== TFT Round 3 DM Test ===")
print("DM statistic:", dm_stat)
print("p-value:", p_value)

# =========================================================
# 10) SAVE
# =========================================================
summary_r3.to_csv("tft_round3_metrics_all.csv", index=False)
summary_mean_r3.to_csv("tft_round3_mean.csv")
summary_std_r3.to_csv("tft_round3_std.csv")
preds_a_tft_r3.to_csv("tft_round3_predictions_scenario_a.csv", index=False)
preds_b_tft_r3.to_csv("tft_round3_predictions_scenario_b.csv", index=False)
study_a_tft.trials_dataframe().to_csv("tft_round3_optuna_trials_scenario_a.csv", index=False)
study_b_tft.trials_dataframe().to_csv("tft_round3_optuna_trials_scenario_b.csv", index=False)

print("\nTFT Round 3 finished successfully.")

# =========================================================
# 11) PLOTS FOR TFT ROUND 3
# =========================================================

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 11.1 Metric by fold
# -----------------------------
def plot_metric_by_fold(metrics_df, metric_name):
    plt.figure(figsize=(8, 5))

    for scenario in metrics_df["scenario"].unique():
        tmp = metrics_df[metrics_df["scenario"] == scenario].copy()
        plt.plot(tmp["fold"], tmp[metric_name], marker="o", linewidth=2, label=scenario)

        for _, row in tmp.iterrows():
            plt.text(row["fold"], row[metric_name], f'{row[metric_name]:.2f}',
                     ha="center", va="bottom")

    plt.title(f"TFT Round 3: {metric_name} by Fold")
    plt.xlabel("Fold")
    plt.ylabel(metric_name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"tft_round3_{metric_name.lower()}_by_fold.png", dpi=300, bbox_inches="tight")
    plt.show()

plot_metric_by_fold(summary_r3, "MAE")
plot_metric_by_fold(summary_r3, "MAPE")
plot_metric_by_fold(summary_r3, "sMAPE")


# -----------------------------
# 11.2 Mean comparison line plot
# -----------------------------
mean_a = metrics_a_tft_r3[["MAE", "MAPE", "sMAPE"]].mean()
mean_b = metrics_b_tft_r3[["MAE", "MAPE", "sMAPE"]].mean()

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
plt.title("TFT Round 3: Mean Forecasting Error Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("tft_round3_mean_metrics_line.png", dpi=300, bbox_inches="tight")
plt.show()


# -----------------------------
# 11.3 Mean ± standard deviation
# -----------------------------
std_a = metrics_a_tft_r3[["MAE", "MAPE", "sMAPE"]].std()
std_b = metrics_b_tft_r3[["MAE", "MAPE", "sMAPE"]].std()

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
plt.title("TFT Round 3: Mean ± Standard Deviation of Forecast Errors")
plt.legend()
plt.tight_layout()
plt.savefig("tft_round3_mean_std_metrics_line.png", dpi=300, bbox_inches="tight")
plt.show()


# -----------------------------
# 11.4 Normalized combined line plot
# -----------------------------
vals_a = np.array(scenario_a_mean, dtype=float)
vals_b = np.array(scenario_b_mean, dtype=float)

all_vals = np.vstack([vals_a, vals_b])
norm_vals = (all_vals - all_vals.min(axis=0)) / (all_vals.max(axis=0) - all_vals.min(axis=0) + 1e-12)

norm_a = norm_vals[0]
norm_b = norm_vals[1]

plt.figure(figsize=(8, 5))
plt.plot(x, norm_a, marker="o", linewidth=2, label="Scenario A")
plt.plot(x, norm_b, marker="o", linewidth=2, label="Scenario B")

for i, v in enumerate(norm_a):
    plt.text(i, v, f"{v:.2f}", ha="center", va="bottom")

for i, v in enumerate(norm_b):
    plt.text(i, v, f"{v:.2f}", ha="center", va="bottom")

plt.xticks(x, metric_names)
plt.ylabel("Normalized Error (0–1)")
plt.title("TFT Round 3: Normalized Mean Error Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("tft_round3_normalized_mean_comparison.png", dpi=300, bbox_inches="tight")
plt.show()


# -----------------------------
# 11.5 Actual vs Predicted (average by date)
# -----------------------------
def plot_actual_vs_pred_average(pred_df, scenario_name):
    tmp = pred_df[pred_df["scenario"] == scenario_name].copy()
    avg_df = tmp.groupby("date")[["actual", "pred"]].mean().reset_index()

    plt.figure(figsize=(10, 5))
    plt.plot(avg_df["date"], avg_df["actual"], linewidth=2, label="Actual RSI")
    plt.plot(avg_df["date"], avg_df["pred"], linewidth=2, label="Predicted RSI")

    plt.title(f"TFT Round 3: {scenario_name} Actual vs Predicted (Average by Date)")
    plt.xlabel("Date")
    plt.ylabel("RSI")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"tft_round3_actual_vs_pred_avg_{scenario_name.lower().replace(' ', '_')}.png",
                dpi=300, bbox_inches="tight")
    plt.show()

plot_actual_vs_pred_average(preds_a_tft_r3, "Scenario A")
plot_actual_vs_pred_average(preds_b_tft_r3, "Scenario B")


# -----------------------------
# 11.6 Actual vs Predicted (last fold only)
# -----------------------------
def plot_actual_vs_pred_last_fold(pred_df, scenario_name):
    tmp = pred_df[pred_df["scenario"] == scenario_name].copy()
    last_fold = tmp["fold"].max()
    tmp = tmp[tmp["fold"] == last_fold].sort_values("date")

    plt.figure(figsize=(10, 5))
    plt.plot(tmp["date"], tmp["actual"], linewidth=2, label="Actual RSI")
    plt.plot(tmp["date"], tmp["pred"], linewidth=2, label="Predicted RSI")

    plt.title(f"TFT Round 3: {scenario_name} Actual vs Predicted (Fold {last_fold})")
    plt.xlabel("Date")
    plt.ylabel("RSI")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"tft_round3_actual_vs_pred_lastfold_{scenario_name.lower().replace(' ', '_')}.png",
                dpi=300, bbox_inches="tight")
    plt.show()

plot_actual_vs_pred_last_fold(preds_a_tft_r3, "Scenario A")
plot_actual_vs_pred_last_fold(preds_b_tft_r3, "Scenario B")


# -----------------------------
# 11.7 Absolute error distribution
# -----------------------------
plt.figure(figsize=(8, 5))
plt.boxplot(
    [preds_a_tft_r3["abs_error"], preds_b_tft_r3["abs_error"]],
    tick_labels=["Scenario A", "Scenario B"]
)
plt.title("TFT Round 3: Absolute Error Distribution")
plt.ylabel("Absolute Error")
plt.tight_layout()
plt.savefig("tft_round3_abs_error_boxplot.png", dpi=300, bbox_inches="tight")
plt.show()


# -----------------------------
# 11.8 Loss differential and cumulative loss differential
# -----------------------------
dm_plot_df = preds_a_tft_r3.merge(
    preds_b_tft_r3,
    on="date",
    suffixes=("_a", "_b")
).sort_values("date")

err_a_plot = dm_plot_df["actual_a"] - dm_plot_df["pred_a"]
err_b_plot = dm_plot_df["actual_b"] - dm_plot_df["pred_b"]

loss_diff = err_a_plot**2 - err_b_plot**2
cum_diff = np.cumsum(loss_diff)

plt.figure(figsize=(10, 5))
plt.plot(dm_plot_df["date"], loss_diff, marker="o")
plt.axhline(0, linestyle="--")
plt.title("TFT Round 3: Loss Differential (Scenario A vs Scenario B)")
plt.xlabel("Date")
plt.ylabel("Squared Error Difference")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("tft_round3_loss_differential.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(dm_plot_df["date"], cum_diff)
plt.axhline(0, linestyle="--")
plt.title("TFT Round 3: Cumulative Loss Differential")
plt.xlabel("Date")
plt.ylabel("Cumulative Difference")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("tft_round3_cumulative_loss_differential.png", dpi=300, bbox_inches="tight")
plt.show()


# -----------------------------
# 11.9 Optuna optimization history
# -----------------------------
try:
    from optuna.visualization.matplotlib import plot_optimization_history

    plot_optimization_history(study_a_tft)
    plt.title("TFT Round 3: Optuna Optimization History - Scenario A")
    plt.tight_layout()
    plt.savefig("tft_round3_optuna_history_scenario_a.png", dpi=300, bbox_inches="tight")
    plt.show()

    plot_optimization_history(study_b_tft)
    plt.title("TFT Round 3: Optuna Optimization History - Scenario B")
    plt.tight_layout()
    plt.savefig("tft_round3_optuna_history_scenario_b.png", dpi=300, bbox_inches="tight")
    plt.show()
except Exception as e:
    print("Optuna optimization history plot skipped:", e)


# -----------------------------
# 11.10 Optuna parameter importance
# -----------------------------
try:
    from optuna.visualization.matplotlib import plot_param_importances

    plot_param_importances(study_a_tft)
    plt.title("TFT Round 3: Parameter Importance - Scenario A")
    plt.tight_layout()
    plt.savefig("tft_round3_param_importance_scenario_a.png", dpi=300, bbox_inches="tight")
    plt.show()

    plot_param_importances(study_b_tft)
    plt.title("TFT Round 3: Parameter Importance - Scenario B")
    plt.tight_layout()
    plt.savefig("tft_round3_param_importance_scenario_b.png", dpi=300, bbox_inches="tight")
    plt.show()
except Exception as e:
    print("Optuna parameter importance plot skipped:", e)

print("\nAll TFT Round 3 plots finished successfully.")