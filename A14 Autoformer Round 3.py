import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import random
import warnings
import numpy as np
import pandas as pd
import optuna

from sklearn.metrics import mean_absolute_error
from scipy.stats import t

from neuralforecast import NeuralForecast
from neuralforecast.models import Autoformer

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
    p_value = 2 * (1 - t.cdf(np.abs(dm_stat), df=len(d) - 1))
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
# 4) SETTINGS
# =========================================================
H = 16
INPUT_SIZE = 24

N_WINDOWS = 5
STEP_SIZE = 10
VAL_SIZE = 16
N_TRIALS = 20

SAFE_MOVING_AVG_WINDOW = 25  # fix เพื่อลด 24 vs 23 mismatch

# =========================================================
# 5) BALANCED MONTHLY DATA
#    สำคัญมาก: dropna ทั้งตารางก่อน เพื่อให้ทุก channel
#    มีช่วงเวลาเดียวกันเป๊ะ
# =========================================================
def make_balanced_wide_df(df, channels, scenario_name):
    cols = ["date"] + channels
    out = df[cols].copy()

    # ลบแถวที่มีค่าว่างใน channel ใด ๆ ออกทั้งหมด
    out = out.dropna(subset=channels).copy()

    # ตรวจสอบ date ซ้ำ
    out = out.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)

    if out.empty:
        raise ValueError(f"{scenario_name}: no data left after balancing.")

    # เช็กว่าทุกคอลัมน์มีจำนวนข้อมูลเท่ากัน
    counts = out[channels].notna().sum()
    if counts.nunique() != 1:
        raise ValueError(f"{scenario_name}: channel lengths are still inconsistent after balancing.")

    print(f"\n[{scenario_name}] Balanced wide df shape: {out.shape}")
    print(f"[{scenario_name}] Date range: {out['date'].min().date()} to {out['date'].max().date()}")
    print(f"[{scenario_name}] Monthly rows available: {len(out)}")
    return out

df_a_bal = make_balanced_wide_df(df_a, scenario_a_channels, "Scenario A")
df_b_bal = make_balanced_wide_df(df_b, scenario_b_channels, "Scenario B")

# =========================================================
# 6) WIDE -> PANEL
# =========================================================
def prepare_autoformer_panel_from_balanced(df_bal, channels):
    panel_list = []

    for col in channels:
        tmp = df_bal[["date", col]].copy()
        tmp["unique_id"] = col
        tmp["ds"] = pd.to_datetime(tmp["date"])
        tmp["y"] = pd.to_numeric(tmp[col], errors="coerce")
        tmp = tmp[["unique_id", "ds", "y"]]
        panel_list.append(tmp)

    panel_df = pd.concat(panel_list, ignore_index=True)
    panel_df = panel_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    # final guard
    panel_df = panel_df.dropna().reset_index(drop=True)

    return panel_df

panel_a = prepare_autoformer_panel_from_balanced(df_a_bal, scenario_a_channels)
panel_b = prepare_autoformer_panel_from_balanced(df_b_bal, scenario_b_channels)

# =========================================================
# 7) PRE-CHECK PANEL LENGTH
# =========================================================
def validate_panel(panel_df, scenario_name, input_size, h, n_windows, step_size, val_size):
    lengths = panel_df.groupby("unique_id").size()
    min_len = lengths.min()

    # เงื่อนไขขั้นต่ำแบบ conservative
    required_min = input_size + h + val_size + (n_windows - 1) * step_size

    print(f"\n[{scenario_name}] Min channel length: {min_len}")
    print(f"[{scenario_name}] Required minimum length (conservative): {required_min}")

    if min_len < required_min:
        raise ValueError(
            f"{scenario_name}: insufficient sequence length. "
            f"Minimum channel length = {min_len}, but required >= {required_min}."
        )

validate_panel(panel_a, "Scenario A", INPUT_SIZE, H, N_WINDOWS, STEP_SIZE, VAL_SIZE)
validate_panel(panel_b, "Scenario B", INPUT_SIZE, H, N_WINDOWS, STEP_SIZE, VAL_SIZE)

# =========================================================
# 8) BUILD AUTOFORMER MODEL
#    คงกรอบเดิม แต่ตัด window ที่เสี่ยง mismatch ออก
# =========================================================
def build_autoformer_model(trial=None, best_params=None):
    if trial is not None:
        hidden_size = trial.suggest_categorical("hidden_size", [32, 64])
        dropout = trial.suggest_float("dropout", 0.05, 0.30)
        factor = trial.suggest_categorical("factor", [1, 3, 5])
        n_head = trial.suggest_categorical("n_head", [2, 4])
        conv_hidden_size = trial.suggest_categorical("conv_hidden_size", [8, 16, 32])
        encoder_layers = trial.suggest_categorical("encoder_layers", [1, 2])
        decoder_layers = trial.suggest_categorical("decoder_layers", [1])

        # FIX สำคัญ: ใช้เฉพาะ window ที่ปลอดภัย
        moving_avg_window = SAFE_MOVING_AVG_WINDOW

        learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-4, log=True)
        max_steps = trial.suggest_categorical("max_steps", [200, 300])
        val_check_steps = trial.suggest_categorical("val_check_steps", [50])
        early_stop_patience_steps = trial.suggest_categorical("early_stop_patience_steps", [2, 3])
        batch_size = trial.suggest_categorical("batch_size", [16, 32])
    else:
        hidden_size = best_params["hidden_size"]
        dropout = best_params["dropout"]
        factor = best_params["factor"]
        n_head = best_params["n_head"]
        conv_hidden_size = best_params["conv_hidden_size"]
        encoder_layers = best_params["encoder_layers"]
        decoder_layers = best_params["decoder_layers"]
        moving_avg_window = best_params["MovingAvg_window"]
        learning_rate = best_params["learning_rate"]
        max_steps = best_params["max_steps"]
        val_check_steps = best_params["val_check_steps"]
        early_stop_patience_steps = best_params["early_stop_patience_steps"]
        batch_size = best_params["batch_size"]

    model = Autoformer(
        h=H,
        input_size=INPUT_SIZE,
        hidden_size=hidden_size,
        dropout=dropout,
        factor=factor,
        n_head=n_head,
        conv_hidden_size=conv_hidden_size,
        activation="gelu",
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        MovingAvg_window=moving_avg_window,
        learning_rate=learning_rate,
        max_steps=max_steps,
        val_check_steps=val_check_steps,
        early_stop_patience_steps=early_stop_patience_steps,
        batch_size=batch_size,
        start_padding_enabled=True,
        random_seed=SEED,
        accelerator="cpu",
        devices=1,
        enable_progress_bar=True
    )
    return model

# =========================================================
# 9) OBJECTIVE FUNCTION
# =========================================================
def objective_factory(panel_df, scenario_name):
    def objective(trial):
        reset_seeds(SEED)

        try:
            model = build_autoformer_model(trial=trial)
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
            print(f"[{scenario_name}] Trial failed: {e}")
            return 1e10

    return objective

# =========================================================
# 10) RUN OPTUNA
# =========================================================
sampler_a = optuna.samplers.TPESampler(seed=SEED)
sampler_b = optuna.samplers.TPESampler(seed=SEED)

study_a_auto = optuna.create_study(direction="minimize", sampler=sampler_a)
study_b_auto = optuna.create_study(direction="minimize", sampler=sampler_b)

study_a_auto.optimize(
    objective_factory(panel_a, "Scenario A"),
    n_trials=N_TRIALS
)

study_b_auto.optimize(
    objective_factory(panel_b, "Scenario B"),
    n_trials=N_TRIALS
)

print("\n=== Autoformer Round 3 Best Params Scenario A ===")
print(study_a_auto.best_params)
print("Best MAE:", study_a_auto.best_value)

print("\n=== Autoformer Round 3 Best Params Scenario B ===")
print(study_b_auto.best_params)
print("Best MAE:", study_b_auto.best_value)

# เพิ่ม MovingAvg_window กลับเข้าไปใน best_params เพื่อใช้รอบ evaluate
best_params_a = study_a_auto.best_params.copy()
best_params_b = study_b_auto.best_params.copy()

best_params_a["MovingAvg_window"] = SAFE_MOVING_AVG_WINDOW
best_params_b["MovingAvg_window"] = SAFE_MOVING_AVG_WINDOW

# =========================================================
# 11) FINAL EVALUATION WITH BEST PARAMS
# =========================================================
def fit_predict_autoformer_cv_best(panel_df, scenario_name, best_params):
    reset_seeds(SEED)

    model = build_autoformer_model(best_params=best_params)
    nf = NeuralForecast(models=[model], freq="MS")

    cv_df = nf.cross_validation(
        df=panel_df,
        n_windows=N_WINDOWS,
        step_size=STEP_SIZE,
        val_size=VAL_SIZE,
        refit=False,
        verbose=0
    )

    pred_col = [c for c in cv_df.columns if c not in ["unique_id", "ds", "cutoff", "y"]][0]

    result = cv_df.copy()
    result.rename(columns={"ds": "date", "y": "actual", pred_col: "pred"}, inplace=True)
    result["scenario"] = scenario_name
    result["abs_error"] = np.abs(result["actual"] - result["pred"])

    rsi_result = result[result["unique_id"] == "RSI"].copy().sort_values(["cutoff", "date"])

    cutoff_order = {c: i + 1 for i, c in enumerate(sorted(rsi_result["cutoff"].unique()))}
    rsi_result["fold"] = rsi_result["cutoff"].map(cutoff_order)

    metrics_list = []
    for fold in sorted(rsi_result["fold"].unique()):
        tmp = rsi_result[rsi_result["fold"] == fold].copy()

        metrics_list.append({
            "scenario": scenario_name,
            "fold": fold,
            "test_size": len(tmp),
            "MAE": mean_absolute_error(tmp["actual"], tmp["pred"]),
            "MAPE": mape(tmp["actual"], tmp["pred"]),
            "sMAPE": smape(tmp["actual"], tmp["pred"])
        })

    metrics_df = pd.DataFrame(metrics_list)
    return metrics_df, rsi_result, result

metrics_a_auto_r3, preds_a_auto_r3, preds_all_a_auto_r3 = fit_predict_autoformer_cv_best(
    panel_df=panel_a,
    scenario_name="Scenario A",
    best_params=best_params_a
)

metrics_b_auto_r3, preds_b_auto_r3, preds_all_b_auto_r3 = fit_predict_autoformer_cv_best(
    panel_df=panel_b,
    scenario_name="Scenario B",
    best_params=best_params_b
)

# =========================================================
# 12) SUMMARY
# =========================================================
summary_r3 = pd.concat([metrics_a_auto_r3, metrics_b_auto_r3], ignore_index=True)

summary_mean_r3 = summary_r3.groupby("scenario")[["MAE", "MAPE", "sMAPE"]].mean()
summary_std_r3 = summary_r3.groupby("scenario")[["MAE", "MAPE", "sMAPE"]].std()

print("\n=== Autoformer Round 3 Metrics by Fold ===")
print(summary_r3)

print("\n=== Autoformer Round 3 Mean ===")
print(summary_mean_r3)

print("\n=== Autoformer Round 3 Std ===")
print(summary_std_r3)

# =========================================================
# 13) DIEBOLD-MARIANO TEST
# =========================================================
dm_df = preds_a_auto_r3.merge(
    preds_b_auto_r3,
    on=["date", "fold"],
    suffixes=("_a", "_b")
).sort_values(["fold", "date"])

err_a = dm_df["actual_a"] - dm_df["pred_a"]
err_b = dm_df["actual_b"] - dm_df["pred_b"]

dm_stat, p_value = dm_test(err_a.values, err_b.values)

print("\n=== Autoformer Round 3 DM Test ===")
print("DM statistic:", dm_stat)
print("p-value:", p_value)

# =========================================================
# 14) SAVE
# =========================================================
summary_r3.to_csv("autoformer_round3_metrics_all.csv", index=False)
summary_mean_r3.to_csv("autoformer_round3_mean.csv")
summary_std_r3.to_csv("autoformer_round3_std.csv")
preds_a_auto_r3.to_csv("autoformer_round3_predictions_scenario_a_rsi.csv", index=False)
preds_b_auto_r3.to_csv("autoformer_round3_predictions_scenario_b_rsi.csv", index=False)
preds_all_a_auto_r3.to_csv("autoformer_round3_predictions_scenario_a_all_channels.csv", index=False)
preds_all_b_auto_r3.to_csv("autoformer_round3_predictions_scenario_b_all_channels.csv", index=False)
study_a_auto.trials_dataframe().to_csv("autoformer_round3_optuna_trials_scenario_a.csv", index=False)
study_b_auto.trials_dataframe().to_csv("autoformer_round3_optuna_trials_scenario_b.csv", index=False)

print("\nAutoformer Round 3 finished successfully.")