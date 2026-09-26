import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import random
import warnings
import numpy as np
import pandas as pd

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
INPUT_SIZE = 24   # baseline ที่รันได้จริง และเชื่อมกับ Round 4
VAL_SIZE = 16

# Round 1 = baseline hold-out evaluation
# คงกรอบเดิมให้ใกล้เคียงงานก่อนหน้า โดยไม่ทำ tuning
TEST_SIZE = 16

# =========================================================
# 5) BALANCED WIDE DF
# =========================================================
def make_balanced_wide_df(df, channels, scenario_name):
    cols = ["date"] + channels
    out = df[cols].copy()

    out = out.dropna(subset=channels).copy()
    out = out.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)

    if out.empty:
        raise ValueError(f"{scenario_name}: no data left after balancing.")

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
def prepare_autoformer_panel(df, channels):
    panel_list = []

    for col in channels:
        tmp = df[["date", col]].copy()
        tmp["unique_id"] = col
        tmp["ds"] = pd.to_datetime(tmp["date"])
        tmp["y"] = pd.to_numeric(tmp[col], errors="coerce")
        tmp = tmp[["unique_id", "ds", "y"]]
        panel_list.append(tmp)

    panel_df = pd.concat(panel_list, ignore_index=True)
    panel_df = panel_df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    panel_df = panel_df.dropna().reset_index(drop=True)
    return panel_df

panel_a = prepare_autoformer_panel(df_a_bal, scenario_a_channels)
panel_b = prepare_autoformer_panel(df_b_bal, scenario_b_channels)

# =========================================================
# 7) VALIDATE PANEL
# =========================================================
def validate_panel(panel_df, scenario_name, input_size, h, test_size, val_size):
    lengths = panel_df.groupby("unique_id").size()
    min_len = lengths.min()

    required_min = input_size + h + test_size + val_size

    print(f"\n[{scenario_name}] Min channel length: {min_len}")
    print(f"[{scenario_name}] Required minimum length: {required_min}")

    if min_len < required_min:
        raise ValueError(
            f"{scenario_name}: insufficient sequence length. "
            f"Minimum channel length = {min_len}, but required >= {required_min}."
        )

validate_panel(panel_a, "Scenario A", INPUT_SIZE, H, TEST_SIZE, VAL_SIZE)
validate_panel(panel_b, "Scenario B", INPUT_SIZE, H, TEST_SIZE, VAL_SIZE)

# =========================================================
# 8) BUILD MODEL
# Baseline / Default Architecture
# =========================================================
def build_autoformer_model():
    model = Autoformer(
        h=H,
        input_size=INPUT_SIZE,
        hidden_size=64,
        dropout=0.05,
        factor=3,
        n_head=4,
        conv_hidden_size=32,
        activation="gelu",
        encoder_layers=2,
        decoder_layers=1,
        MovingAvg_window=25,
        learning_rate=1e-4,
        max_steps=300,
        val_check_steps=50,
        early_stop_patience_steps=10,
        batch_size=32,
        start_padding_enabled=True,
        random_seed=SEED,
        accelerator="cpu",
        devices=1,
        enable_progress_bar=True
    )
    return model

# =========================================================
# 9) HOLD-OUT FIT + PREDICT
# ใช้การทดสอบแบบ baseline ด้วย 1 split ท้ายชุดข้อมูล
# =========================================================
def fit_predict_autoformer_holdout(panel_df, scenario_name):
    reset_seeds(SEED)

    model = build_autoformer_model()
    nf = NeuralForecast(models=[model], freq="MS")

    cv_df = nf.cross_validation(
        df=panel_df,
        n_windows=1,
        step_size=TEST_SIZE,
        val_size=VAL_SIZE,
        refit=False,
        verbose=0
    )

    pred_col = [c for c in cv_df.columns if c not in ["unique_id", "ds", "cutoff", "y"]][0]

    result = cv_df.copy()
    result.rename(columns={"ds": "date", "y": "actual", pred_col: "pred"}, inplace=True)
    result["scenario"] = scenario_name
    result["abs_error"] = np.abs(result["actual"] - result["pred"])

    rsi_result = result[result["unique_id"] == "RSI"].copy().sort_values("date")

    metrics_df = pd.DataFrame([{
        "scenario": scenario_name,
        "test_size": len(rsi_result),
        "MAE": mean_absolute_error(rsi_result["actual"], rsi_result["pred"]),
        "MAPE": mape(rsi_result["actual"], rsi_result["pred"]),
        "sMAPE": smape(rsi_result["actual"], rsi_result["pred"])
    }])

    return metrics_df, rsi_result, result

# =========================================================
# 10) RUN SCENARIO A / B
# =========================================================
metrics_a_auto_r1, preds_a_auto_r1, preds_all_a_auto_r1 = fit_predict_autoformer_holdout(
    panel_df=panel_a,
    scenario_name="Scenario A"
)

metrics_b_auto_r1, preds_b_auto_r1, preds_all_b_auto_r1 = fit_predict_autoformer_holdout(
    panel_df=panel_b,
    scenario_name="Scenario B"
)

# =========================================================
# 11) SUMMARY
# =========================================================
summary_r1 = pd.concat([metrics_a_auto_r1, metrics_b_auto_r1], ignore_index=True)

print("\n=== Autoformer Round 1 Metrics ===")
print(summary_r1)

# =========================================================
# 12) DIEBOLD-MARIANO TEST
# =========================================================
dm_df = preds_a_auto_r1.merge(
    preds_b_auto_r1,
    on=["date"],
    suffixes=("_a", "_b")
).sort_values("date")

err_a = dm_df["actual_a"] - dm_df["pred_a"]
err_b = dm_df["actual_b"] - dm_df["pred_b"]

dm_stat, p_value = dm_test(err_a.values, err_b.values)

print("\n=== Autoformer Round 1 DM Test ===")
print("DM statistic:", dm_stat)
print("p-value:", p_value)

# =========================================================
# 13) SAVE
# =========================================================
summary_r1.to_csv("autoformer_round1_metrics_all.csv", index=False)
preds_a_auto_r1.to_csv("autoformer_round1_predictions_scenario_a_rsi.csv", index=False)
preds_b_auto_r1.to_csv("autoformer_round1_predictions_scenario_b_rsi.csv", index=False)
preds_all_a_auto_r1.to_csv("autoformer_round1_predictions_scenario_a_all_channels.csv", index=False)
preds_all_b_auto_r1.to_csv("autoformer_round1_predictions_scenario_b_all_channels.csv", index=False)

print("\nAutoformer Round 1 finished successfully.")