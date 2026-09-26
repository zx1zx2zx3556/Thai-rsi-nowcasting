# =========================================================
# TimesNet Round 3 (standalone)
# Research-consistent version
# - Custom 5-fold rolling-origin split
# - LOOKBACK = 24
# - HORIZON = 16
# - Direct multi-step forecasting
# - RevIN applied instance-wise
# - Hyperparameter tuning with Optuna
# =========================================================

import math
import copy
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import optuna

from torch.utils.data import Dataset, DataLoader
from scipy.stats import t

warnings.filterwarnings("ignore")

# =========================================================
# 0) REPRODUCIBILITY
# =========================================================
SEED = 42

def reset_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

reset_seeds(SEED)
device = get_device()
print("Using device:", device)

# =========================================================
# 1) PATHS
# =========================================================
PATH_FULL_A = "/Users/methask./Retail_sales 2015 - 2023/scenario_a_full_df.csv"
PATH_FULL_B = "/Users/methask./Retail_sales 2015 - 2023/scenario_b_full_df.csv"

# =========================================================
# 2) RESEARCH SETTINGS
# =========================================================
LOOKBACK = 24
HORIZON = 16
N_TRIALS = 30

# =========================================================
# 3) METRICS
# =========================================================
def mape(y_true, y_pred, eps=1e-6):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100

def smape(y_true, y_pred, eps=1e-6):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.maximum(denom, eps)
    return np.mean(np.abs(y_true - y_pred) / denom) * 100

def dm_test(e1, e2):
    e1 = np.asarray(e1, dtype=float)
    e2 = np.asarray(e2, dtype=float)
    n = min(len(e1), len(e2))
    e1 = e1[:n]
    e2 = e2[:n]
    d = e1**2 - e2**2
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)

    if np.isnan(var_d) or var_d == 0:
        return np.nan, np.nan

    dm_stat = mean_d / np.sqrt(var_d / len(d))
    p_value = 2 * (1 - t.cdf(np.abs(dm_stat), df=len(d) - 1))
    return dm_stat, p_value

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
# 5) DATA HELPERS
# =========================================================
def ensure_required_columns(df, feature_cols):
    out = df.copy()
    if "target_raw" not in out.columns:
        out["target_raw"] = out["RSI"]

    needed = ["date", "target_raw"] + feature_cols
    missing = [c for c in needed if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return out

def load_full_data(path_a, path_b):
    df_a = pd.read_csv(path_a, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    df_b = pd.read_csv(path_b, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    df_a = ensure_required_columns(df_a, feature_cols_a)
    df_b = ensure_required_columns(df_b, feature_cols_b)

    df_a["time_idx"] = np.arange(len(df_a))
    df_b["time_idx"] = np.arange(len(df_b))
    return df_a, df_b

def make_channel_cols(feature_cols):
    return ["target_raw"] + feature_cols

def custom_rolling_origin_splits(df, initial_train_size=42, test_size=16, step_size=10, n_splits=5):
    splits = []
    n = len(df)

    for i in range(n_splits):
        train_end = initial_train_size + i * step_size
        test_start = train_end
        test_end = test_start + test_size

        if test_end > n:
            raise ValueError(f"Split {i+1} exceeds data length: test_end={test_end}, n={n}")

        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        splits.append((train_idx, test_idx))

    return splits

# =========================================================
# 6) REVIN
# =========================================================
def revin_norm_window(x_window):
    mu = np.mean(x_window, axis=0, keepdims=True)
    sd = np.std(x_window, axis=0, keepdims=True, ddof=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    x_norm = (x_window - mu) / sd
    return x_norm, mu, sd

def revin_norm_target(y_future, target_mu, target_sd):
    return (y_future - target_mu) / target_sd

def revin_denorm_target(y_pred_norm, target_mu, target_sd):
    return y_pred_norm * target_sd + target_mu

# =========================================================
# 7) DATASET
# =========================================================
class MultiStepRevINDataset(Dataset):
    def __init__(self, df, channel_cols, lookback, horizon):
        self.x = []
        self.y = []

        values = df[channel_cols].to_numpy(dtype=np.float32)
        target = df["target_raw"].to_numpy(dtype=np.float32)

        max_start = len(df) - lookback - horizon + 1
        for start in range(max_start):
            x_window = values[start:start + lookback]
            y_future = target[start + lookback:start + lookback + horizon]

            x_norm, mu, sd = revin_norm_window(x_window)
            target_mu = mu[0, 0]
            target_sd = sd[0, 0]
            y_norm = revin_norm_target(y_future, target_mu, target_sd)

            self.x.append(x_norm.astype(np.float32))
            self.y.append(y_norm.astype(np.float32))

        self.x = np.array(self.x, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.x[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32)
        )

# =========================================================
# 8) TIMESNET BLOCKS
# =========================================================
class InceptionBlockV1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6):
        super().__init__()
        kernel_sizes = [1, 3, 5, 7, 9, 11][:max(1, num_kernels)]
        self.kernels = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, k), padding=(0, k // 2))
            for k in kernel_sizes
        ])

    def forward(self, x):
        outs = [kernel(x) for kernel in self.kernels]
        return torch.mean(torch.stack(outs, dim=-1), dim=-1)

class TimesBlock(nn.Module):
    def __init__(self, seq_len, d_model, d_ff, top_k, num_kernels, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.top_k = top_k
        self.inception = nn.Sequential(
            InceptionBlockV1(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            nn.Dropout(dropout),
            InceptionBlockV1(d_ff, d_model, num_kernels=num_kernels)
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def _fft_period(self, x):
        xf = torch.fft.rfft(x, dim=1)
        freq_amp = torch.mean(torch.abs(xf), dim=(0, 2))
        freq_amp[0] = 0
        top_k = min(self.top_k, max(1, len(freq_amp) - 1))
        values, indices = torch.topk(freq_amp, k=top_k)
        periods = []
        T = x.size(1)

        for idx in indices.detach().cpu().numpy().tolist():
            p = int(np.round(T / idx)) if idx != 0 else T
            periods.append(max(1, p))

        weights = torch.softmax(values, dim=0)
        return periods, weights

    def forward(self, x):
        B, T, D = x.size()
        periods, period_weights = self._fft_period(x)

        res = []
        for period in periods:
            length = int(math.ceil(T / period) * period)

            if length > T:
                pad_x = torch.zeros(B, length - T, D, device=x.device, dtype=x.dtype)
                out = torch.cat([x, pad_x], dim=1)
            else:
                out = x

            out = out.reshape(B, length // period, period, D).permute(0, 3, 1, 2)
            out = self.inception(out)
            out = out.permute(0, 2, 3, 1).reshape(B, length, D)
            out = out[:, :T, :]
            res.append(out)

        stacked = torch.stack(res, dim=-1)
        weights = period_weights.view(1, 1, 1, -1).to(x.device)
        out = torch.sum(stacked * weights, dim=-1)
        out = self.dropout(out)
        return self.layer_norm(out + x)

class TimesNetRegressor(nn.Module):
    def __init__(self, seq_len, num_channels, horizon, d_model=64, d_ff=128, e_layers=2, top_k=3, num_kernels=6, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(num_channels, d_model)
        self.blocks = nn.ModuleList([
            TimesBlock(seq_len, d_model, d_ff, top_k, num_kernels, dropout)
            for _ in range(e_layers)
        ])
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(seq_len * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon)
        )

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)

# =========================================================
# 9) TRAINING
# =========================================================
def train_model(train_df, channel_cols, lookback, horizon, params, device):
    reset_seeds(SEED)

    dataset = MultiStepRevINDataset(train_df, channel_cols, lookback, horizon)
    if len(dataset) == 0:
        raise ValueError(f"Not enough training observations for lookback={lookback}, horizon={horizon}")

    loader = DataLoader(dataset, batch_size=int(params["batch_size"]), shuffle=False, num_workers=0)

    model = TimesNetRegressor(
        seq_len=lookback,
        num_channels=len(channel_cols),
        horizon=horizon,
        d_model=int(params["d_model"]),
        d_ff=int(params["d_ff"]),
        e_layers=int(params["e_layers"]),
        top_k=int(params["top_k"]),
        num_kernels=int(params["num_kernels"]),
        dropout=float(params["dropout"])
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"])
    )
    criterion = nn.L1Loss()

    best_state = copy.deepcopy(model.state_dict())
    best_loss = np.inf
    patience_counter = 0

    for epoch in range(int(params["max_epochs"])):
        model.train()
        losses = []

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            losses.append(loss.item())

        epoch_loss = float(np.mean(losses))
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= int(params["patience"]):
            break

    model.load_state_dict(best_state)
    model.eval()
    return model

@torch.no_grad()
def predict_direct_multistep(model, history_df, channel_cols, lookback, horizon, device):
    x_window = history_df[channel_cols].tail(lookback).to_numpy(dtype=np.float32)
    x_norm, mu, sd = revin_norm_window(x_window)

    target_mu = mu[0, 0]
    target_sd = sd[0, 0]

    x_tensor = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0).to(device)
    pred_norm = model(x_tensor).detach().cpu().numpy().reshape(-1)

    pred_raw = revin_denorm_target(pred_norm, target_mu, target_sd)
    return pred_raw[:horizon]

def direct_predict_split(train_df, test_df, feature_cols, lookback, horizon, params, device):
    channel_cols = make_channel_cols(feature_cols)

    model = train_model(
        train_df=train_df,
        channel_cols=channel_cols,
        lookback=lookback,
        horizon=horizon,
        params=params,
        device=device
    )

    preds_raw = predict_direct_multistep(
        model=model,
        history_df=train_df,
        channel_cols=channel_cols,
        lookback=lookback,
        horizon=horizon,
        device=device
    )

    actual_df = test_df.head(horizon).copy().reset_index(drop=True)
    preds_df = pd.DataFrame({
        "date": actual_df["date"],
        "actual": actual_df["target_raw"].to_numpy(dtype=float),
        "pred": preds_raw[:len(actual_df)]
    })
    preds_df["abs_error"] = np.abs(preds_df["actual"] - preds_df["pred"])
    return preds_df

# =========================================================
# 10) LOAD DATA
# =========================================================
df_a, df_b = load_full_data(PATH_FULL_A, PATH_FULL_B)

# =========================================================
# 11) OPTUNA OBJECTIVE
# =========================================================
def objective_factory(df, feature_cols, scenario_name):
    def objective(trial):
        params = {
            "d_model": trial.suggest_categorical("d_model", [32, 64, 128]),
            "d_ff": trial.suggest_categorical("d_ff", [64, 128, 256]),
            "e_layers": trial.suggest_int("e_layers", 1, 3),
            "top_k": trial.suggest_categorical("top_k", [2, 3, 4, 5]),
            "num_kernels": trial.suggest_categorical("num_kernels", [3, 4, 6]),
            "dropout": trial.suggest_float("dropout", 0.05, 0.30),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
            "max_epochs": trial.suggest_categorical("max_epochs", [60, 80, 100]),
            "patience": trial.suggest_categorical("patience", [8, 10, 12, 15]),
        }

        splits = custom_rolling_origin_splits(
            df=df,
            initial_train_size=42,
            test_size=16,
            step_size=10,
            n_splits=5
        )

        fold_maes = []

        for fold, (train_idx, test_idx) in enumerate(splits, start=1):
            train_df = df.iloc[train_idx].copy().reset_index(drop=True)
            test_df = df.iloc[test_idx].copy().reset_index(drop=True)

            preds_df = direct_predict_split(
                train_df=train_df,
                test_df=test_df,
                feature_cols=feature_cols,
                lookback=LOOKBACK,
                horizon=HORIZON,
                params=params,
                device=device
            )

            fold_mae = np.mean(np.abs(preds_df["actual"] - preds_df["pred"]))
            if np.isnan(fold_mae) or np.isinf(fold_mae):
                fold_mae = 1e9
            fold_maes.append(fold_mae)

        return float(np.mean(fold_maes))
    return objective

# =========================================================
# 12) RUN OPTUNA
# =========================================================
study_a_timesnet = optuna.create_study(direction="minimize", study_name="timesnet_scenario_a")
study_a_timesnet.optimize(objective_factory(df_a, feature_cols_a, "Scenario A"), n_trials=N_TRIALS)

study_b_timesnet = optuna.create_study(direction="minimize", study_name="timesnet_scenario_b")
study_b_timesnet.optimize(objective_factory(df_b, feature_cols_b, "Scenario B"), n_trials=N_TRIALS)

print("\n=== TimesNet Round 3 Best Params Scenario A ===")
print(study_a_timesnet.best_params)
print("Best MAE:", study_a_timesnet.best_value)

print("\n=== TimesNet Round 3 Best Params Scenario B ===")
print(study_b_timesnet.best_params)
print("Best MAE:", study_b_timesnet.best_value)

# =========================================================
# 13) FINAL EVALUATION WITH BEST PARAMS
# =========================================================
def run_timesnet_best_model(df, feature_cols, scenario_name, best_params):
    splits = custom_rolling_origin_splits(
        df=df,
        initial_train_size=42,
        test_size=16,
        step_size=10,
        n_splits=5
    )

    metrics_list = []
    preds_list = []

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        train_df = df.iloc[train_idx].copy().reset_index(drop=True)
        test_df = df.iloc[test_idx].copy().reset_index(drop=True)

        preds_df = direct_predict_split(
            train_df=train_df,
            test_df=test_df,
            feature_cols=feature_cols,
            lookback=LOOKBACK,
            horizon=HORIZON,
            params=best_params,
            device=device
        )

        metrics_list.append({
            "scenario": scenario_name,
            "fold": fold,
            "train_size": len(train_df),
            "test_size": len(test_df),
            "MAE": np.mean(np.abs(preds_df["actual"] - preds_df["pred"])),
            "MAPE": mape(preds_df["actual"], preds_df["pred"]),
            "sMAPE": smape(preds_df["actual"], preds_df["pred"])
        })

        preds_df["scenario"] = scenario_name
        preds_df["fold"] = fold
        preds_list.append(preds_df)

    return pd.DataFrame(metrics_list), pd.concat(preds_list, ignore_index=True)

metrics_a_timesnet_r3, preds_a_timesnet_r3 = run_timesnet_best_model(
    df_a, feature_cols_a, "Scenario A", study_a_timesnet.best_params
)
metrics_b_timesnet_r3, preds_b_timesnet_r3 = run_timesnet_best_model(
    df_b, feature_cols_b, "Scenario B", study_b_timesnet.best_params
)

summary_r3 = pd.concat([metrics_a_timesnet_r3, metrics_b_timesnet_r3], ignore_index=True)
summary_mean_r3 = summary_r3.groupby("scenario")[["MAE", "MAPE", "sMAPE"]].mean()
summary_std_r3 = summary_r3.groupby("scenario")[["MAE", "MAPE", "sMAPE"]].std()

print("\n=== TimesNet Round 3 Mean ===")
print(summary_mean_r3)

print("\n=== TimesNet Round 3 Std ===")
print(summary_std_r3)

# =========================================================
# 14) DM TEST
# =========================================================
dm_df = preds_a_timesnet_r3.merge(
    preds_b_timesnet_r3,
    on="date",
    suffixes=("_a", "_b")
).sort_values("date")

err_a = dm_df["actual_a"] - dm_df["pred_a"]
err_b = dm_df["actual_b"] - dm_df["pred_b"]

dm_stat, p_value = dm_test(err_a.values, err_b.values)

print("\n=== TimesNet Round 3 DM Test ===")
print("DM statistic:", dm_stat)
print("p-value:", p_value)

# =========================================================
# 15) SAVE
# =========================================================
summary_r3.to_csv("timesnet_round3_metrics_all.csv", index=False)
summary_mean_r3.to_csv("timesnet_round3_mean.csv")
summary_std_r3.to_csv("timesnet_round3_std.csv")
preds_a_timesnet_r3.to_csv("timesnet_round3_predictions_scenario_a.csv", index=False)
preds_b_timesnet_r3.to_csv("timesnet_round3_predictions_scenario_b.csv", index=False)
study_a_timesnet.trials_dataframe().to_csv("timesnet_round3_optuna_trials_scenario_a.csv", index=False)
study_b_timesnet.trials_dataframe().to_csv("timesnet_round3_optuna_trials_scenario_b.csv", index=False)

best_params_table = pd.DataFrame([
    {"scenario": "Scenario A", **study_a_timesnet.best_params, "Best value": study_a_timesnet.best_value},
    {"scenario": "Scenario B", **study_b_timesnet.best_params, "Best value": study_b_timesnet.best_value},
])
best_params_table.to_csv("timesnet_round3_best_params.csv", index=False)

print("\nTimesNet Round 3 finished successfully.")

# =========================================================
# 16) PLOTS
# =========================================================
def plot_metric_by_fold(metrics_df, metric_name):
    plt.figure(figsize=(8, 5))
    for scenario in metrics_df["scenario"].unique():
        tmp = metrics_df[metrics_df["scenario"] == scenario].copy()
        plt.plot(tmp["fold"], tmp[metric_name], marker="o", linewidth=2, label=scenario)
        for _, row in tmp.iterrows():
            plt.text(row["fold"], row[metric_name], f'{row[metric_name]:.2f}', ha="center", va="bottom")

    plt.title(f"TimesNet Round 3: {metric_name} by Fold")
    plt.xlabel("Fold")
    plt.ylabel(metric_name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"timesnet_round3_{metric_name.lower()}_by_fold.png", dpi=300, bbox_inches="tight")
    plt.show()

plot_metric_by_fold(summary_r3, "MAE")
plot_metric_by_fold(summary_r3, "MAPE")
plot_metric_by_fold(summary_r3, "sMAPE")

mean_a = metrics_a_timesnet_r3[["MAE", "MAPE", "sMAPE"]].mean()
mean_b = metrics_b_timesnet_r3[["MAE", "MAPE", "sMAPE"]].mean()

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
plt.title("TimesNet Round 3: Mean Forecasting Error Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("timesnet_round3_mean_metrics_line.png", dpi=300, bbox_inches="tight")
plt.show()

std_a = metrics_a_timesnet_r3[["MAE", "MAPE", "sMAPE"]].std()
std_b = metrics_b_timesnet_r3[["MAE", "MAPE", "sMAPE"]].std()

scenario_a_std = [std_a["MAE"], std_a["MAPE"], std_a["sMAPE"]]
scenario_b_std = [std_b["MAE"], std_b["MAPE"], std_b["sMAPE"]]

plt.figure(figsize=(8, 5))
plt.errorbar(x, scenario_a_mean, yerr=scenario_a_std, marker="o", linewidth=2, capsize=5, label="Scenario A")
plt.errorbar(x, scenario_b_mean, yerr=scenario_b_std, marker="o", linewidth=2, capsize=5, label="Scenario B")
plt.xticks(x, metric_names)
plt.ylabel("Mean Error Value")
plt.title("TimesNet Round 3: Mean ± Standard Deviation of Forecast Errors")
plt.legend()
plt.tight_layout()
plt.savefig("timesnet_round3_mean_std_metrics_line.png", dpi=300, bbox_inches="tight")
plt.show()

vals_a = np.array(scenario_a_mean, dtype=float)
vals_b = np.array(scenario_b_mean, dtype=float)
all_vals = np.vstack([vals_a, vals_b])
norm_vals = (all_vals - all_vals.min(axis=0)) / (all_vals.max(axis=0) - all_vals.min(axis=0) + 1e-12)

plt.figure(figsize=(8, 5))
plt.plot(x, norm_vals[0], marker="o", linewidth=2, label="Scenario A")
plt.plot(x, norm_vals[1], marker="o", linewidth=2, label="Scenario B")
for i, v in enumerate(norm_vals[0]):
    plt.text(i, v, f"{v:.2f}", ha="center", va="bottom")
for i, v in enumerate(norm_vals[1]):
    plt.text(i, v, f"{v:.2f}", ha="center", va="bottom")
plt.xticks(x, metric_names)
plt.ylabel("Normalized Error (0–1)")
plt.title("TimesNet Round 3: Normalized Mean Error Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("timesnet_round3_normalized_mean_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

dm_plot_df = dm_df.copy()
loss_diff = (dm_plot_df["actual_a"] - dm_plot_df["pred_a"])**2 - (dm_plot_df["actual_b"] - dm_plot_df["pred_b"])**2
cum_diff = np.cumsum(loss_diff)

plt.figure(figsize=(10, 5))
plt.plot(dm_plot_df["date"], loss_diff, marker="o")
plt.axhline(0, linestyle="--")
plt.title("TimesNet Round 3: Loss Differential (Scenario A vs Scenario B)")
plt.xlabel("Date")
plt.ylabel("Squared Error Difference")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("timesnet_round3_loss_differential.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(dm_plot_df["date"], cum_diff)
plt.axhline(0, linestyle="--")
plt.title("TimesNet Round 3: Cumulative Loss Differential")
plt.xlabel("Date")
plt.ylabel("Cumulative Difference")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("timesnet_round3_cumulative_loss_differential.png", dpi=300, bbox_inches="tight")
plt.show()

try:
    from optuna.visualization.matplotlib import plot_optimization_history

    plot_optimization_history(study_a_timesnet)
    plt.title("TimesNet Round 3: Optuna Optimization History - Scenario A")
    plt.tight_layout()
    plt.savefig("timesnet_round3_optuna_history_scenario_a.png", dpi=300, bbox_inches="tight")
    plt.show()

    plot_optimization_history(study_b_timesnet)
    plt.title("TimesNet Round 3: Optuna Optimization History - Scenario B")
    plt.tight_layout()
    plt.savefig("timesnet_round3_optuna_history_scenario_b.png", dpi=300, bbox_inches="tight")
    plt.show()
except Exception as e:
    print("Optuna optimization history plot skipped:", e)

try:
    from optuna.visualization.matplotlib import plot_param_importances

    plot_param_importances(study_a_timesnet)
    plt.title("TimesNet Round 3: Parameter Importance - Scenario A")
    plt.tight_layout()
    plt.savefig("timesnet_round3_param_importance_scenario_a.png", dpi=300, bbox_inches="tight")
    plt.show()

    plot_param_importances(study_b_timesnet)
    plt.title("TimesNet Round 3: Parameter Importance - Scenario B")
    plt.tight_layout()
    plt.savefig("timesnet_round3_param_importance_scenario_b.png", dpi=300, bbox_inches="tight")
    plt.show()
except Exception as e:
    print("Optuna parameter importance plot skipped:", e)

print("\nAll TimesNet Round 3 plots finished successfully.")