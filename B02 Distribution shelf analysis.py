import numpy as np
import pandas as pd

# =========================================================
# 1) LOAD PREDICTIONS
# =========================================================
df = pd.read_csv("tft_round5_predictions_all.csv", parse_dates=["date"])

# ถ้าต้องการวิเคราะห์เฉพาะ Best Scenario จาก Round 5
# เช่น Scenario E (Combined)
df = df[df["scenario"] == "Scenario E (Combined)"].copy()

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
# 3) DEFINE ECONOMIC REGIMES
# =========================================================
periods = {
    "Pre-COVID": ("2015-01-01", "2019-12-31"),
    "COVID Crisis": ("2020-01-01", "2022-12-31"),
    "Recovery": ("2023-01-01", df["date"].max())
}

# =========================================================
# 4) CALCULATE METRICS BY PERIOD
# =========================================================
results = []

for regime, (start_date, end_date) in periods.items():
    temp = df[(df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))].copy()

    if len(temp) == 0:
        continue

    mae_val = np.mean(np.abs(temp["actual"] - temp["pred"]))
    mape_val = mape(temp["actual"], temp["pred"])
    smape_val = smape(temp["actual"], temp["pred"])

    results.append({
        "Period": regime,
        "Start": pd.to_datetime(start_date).date(),
        "End": pd.to_datetime(end_date).date(),
        "N": len(temp),
        "MAE": mae_val,
        "MAPE": mape_val,
        "sMAPE": smape_val
    })

results_df = pd.DataFrame(results)

# =========================================================
# 5) SHOW + SAVE
# =========================================================
print("\n=== Distribution Shift Analysis ===")
print(results_df)

results_df.to_csv("tft_round5_distribution_shift_analysis.csv", index=False)
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(results_df["Period"], results_df["MAPE"], marker="o", linewidth=2)
plt.ylabel("MAPE (%)")
plt.title("TFT Distribution Shift Analysis by Economic Regime")
plt.tight_layout()
plt.savefig("tft_round5_distribution_shift_mape.png", dpi=300, bbox_inches="tight")
plt.show()