import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_error
import numpy as np

from scipy.stats import t
#########################################################################################   Round 1 ############################################################################################
# ============================
# Load data
# ============================

df_a = pd.read_csv(
'/Users/methask./Retail_sales 2015 - 2023/final_preprocessing_export/scenario_a/python_ready_scenario_a.csv',
parse_dates=["date"]
)

df_b = pd.read_csv(
'/Users/methask./Retail_sales 2015 - 2023/final_preprocessing_export/scenario_b/python_ready_scenario_b.csv',
parse_dates=["date"]
)

# ============================
# Metrics
# ============================

def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def smape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(
        2 * np.abs(y_pred - y_true) /
        (np.abs(y_true) + np.abs(y_pred))
    ) * 100

# ============================
# Feature columns
# ============================

feature_cols_a = [
"RSI_lag1","RSI_lag2","RSI_lag3",
"PCI_lag1","PCI_lag2","PCI_lag3",
"CPI_lag1","CPI_lag2","CPI_lag3",
"Discount_rm3_lag1","Discount_rm3_lag2","Discount_rm3_lag3",
"Promotion_rm3_lag1","Promotion_rm3_lag2","Promotion_rm3_lag3",
"OnlineBuy_rm3_lag1","OnlineBuy_rm3_lag2","OnlineBuy_rm3_lag3",
"Coupon_rm3_lag1","Coupon_rm3_lag2","Coupon_rm3_lag3",
"Gift_rm3_lag1","Gift_rm3_lag2","Gift_rm3_lag3"
]

feature_cols_b = [
"RSI_lag1","RSI_lag2","RSI_lag3",
"PCI_lag1","PCI_lag2","PCI_lag3",
"CPI_lag1","CPI_lag2","CPI_lag3",
"Discount_lag4","Discount_lag8","Discount_lag12",
"Promotion_lag4","Promotion_lag8","Promotion_lag12",
"OnlineBuy_lag4","OnlineBuy_lag8","OnlineBuy_lag12",
"Coupon_lag4","Coupon_lag8","Coupon_lag12",
"Gift_lag4","Gift_lag8","Gift_lag12"
]

# ============================
# Run model across splits
# ============================

def run_model(df, feature_cols):

    results = []
    preds_all = []

    splits = sorted(df["split_id"].unique())

    for split_id in splits:

        split_df = df[df["split_id"] == split_id]

        train_df = split_df[split_df["set"] == "train"]
        test_df  = split_df[split_df["set"] == "test"]

        X_train = train_df[feature_cols]
        y_train = train_df["target_norm"]

        X_test = test_df[feature_cols]

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred_norm = model.predict(X_test)

        y_pred_raw = (
            y_pred_norm * test_df["target_sd"].values
            + test_df["target_mean"].values
        )

        y_true = test_df["target_raw"].values

        mae = mean_absolute_error(y_true, y_pred_raw)
        mape_val = mape(y_true, y_pred_raw)
        smape_val = smape(y_true, y_pred_raw)

        results.append({
            "split": split_id,
            "MAE": mae,
            "MAPE": mape_val,
            "sMAPE": smape_val
        })

        pred_df = pd.DataFrame({
            "split": split_id,
            "date": test_df["date"],
            "actual": y_true,
            "pred": y_pred_raw
        })

        preds_all.append(pred_df)

    return pd.DataFrame(results), pd.concat(preds_all)


metrics_a, preds_a = run_model(df_a, feature_cols_a)
metrics_b, preds_b = run_model(df_b, feature_cols_b)

print(metrics_a)
print(metrics_b)

# ============================
# Summary
# ============================

print("\nScenario A mean metrics")
print(metrics_a.mean())

print("\nScenario B mean metrics")
print(metrics_b.mean())

# ============================
# Plot metric comparison
# ============================

plt.figure(figsize=(8,5))

plt.plot(metrics_a["split"], metrics_a["MAE"], label="Scenario A")
plt.plot(metrics_b["split"], metrics_b["MAE"], label="Scenario B")

plt.xlabel("Split")
plt.ylabel("MAE")
plt.title("MAE comparison")
plt.legend()
plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

mae_a = metrics_a["MAE"].mean()
mape_a = metrics_a["MAPE"].mean()
smape_a = metrics_a["sMAPE"].mean()

mae_b = metrics_b["MAE"].mean()
mape_b = metrics_b["MAPE"].mean()
smape_b = metrics_b["sMAPE"].mean()

metric_names = ["MAE", "MAPE", "sMAPE"]
scenario_a_values = [mae_a, mape_a, smape_a]
scenario_b_values = [mae_b, mape_b, smape_b]

x = np.arange(len(metric_names))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 6))

bars1 = ax.bar(x - width/2, scenario_a_values, width, label="Scenario A")
bars2 = ax.bar(x + width/2, scenario_b_values, width, label="Scenario B")

ax.set_xticks(x)
ax.set_xticklabels(metric_names)
ax.set_ylabel("Value")
ax.set_title("Comparison of Forecasting Error Metrics")
ax.legend()

# ใส่ค่าบนแท่ง
for bar in bars1:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width()/2,
        height,
        f"{height:.2f}",
        ha="center",
        va="bottom"
    )

for bar in bars2:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width()/2,
        height,
        f"{height:.2f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.show()
# ============================
# Diebold-Mariano Test
# ============================

def dm_test(e1, e2):

    d = e1**2 - e2**2
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)

    DM = mean_d / np.sqrt(var_d/len(d))

    p_value = 2 * (1 - t.cdf(np.abs(DM), df=len(d)-1))

    return DM, p_value



# errors
err_a = preds_a["actual"] - preds_a["pred"]
err_b = preds_b["actual"] - preds_b["pred"]

DM_stat, p_val = dm_test(err_a.values, err_b.values)

print("\nDiebold-Mariano Test")
print("DM statistic:", DM_stat)
print("p-value:", p_val)


def tolerance_accuracy(y_true, y_pred, tolerance=0.05):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    pct_error = np.abs((y_true - y_pred) / y_true)
    return np.mean(pct_error <= tolerance) * 100

# Scenario A
r2_a = r2_score(preds_a["actual"], preds_a["pred"])
acc5_a = tolerance_accuracy(preds_a["actual"], preds_a["pred"], tolerance=0.05)
acc10_a = tolerance_accuracy(preds_a["actual"], preds_a["pred"], tolerance=0.10)

# Scenario B
r2_b = r2_score(preds_b["actual"], preds_b["pred"])
acc5_b = tolerance_accuracy(preds_b["actual"], preds_b["pred"], tolerance=0.05)
acc10_b = tolerance_accuracy(preds_b["actual"], preds_b["pred"], tolerance=0.10)

print("\nScenario A")
print("R² =", round(r2_a, 4))
print("Accuracy within 5% =", round(acc5_a, 2), "%")
print("Accuracy within 10% =", round(acc10_a, 2), "%")

print("\nScenario B")
print("R² =", round(r2_b, 4))
print("Accuracy within 5% =", round(acc5_b, 2), "%")
print("Accuracy within 10% =", round(acc10_b, 2), "%")
#########################################################################################   Round 1 ############################################################################################

