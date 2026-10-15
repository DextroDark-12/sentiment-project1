import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/processed/features.csv", parse_dates=["date"])
feature_cols = open("data/processed/feature_cols.txt").read().split()
MIN_TRAIN = 60   # first prediction after 60 training days

def reg():   # scaler lives INSIDE the pipeline -> fit on the training window only
    return make_pipeline(StandardScaler(), Ridge(alpha=10.0))

def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000))

# ---------- walk-forward ----------
records = []
for h in range(6):
    yr, yu = f"y_ret_{h}", f"y_up_{h}"
    for i in range(MIN_TRAIN + h, len(df)):
        test = df.iloc[[i]]
        if np.isnan(test[yr].iloc[0]):
            continue                              # future not known yet
        train = df.iloc[: i - h].dropna(subset=[yr])   # only targets already known at day n
        X, Xt = train[feature_cols], test[feature_cols]

        pred_ret = reg().fit(X, train[yr]).predict(Xt)[0]
        if train[yu].nunique() == 2:
            prob_up = clf().fit(X, train[yu]).predict_proba(Xt)[0, 1]
        else:
            prob_up = float(train[yu].mean())

        cp = test["close_prev"].iloc[0]
        records.append({
            "date": test["date"].iloc[0], "h": h, "close_prev": cp,
            "y_ret": test[yr].iloc[0], "y_up": test[yu].iloc[0],
            "pred_ret": pred_ret, "prob_up": prob_up,
            "base_up": float(train[yu].mean()),          # baseline: majority class so far
            "pred_price": cp * np.exp(pred_ret),
            "actual_price": cp * np.exp(test[yr].iloc[0]),
        })

pred = pd.DataFrame(records)
pred.to_csv("data/processed/predictions.csv", index=False)

# ---------- forecast metrics per horizon ----------
rows = []
for h, g in pred.groupby("h"):
    rows.append({
        "h": h, "n": len(g),
        "MAE_ret_model": (g.pred_ret - g.y_ret).abs().mean(),
        "MAE_ret_naive": g.y_ret.abs().mean(),                        # naive = predict 0 return
        "RMSE_ret_model": np.sqrt(((g.pred_ret - g.y_ret) ** 2).mean()),
        "RMSE_ret_naive": np.sqrt((g.y_ret ** 2).mean()),
        "MAE_price_model": (g.pred_price - g.actual_price).abs().mean(),
        "MAE_price_naive": (g.close_prev - g.actual_price).abs().mean(),
        "dir_acc_ridge": (np.sign(g.pred_ret) == np.sign(g.y_ret)).mean(),
        "acc_logit": ((g.prob_up > 0.5) == (g.y_up == 1)).mean(),
        "acc_majority": ((g.base_up > 0.5) == (g.y_up == 1)).mean(),
    })
summary = pd.DataFrame(rows).round(4)
summary.to_csv("data/processed/forecast_metrics.csv", index=False)
print(summary.to_string(index=False))

# ---------- live forecast from the latest row ----------
last = df.iloc[[-1]]
live = []
for h in range(6):
    tr = df.dropna(subset=[f"y_ret_{h}"])
    r = reg().fit(tr[feature_cols], tr[f"y_ret_{h}"]).predict(last[feature_cols])[0]
    p = clf().fit(tr[feature_cols], tr[f"y_up_{h}"]).predict_proba(last[feature_cols])[0, 1]
    live.append({"h": h, "pred_ret": r, "prob_up": p,
                 "pred_price": last["close_prev"].iloc[0] * np.exp(r)})
pd.DataFrame(live).to_csv("data/processed/live_forecast.csv", index=False)
print("\nlive forecast (base = close of day before", last["date"].iloc[0].date(), ")")
print(pd.DataFrame(live).round(4).to_string(index=False))