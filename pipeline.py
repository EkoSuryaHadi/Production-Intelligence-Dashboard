"""
pipeline.py
Jalankan script ini untuk regenerate semua artifact sebelum menjalankan dashboard.
Usage: python pipeline.py
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import json
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_absolute_error, r2_score

print("=" * 55)
print("KSL FIELD — ML PIPELINE")
print("=" * 55)

# ── 1. Parse DPR ──────────────────────────────────────────────
print("\n[1/5] Loading DPR...")
df_raw = pd.read_excel("251231_DPR_PPRL_Desember_2025.xlsx", sheet_name="Data", header=None)
well_cols = {
    "KSL-1": {"online":3,"thp":5,"esp_freq":6,"gross":10,"wc":11,"net":12,"gas":13,"water":14,"gor":16},
    "KSL-3": {"online":21,"thp":23,"esp_freq":24,"gross":28,"wc":29,"net":30,"water":31,"gas":32,"gor":34},
    "KSL-4": {"online":41,"thp":43,"esp_freq":44,"gross":48,"wc":49,"net":50,"gas":51,"water":52,"gor":54},
    "KSL-5": {"online":57,"thp":59,"esp_freq":60,"gross":64,"wc":65,"net":66,"gas":67,"water":68,"gor":70},
}
data = df_raw.iloc[5:].copy()
data["date"] = pd.to_datetime(data.iloc[:, 2], errors="coerce")
data = data.dropna(subset=["date"])
records = []
for well, cols in well_cols.items():
    tmp = pd.DataFrame({"date": data["date"].values, "well": well})
    for m, c in cols.items():
        tmp[m] = pd.to_numeric(data.iloc[:, c].values, errors="coerce")
    records.append(tmp)
dpr = pd.concat(records, ignore_index=True).sort_values(["well","date"])
print(f"   DPR: {len(dpr)} rows, {dpr.date.min().date()} → {dpr.date.max().date()}")

# ── 2. Parse Sonolog ──────────────────────────────────────────
print("[2/5] Loading Sonolog...")
def parse_sono_k1():
    df = pd.read_excel("Sonoloq_PPRL.xlsx", sheet_name="KSL-01", header=None)
    d = df.iloc[8:].copy(); d.columns = range(d.shape[1])
    d["date"] = pd.to_datetime(d[4], errors="coerce")
    d["ampere"] = pd.to_numeric(d[3], errors="coerce")
    d["fl_dyn"] = pd.to_numeric(d[10], errors="coerce")
    d["pump_int"] = pd.to_numeric(d[12], errors="coerce")
    d["pbhp"] = pd.to_numeric(d[20], errors="coerce") if d.shape[1]>20 else np.nan
    d = d.dropna(subset=["date"])
    r = d.groupby("date").agg(ampere=("ampere","mean"),fl_dyn=("fl_dyn","mean"),
                               pump_int=("pump_int","mean"),pbhp=("pbhp","mean")).reset_index()
    r["well"] = "KSL-1"; return r

def parse_sono_k5():
    df = pd.read_excel("Sonoloq_PPRL.xlsx", sheet_name="KSL-5", header=None)
    d = df.iloc[6:].copy(); d.columns = range(d.shape[1])
    d["_dr"] = d[3].ffill()
    d["date"] = pd.to_datetime(d["_dr"], errors="coerce", dayfirst=True)
    d["ampere"] = pd.to_numeric(d[2], errors="coerce")
    d["fl_dyn"] = pd.to_numeric(d[6], errors="coerce")
    d["pump_int"] = pd.to_numeric(d[7], errors="coerce")
    d["pbhp"] = pd.to_numeric(d[14], errors="coerce")
    d = d.dropna(subset=["date"])
    r = d.groupby("date").agg(ampere=("ampere","mean"),fl_dyn=("fl_dyn","mean"),
                               pump_int=("pump_int","mean"),pbhp=("pbhp","mean")).reset_index()
    r["well"] = "KSL-5"; return r

sono = pd.concat([parse_sono_k1(), parse_sono_k5()], ignore_index=True)
print(f"   Sonolog: {len(sono)} rows (K1 + K5)")

# ── 3. Merge & clean ──────────────────────────────────────────
print("[3/5] Merging & cleaning...")
merged = dpr.merge(sono[["date","well","ampere","fl_dyn","pump_int","pbhp"]],
                   on=["date","well"], how="left")
merged = merged.sort_values(["well","date"]).reset_index(drop=True)
merged["is_shutdown"] = ((merged["gross"].isna())|(merged["gross"]==0)).astype(int)
num_cols = ["thp","esp_freq","gross","wc","net","gas","water","gor","ampere","fl_dyn","pump_int","pbhp","online"]
merged[num_cols] = merged.groupby("well")[num_cols].transform(lambda x: x.ffill().fillna(x.median()))
merged[num_cols] = merged[num_cols].fillna(0)
merged.to_csv("merged_clean.csv", index=False)
print(f"   Saved merged_clean.csv — {len(merged)} rows")

# ── 4. Feature engineering ────────────────────────────────────
print("[4/5] Feature engineering...")
def make_features(df):
    df = df.sort_values(["well","date"]).copy()
    g = df.groupby("well")
    for lag in [1,3,7,14]:
        df[f"gross_lag{lag}"] = g["gross"].shift(lag)
        df[f"thp_lag{lag}"]   = g["thp"].shift(lag)
        df[f"wc_lag{lag}"]    = g["wc"].shift(lag)
        df[f"esp_freq_lag{lag}"] = g["esp_freq"].shift(lag)
    for w in [7,14,30]:
        df[f"gross_roll{w}"] = g["gross"].shift(1).transform(lambda x: x.rolling(w,min_periods=1).mean())
        df[f"gross_std{w}"]  = g["gross"].shift(1).transform(lambda x: x.rolling(w,min_periods=1).std().fillna(0))
        df[f"wc_roll{w}"]    = g["wc"].shift(1).transform(lambda x: x.rolling(w,min_periods=1).mean())
    df["wor"]          = df["water"]/(df["gross"]-df["water"]+1e-6)
    df["gross_delta1"] = g["gross"].diff(1)
    df["gross_delta7"] = g["gross"].diff(7)
    df["thp_delta1"]   = g["thp"].diff(1)
    df["cum_gross"]    = g["gross"].cumsum()
    def dss(s):
        c,r=[],0
        for v in s: r2=0 if v==1 else r+1; r=r2; c.append(r2)
        return c
    df["days_since_sd"] = g["is_shutdown"].transform(dss)
    df["well_id"] = df["well"].map({"KSL-1":0,"KSL-3":1,"KSL-4":2,"KSL-5":3})
    df["month"]   = df["date"].dt.month
    df["dow"]     = df["date"].dt.dayofweek
    df["target"]  = g["gross"].shift(-1)
    return df.dropna(subset=["target","gross_lag1","gross_lag7"])

feat_df = make_features(merged)
drop_cols = ["date","well","gross","net","target","is_shutdown"]
feature_cols = [c for c in feat_df.columns if c not in drop_cols]
feat_df.to_csv("feat_df_full.csv", index=False)
with open("feature_cols.json","w") as f: json.dump(feature_cols, f)
print(f"   {len(feat_df)} rows, {len(feature_cols)} features")

# ── 5. Train & evaluate ───────────────────────────────────────
print("[5/5] Training XGBoost model...")
cutoff = feat_df["date"].max() - pd.DateOffset(months=3)
train = feat_df[feat_df["date"] <= cutoff]
test  = feat_df[feat_df["date"] >  cutoff]
X_tr, y_tr = train[feature_cols], train["target"]
X_te, y_te = test[feature_cols],  test["target"]
print(f"   Train: {train.date.min().date()} → {train.date.max().date()} ({len(train)} rows)")
print(f"   Test : {test.date.min().date()} → {test.date.max().date()} ({len(test)} rows)")

model = xgb.XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=5,
                          subsample=0.8, colsample_bytree=0.8,
                          early_stopping_rounds=20, random_state=42, verbosity=0)
model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
model.save_model("KSL_model.json")

preds = model.predict(X_te)
overall = {
    "mae": round(float(mean_absolute_error(y_te, preds)),1),
    "mape": round(float((np.abs((y_te - preds)/(y_te+1e-6)).mean())*100),1),
    "r2": round(float(r2_score(y_te, preds)),3)
}
per_well = {}
test["pred"] = preds
for w in test["well"].unique():
    wt = test[test["well"]==w]
    if len(wt) > 0:
        per_well[w] = {
            "mae": round(float(mean_absolute_error(wt["gross"], wt["pred"])),1),
            "mape": round(float((np.abs((wt["gross"]-wt["pred"])/(wt["gross"]+1e-6)).mean())*100),1),
            "r2": round(float(r2_score(wt["gross"], wt["pred"])),3)
        }
with open("metrics.json","w") as f: json.dump(per_well, f)

print(f"\n   Overall — MAE: {overall['mae']} BOPD | MAPE: {overall['mape']}% | R²: {overall['r2']}")
for w, m in per_well.items():
    print(f"   {w}: MAE={m['mae']} | MAPE={m['mape']}% | R²={m['r2']}")

print("\n" + "=" * 55)
print("Pipeline selesai. Jalankan: streamlit run dashboard.py")
print("=" * 55)
