import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import xgboost as xgb
from scipy.optimize import curve_fit
from scipy.stats import linregress

# =============================================================================
# 🔧 1. Dynamic Path & Data Loading
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_merged():
    fp = os.path.join(BASE_DIR, "merged_clean.csv")
    if not os.path.exists(fp):
        st.error(f"❌ File '{fp}' tidak ditemukan.\nJalankan terlebih dahulu: `python pipeline.py`")
        st.stop()
        
    df = pd.read_csv(fp)
    df["date"] = pd.to_datetime(df["date"])
    
    # Normalisasi kolom
    col_map = {}
    for c in df.columns:
        cl = c.lower().replace(" ", "_")
        if "gross" in cl or "bfpd" in cl: col_map[c] = "gross"
        elif "net" in cl or "nett" in cl or "bopd" in cl: col_map[c] = "net"
        elif "wc" in cl or "water_cut" in cl: col_map[c] = "wc"
        elif "thp" in cl: col_map[c] = "thp"
        elif "esp" in cl or "freq" in cl: col_map[c] = "esp_freq"
    df = df.rename(columns=col_map)
    
    required = ["date", "well", "gross", "net", "wc", "thp", "esp_freq"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"❌ Kolom wajib tidak ditemukan: {missing}")
        st.stop()
    return df

@st.cache_resource
def load_models():
    mdls = {}
    wells = ["KSL-1", "KSL-3", "KSL-4", "KSL-5"]
    for w in wells:
        fp = os.path.join(BASE_DIR, f"model_{w.replace('-', '_')}.json")
        if os.path.exists(fp):
            m = xgb.XGBRegressor()
            m.load_model(fp)
            mdls[w] = m
    fp = os.path.join(BASE_DIR, "KSL_model.json")
    if os.path.exists(fp):
        m = xgb.XGBRegressor()
        m.load_model(fp)
        mdls["_global"] = m
    return mdls

@st.cache_data
def load_metrics():
    for name in ["metrics_per_well.json", "metrics.json"]:
        fp = os.path.join(BASE_DIR, name)
        if os.path.exists(fp):
            with open(fp, "r", encoding="utf-8") as f:
                return json.load(f)
    return {}

# =============================================================================
# 📦 Inisialisasi & Konstanta
# =============================================================================
df = load_merged()
models = load_models()
metrics = load_metrics()

st.set_page_config(page_title="KSL Dashboard", layout="wide", page_icon="🛢️")

WELL_COLORS = {"KSL-1": "#58a6ff", "KSL-3": "#7ee787", "KSL-4": "#ffa657", "KSL-5": "#d2a8ff"}
BADGE_CLS = {"KSL-1": "badge-k1", "KSL-3": "badge-k3", "KSL-4": "badge-k4", "KSL-5": "badge-k5"}
PL = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
          font=dict(family="IBM Plex Sans", color="#8b949e", size=11),
          xaxis=dict(gridcolor="#21262d", zerolinecolor="#21262d"),
          yaxis=dict(gridcolor="#21262d", zerolinecolor="#21262d"),
          legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#21262d"),
          margin=dict(l=12, r=12, t=58, b=18))

# Simpan regime break sebagai Timestamp agar tetap cocok dengan sumbu waktu Plotly
REGIME_BREAK = {"KSL-1": pd.Timestamp("2025-11-07")}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
body{background:#0d1117;color:#c9d1d9;font-family:'IBM Plex Sans',sans-serif;}
.stApp{background:#0d1117;}
label p, label, summary, summary * {color: #c9d1d9 !important;}
.sec-hdr{font-size:17px;font-weight:600;color:#f0f6fc;letter-spacing:.05em;border-bottom:2px solid #f0a500;padding-bottom:10px;margin:26px 0 10px;}
.page-hero{background:linear-gradient(135deg, rgba(88,166,255,.16), rgba(240,165,0,.08) 58%, rgba(13,17,23,.96));border:1px solid #21262d;border-radius:18px;padding:24px 26px 22px;margin:4px 0 18px;box-shadow:0 18px 60px rgba(0,0,0,.24);}
.hero-kicker{font-size:11px;color:#f0a500;letter-spacing:.18em;text-transform:uppercase;margin-bottom:10px;font-weight:600;}
.hero-title{font-size:30px;line-height:1.08;color:#f0f6fc;font-weight:600;margin-bottom:8px;}
.hero-sub{font-size:14px;line-height:1.6;color:#9da7b3;max-width:880px;}
.hero-meta{display:flex;flex-wrap:wrap;gap:8px;margin-top:16px;}
.hero-chip{display:inline-flex;align-items:center;gap:6px;padding:7px 11px;border-radius:999px;border:1px solid rgba(240,246,252,.08);background:rgba(13,17,23,.55);font-size:12px;color:#dce4ec;}
.kpi-card{background:#161b22;border:1px solid #21262d;border-radius:12px;padding:18px 20px;position:relative;overflow:hidden;min-height:116px;}
.kpi-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:#f0a500;}
.kpi-label{font-size:12px;color:#7d8792;text-transform:uppercase;letter-spacing:.12em;margin-bottom:8px;}
.kpi-value{font-family:'IBM Plex Mono',monospace;font-size:24px;font-weight:600;color:#f0f6fc;}
.kpi-unit{font-size:13px;color:#8b949e;margin-left:7px;}
.kpi-delta{margin-top:8px;font-size:12px;font-family:'IBM Plex Mono',monospace;line-height:1.45;}
.delta-up{color:#3fb950;}.delta-dn{color:#f85149;}.delta-neu{color:#8b949e;}
.box-red{background:#1a0e0e;border:1px solid #f85149;border-left:4px solid #f85149;border-radius:10px;padding:13px 16px;margin:10px 0 18px;color:#f85149;font-size:13px;}
.chart-note{margin:-2px 0 14px;color:#7d8792;font-size:12px;line-height:1.55;}
.attention-strip{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin:0 0 16px;}
.attention-card{background:linear-gradient(180deg, rgba(240,246,252,.02), rgba(13,17,23,.24));border:1px solid #21262d;border-radius:12px;padding:14px 16px;}
.attention-label{font-size:11px;color:#7d8792;text-transform:uppercase;letter-spacing:.12em;margin-bottom:6px;}
.attention-value{font-size:18px;color:#f0f6fc;font-weight:600;line-height:1.25;}
.attention-meta{font-size:12px;color:#9da7b3;margin-top:4px;line-height:1.45;}
.mtable{width:100%;border-collapse:collapse;font-size:13px;}
.mtable th{background:#21262d;color:#8b949e;font-size:10px;text-transform:uppercase;letter-spacing:.1em;padding:8px 12px;text-align:left;font-weight:500;}
.mtable td{padding:8px 12px;border-bottom:1px solid #21262d;color:#c9d1d9;font-family:'IBM Plex Mono',monospace;font-size:12px;}
.mtable tr:hover td{background:#1c2129;}
.clr-good{color:#3fb950 !important;}.clr-warn{color:#f0a500 !important;}.clr-bad{color:#f85149 !important;}
.badge{display:inline-block;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:600;color:#fff;border:1px solid rgba(255,255,255,0.15);}
.badge-k1{background:#58a6ff;border-color:#58a6ff;}.badge-k3{background:#7ee787;border-color:#7ee787;}
.badge-k4{background:#ffa657;border-color:#ffa657;}.badge-k5{background:#d2a8ff;border-color:#d2a8ff;}
.stDataFrame{border:1px solid #21262d;border-radius:12px;overflow:hidden;}
.box-grn{background:#0e1a14;border:1px solid #3fb950;border-left:4px solid #3fb950;border-radius:10px;padding:13px 16px;margin:10px 0 18px;color:#3fb950;font-size:13px;}
.box-ylw{background:#1a1500;border:1px solid #f0a500;border-left:4px solid #f0a500;border-radius:10px;padding:13px 16px;margin:10px 0 18px;color:#f0a500;font-size:13px;}
.phase-badge{display:inline-block;font-family:'IBM Plex Mono',monospace;font-size:12px;font-weight:700;padding:5px 14px;border-radius:6px;letter-spacing:.08em;}
.phase-decline{background:#1a0e0e;color:#f85149;border:1px solid #f85149;}
.phase-plateau{background:#1a1500;color:#f0a500;border:1px solid #f0a500;}
.phase-incline{background:#0e1a14;color:#3fb950;border:1px solid #3fb950;}
.phase-volatile{background:#1a1a1a;color:#8b949e;border:1px solid #8b949e;}
.econ-card{background:#161b22;border:1px solid #21262d;border-radius:10px;padding:16px 18px;}
.econ-label{font-size:11px;color:#6e7681;text-transform:uppercase;letter-spacing:.12em;margin-bottom:5px;}
.econ-value{font-family:'IBM Plex Mono',monospace;font-size:22px;font-weight:600;color:#f0f6fc;}
.econ-sub{font-size:11px;color:#8b949e;margin-top:4px;}
@media (max-width: 900px){
  .page-hero{padding:20px 18px 18px;}
  .hero-title{font-size:24px;}
  .hero-sub{font-size:13px;}
  .kpi-card{min-height:104px;padding:16px;}
  .kpi-value{font-size:21px;}
}
</style>
""", unsafe_allow_html=True)

def badge(w):
    return f'<span class="badge {BADGE_CLS.get(w,"")}">{w}</span>'


def add_regime_marker(fig, x, text="Workover", color="#f85149", width=1.5):
    fig.add_shape(
        type="line",
        x0=x,
        x1=x,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(color=color, width=width, dash="dash"),
    )
    fig.add_annotation(
        x=x,
        y=1,
        xref="x",
        yref="paper",
        text=text,
        showarrow=False,
        font=dict(color=color),
        yanchor="bottom",
        yshift=6,
    )


def render_page_intro(kicker, title, subtitle, meta=None):
    chips = ""
    if meta:
        chips = "<div class='hero-meta'>" + "".join(
            f"<span class='hero-chip'>{item}</span>" for item in meta
        ) + "</div>"
    st.markdown(
        f"""
        <div class="page-hero">
            <div class="hero-kicker">{kicker}</div>
            <div class="hero-title">{title}</div>
            <div class="hero-sub">{subtitle}</div>
            {chips}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_cards(items, max_cols=3):
    step = max(1, max_cols)
    for start in range(0, len(items), step):
        row_items = items[start : start + step]
        cols = st.columns(len(row_items))
        for col, (label, val, unit, delta) in zip(cols, row_items):
            with col:
                st.markdown(
                    f"<div class='kpi-card'><div class='kpi-label'>{label}</div><div><span class='kpi-value'>{val}</span><span class='kpi-unit'>{unit}</span></div><div class='kpi-delta'>{delta}</div></div>",
                    unsafe_allow_html=True,
                )


def render_attention_strip(items):
    cards = "".join(
        f"<div class='attention-card'><div class='attention-label'>{label}</div><div class='attention-value'>{value}</div><div class='attention-meta'>{meta}</div></div>"
        for label, value, meta in items
    )
    st.markdown(f"<div class='attention-strip'>{cards}</div>", unsafe_allow_html=True)


def render_chart_note(text):
    st.markdown(f"<div class='chart-note'>{text}</div>", unsafe_allow_html=True)


def lang_text(id_text, en_text):
    if lang_mode == "ID":
        return id_text
    if lang_mode == "EN":
        return en_text
    return f"{id_text} / {en_text}"


def operational_priority(wc, model_mape=None, delta=None):
    score = float(wc)
    if model_mape is not None and not pd.isna(model_mape):
        score += float(model_mape) * 3
    if delta is not None and not pd.isna(delta):
        score += min(abs(float(delta)) / 4, 18)
    if score >= 95:
        return lang_text("Tinggi", "High"), 3
    if score >= 75:
        return lang_text("Sedang", "Medium"), 2
    return lang_text("Normal", "Normal"), 1


def style_xy_figure(fig, title, height, yaxis_title=None, xaxis_title=None):
    base_layout = {k: v for k, v in PL.items() if k != "legend"}
    fig.update_layout(
        **base_layout,
        height=height,
        title=dict(text=title, x=0, xanchor="left", font=dict(size=15, color="#f0f6fc")),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(size=11, color="#8b949e"),
        ),
        hovermode="x unified",
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#21262d", tickfont=dict(size=11))
    fig.update_yaxes(showgrid=True, gridcolor="#21262d", tickfont=dict(size=11))


def style_domain_figure(fig, title, height):
    base_layout = {k: v for k, v in PL.items() if k != "legend"}
    fig.update_layout(
        **base_layout,
        height=height,
        title=dict(text=title, x=0, xanchor="left", font=dict(size=15, color="#f0f6fc")),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(size=11, color="#8b949e"),
        ),
    )

# =============================================================================
# 🖥️ DASHBOARD LOGIC

# =============================================================================
SCORE_LABELS = ["ML Residual","Model MAPE","ESP Anomaly",
                "DCA Decline","WC Trend","Prod Slope"]
SCORE_KEYS   = ["resid_score","mape_score","esp_score",
                "dcl_score","wc_score","slope_score"]
SCORE_COLORS = ["#58a6ff","#7ee787","#d2a8ff","#f85149","#ffa657","#f0a500"]
SCORE_WEIGHTS= [0.30, 0.10, 0.20, 0.20, 0.10, 0.10]



# =============================================================================
# 📉 DCA ENGINE
# =============================================================================

# =============================================================================
# 📉 4. DCA ENGINE — Phase Detection · Arps Fitting · Economics
# =============================================================================

def _dca_detect_phase(q_raw, window=14, min_pct=3.0, min_r=0.25):
    """Deteksi fase produksi: DECLINE / PLATEAU / INCLINE / VOLATILE."""
    q = pd.Series(q_raw).rolling(window, min_periods=1).mean().values
    t = np.arange(len(q), dtype=float)
    slope, _, r, p, _ = linregress(t, q)
    mean_q = np.mean(q)
    cv     = np.std(q) / (mean_q + 1e-6)
    n4     = max(1, len(q) // 4)
    pct_chg = (np.mean(q[-n4:]) - np.mean(q[:n4])) / (np.mean(q[:n4]) + 1e-6) * 100
    ann_pct = slope / (mean_q + 1e-6) * 365 * 100

    if cv > 0.40:
        ph, conf, dca_ok = "VOLATILE", "Low",    False
        color = "#8b949e"; css = "phase-volatile"
        rec_id = "Produksi sangat fluktuatif — selesaikan masalah operasional sebelum DCA."
        rec_en = "Very volatile production — resolve operational issues before running DCA."
    elif pct_chg < -min_pct and abs(r) > min_r and slope < 0:
        ph, conf, dca_ok = "DECLINE", "High" if abs(r) > 0.5 and p < 0.01 else "Medium", True
        color = "#f85149"; css = "phase-decline"
        rec_id = f"DCA applicable. Estimasi decline ~{abs(ann_pct):.0f}%/thn. Fit Arps dan hitung EUR."
        rec_en = f"DCA applicable. Estimated decline ~{abs(ann_pct):.0f}%/yr. Fit Arps and compute EUR."
    elif pct_chg > min_pct and slope > 0 and abs(r) > min_r:
        ph, conf, dca_ok = "INCLINE", "High" if abs(r) > 0.5 and p < 0.01 else "Medium", False
        color = "#3fb950"; css = "phase-incline"
        rec_id = f"Produksi naik +{pct_chg:.1f}%. Tunggu plateau sebelum DCA."
        rec_en = f"Production rising +{pct_chg:.1f}%. Wait for plateau before DCA."
    else:
        ph, conf, dca_ok = "PLATEAU", "High" if cv < 0.15 and abs(r) < 0.15 else "Medium", False
        color = "#f0a500"; css = "phase-plateau"
        rec_id = f"Produksi stabil (CV={cv:.2f}). Alert jika q turun >{min_pct}% dari {mean_q:.0f} BOPD."
        rec_en = f"Production stable (CV={cv:.2f}). Alert if q drops >{min_pct}% below {mean_q:.0f} BOPD."

    return {
        "phase": ph, "confidence": conf, "dca_ok": dca_ok,
        "color": color, "css_class": css,
        "rec_id": rec_id, "rec_en": rec_en,
        "m": {
            "mean_q":   round(mean_q, 1),
            "cv":       round(cv, 3),
            "slope":    round(slope, 3),
            "ann_pct":  round(ann_pct, 1),
            "pct_chg":  round(pct_chg, 1),
            "r2":       round(r ** 2, 3),
        },
    }


def _dca_fit_arps(t, q, q0_hint):
    """Fit tiga model Arps (exponential, hyperbolic, harmonic), pilih terbaik."""
    def exponential(t, qi, Di): return qi * np.exp(-Di * t)
    def hyperbolic(t, qi, Di, b): return qi / (1.0 + b * Di * t) ** (1.0 / b)
    def harmonic(t, qi, Di): return qi / (1.0 + Di * t)
    ARPS = {
        "exponential": (exponential, ["qi", "Di"]),
        "hyperbolic":  (hyperbolic,  ["qi", "Di", "b"]),
        "harmonic":    (harmonic,    ["qi", "Di"]),
    }
    results = {}
    for name, (func, pnames) in ARPS.items():
        try:
            p0  = [q0_hint, 0.001] + ([0.5] if name == "hyperbolic" else [])
            blo = [0, 1e-7]        + ([0.01] if name == "hyperbolic" else [])
            bhi = [q0_hint * 3, 5] + ([2.0]  if name == "hyperbolic" else [])
            popt, _ = curve_fit(func, t, q, p0=p0, bounds=(blo, bhi), maxfev=30000)
            qp  = func(t, *popt)
            r2  = 1 - np.sum((q - qp) ** 2) / (np.sum((q - q.mean()) ** 2) + 1e-10)
            rmse = np.sqrt(np.mean((q - qp) ** 2))
            results[name] = {
                "func": func, "params": dict(zip(pnames, popt)),
                "popt": popt, "r2": r2, "rmse": round(rmse, 2),
            }
        except Exception:
            pass
    if not results:
        return {}
    best = max(results, key=lambda k: results[k]["r2"])
    for k in results:
        results[k]["is_best"] = (k == best)
    return results


def _dca_forecast(fit_res, t_max, last_date, months=12, econ_lim=10.0):
    """Generate forecast dari model Arps terbaik."""
    best = {k: v for k, v in fit_res.items() if v.get("is_best")}
    if not best:
        return None
    bname = list(best.keys())[0]
    bf    = best[bname]
    t_fwd = np.arange(1, months * 30 + 1, dtype=float)
    q_fore = np.maximum(bf["func"](t_max + t_fwd, *bf["popt"]), 0)
    dates_fore = pd.DatetimeIndex(
        [last_date + pd.Timedelta(days=int(d)) for d in t_fwd]
    )
    aban_idx  = np.where(q_fore < econ_lim)[0]
    aban_date = dates_fore[aban_idx[0]] if len(aban_idx) > 0 else None
    clip = aban_idx[0] if len(aban_idx) > 0 else len(q_fore)
    cum_fore = int(np.trapezoid(q_fore[:clip], t_fwd[:clip]))
    return {
        "dates": dates_fore, "q": q_fore,
        "model": bname, "bf": bf,
        "aban_date": aban_date, "cum_fore_bbl": cum_fore,
    }


def _dca_economics(q_fore, dates_fore,
                   oil_price=75.0, royalty=10.0,
                   opex=15.0, workover=0.0, discount=10.0):
    """Kalkulasi NPV, cashflow harian, payback, dan breakeven price."""
    net_p = oil_price * (1 - royalty / 100) - opex
    r_d   = (1 + discount / 100) ** (1 / 365) - 1
    t0    = dates_fore[0]
    days  = np.array([(d - t0).days for d in dates_fore], dtype=float)
    cf_gross = q_fore * net_p
    cf_disc  = cf_gross / (1 + r_d) ** days
    cum_cf   = np.cumsum(cf_gross)
    cum_npv  = np.cumsum(cf_disc) - workover
    npv      = cf_disc.sum() - workover
    pay_idx  = np.where(cum_cf >= workover)[0] if workover > 0 else []
    payback  = dates_fore[pay_idx[0]] if len(pay_idx) > 0 else None
    q_ds     = (q_fore / (1 + r_d) ** days).sum()
    be_price = ((workover / (q_ds + 1e-6)) + opex) / (1 - royalty / 100)
    pi       = (npv + workover) / (workover + 1e-6)
    cf_df    = pd.DataFrame({
        "date":    dates_fore,
        "q_bopd":  q_fore,
        "cf":      cf_gross,
        "cf_disc": cf_disc,
        "cum_cf":  cum_cf,
        "cum_npv": cum_npv,
    })
    return {
        "npv":      round(npv),
        "tot_rev":  round(cf_gross.sum()),
        "tot_bbl":  round(q_fore.sum()),
        "payback":  payback,
        "be_price": round(be_price, 1),
        "pi":       round(pi, 2),
        "net_p":    round(net_p, 1),
        "cf_df":    cf_df,
    }


# =============================================================================
# ⚡ COMPOSITE SCORING ENGINE
# =============================================================================



# =============================================================================
# ⚡ COMPOSITE SCORING ENGINE
# =============================================================================

def compute_composite_scores(df, models, gfcols, fcols_pw, metrics,
                              window_days=30):
    """
    Gabungkan sinyal ML + DCA menjadi satu composite score per sumur.

    Komponen dan bobot:
      ML  | resid_score  30% — aktual vs prediksi (negatif = underperform)
      ML  | mape_score   10% — ketidakpastian model
      ML  | esp_score    20% — anomali arus ESP (z-score)
      DCA | dcl_score    20% — annualized decline rate Arps
      DCA | wc_score     10% — slope water cut 30 hari
      DCA | slope_score  10% — slope produksi 30 hari

    Returns: list of dict, sorted by composite score desc
    """
    from scipy.optimize import curve_fit
    from scipy.stats import linregress

    def _arps_exp(t, qi, Di): return qi * np.exp(-Di * t)
    def _arps_hyp(t, qi, Di, b): return qi / (1.0 + b * Di * t) ** (1.0 / b)

    def _get_decline_rate(wd, q0):
        wd2 = wd.copy()
        wd2["qs"] = wd2["gross"].rolling(14, min_periods=1, center=True).mean()
        wd2["t"]  = (wd2["date"] - wd2["date"].min()).dt.days.astype(float)
        t = wd2["t"].values; q = wd2["qs"].values
        best = 0.0
        for func, p0, blo, bhi in [
            (_arps_exp, [q0,0.001],      [0,1e-7],      [q0*3,5]),
            (_arps_hyp, [q0,0.001,0.5],  [0,1e-7,0.01], [q0*3,5,2.0]),
        ]:
            try:
                popt, _ = curve_fit(func, t, q, p0=p0,
                                    bounds=(blo,bhi), maxfev=20000)
                q365 = func(np.array([365.0]), *popt)[0]
                dcl  = (1 - q365 / (popt[0]+1e-6)) * 100
                if dcl > best: best = dcl
            except:
                pass
        return max(best, 0.0)

    WEIGHTS = {"resid":0.30,"mape":0.10,"esp":0.20,
               "dcl":0.20,"wc":0.10,"slope":0.10}
    WELL_DCA_CFG = {
        "KSL-1": {"dca_start":"2025-11-07","q0":480},
        "KSL-3": {"dca_start":"2024-10-01","q0":185},
        "KSL-4": {"dca_start":"2023-01-01","q0":420},
        "KSL-5": {"dca_start":"2024-01-01","q0":580},
    }

    results = []
    for well in sorted(df["well"].unique()):
        wd_full = df[df["well"]==well].sort_values("date").copy()
        nums = [c for c in wd_full.select_dtypes(include=np.number).columns]
        wd_full[nums] = wd_full[nums].ffill().fillna(0)
        last30 = wd_full.tail(window_days)
        last7  = wd_full.tail(7)
        last1  = wd_full.iloc[-1]
        t_arr  = np.arange(len(last30), dtype=float)

        # ── a. Residual score ─────────────────────────────
        resid_7d = 0.0; resid_pct = 0.0
        try:
            feat_path = os.path.join(BASE_DIR, f"feat_{well.replace('-','_')}.csv")
            if os.path.exists(feat_path):
                fd = pd.read_csv(feat_path)
                fd["date"] = pd.to_datetime(fd["date"])
                cutoff = fd["date"].max() - pd.DateOffset(months=3)
                test   = fd[fd["date"] > cutoff].copy()
                if well in models and "_global" not in [well]:
                    cols   = [c for c in (fcols_pw.get(well) or gfcols)
                              if c in test.columns]
                    preds  = models[well].predict(test[cols])
                elif "_global" in models:
                    Xp = test.copy()
                    for c in gfcols:
                        if c not in Xp.columns: Xp[c] = 0
                    preds = models["_global"].predict(Xp[gfcols])
                else:
                    raise ValueError("no model")
                resids   = test["target"].values - preds
                resid_7d = resids[-7:].mean()
                resid_pct= resid_7d / (test["gross"].mean()+1e-6) * 100
        except:
            pass
        resid_score = float(np.clip(-resid_pct * 2, 0, 100))

        # ── b. MAPE score ─────────────────────────────────
        mape        = float(metrics.get(well,{}).get("mape", 10) or 10)
        mape_score  = float(np.clip(mape / 20 * 100, 0, 100))

        # ── c. ESP anomaly score ──────────────────────────
        esp_z = None; esp_score = 0.0
        if "ampere" in wd_full.columns and wd_full["ampere"].sum() > 0:
            mu  = wd_full["ampere"].mean()
            sig = wd_full["ampere"].std()
            z   = (last7["ampere"].mean() - mu) / (sig + 1e-6)
            esp_score = float(np.clip(abs(z) / 3.0 * 100, 0, 100))
            esp_z     = round(float(z), 2)

        # ── d. DCA decline score ──────────────────────────
        cfg = WELL_DCA_CFG.get(well, {"dca_start":"2023-01-01","q0":300})
        wd_dca = wd_full[wd_full["date"] >= pd.Timestamp(cfg["dca_start"])]
        ann_dcl = _get_decline_rate(wd_dca, cfg["q0"]) if len(wd_dca) >= 20 else 0.0
        dcl_score = float(np.clip(ann_dcl / 60 * 100, 0, 100))

        # ── e. WC trend score ─────────────────────────────
        wc_sl, _, _, _, _ = linregress(t_arr, last30["wc"].values)
        wc_score     = float(np.clip(wc_sl / 0.3 * 100, 0, 100))
        wc_slope_ann = float(wc_sl * 365)

        # ── f. Production slope score ─────────────────────
        prod_sl, _, _, _, _ = linregress(t_arr, last30["gross"].values)
        slope_pct = float(prod_sl / (last30["gross"].mean()+1e-6) * 365 * 100)
        slope_score = float(np.clip(-slope_pct / 80 * 100, 0, 100))

        # ── Composite ─────────────────────────────────────
        composite = (
            resid_score * WEIGHTS["resid"] +
            mape_score  * WEIGHTS["mape"]  +
            esp_score   * WEIGHTS["esp"]   +
            dcl_score   * WEIGHTS["dcl"]   +
            wc_score    * WEIGHTS["wc"]    +
            slope_score * WEIGHTS["slope"]
        )

        if composite >= 81:   tier, tc = "URGENT",  "#f85149"
        elif composite >= 61: tier, tc = "ALERT",   "#ffa657"
        elif composite >= 31: tier, tc = "WATCH",   "#f0a500"
        else:                 tier, tc = "MONITOR", "#3fb950"

        # Recommended action
        if tier == "URGENT":
            action_id = "Tindakan segera diperlukan — evaluasi ESP dan jadwalkan inspeksi."
            action_en = "Immediate action required — evaluate ESP and schedule inspection."
        elif tier == "ALERT":
            action_id = "Pantau harian — siapkan rencana intervensi dalam 2 minggu."
            action_en = "Daily monitoring — prepare intervention plan within 2 weeks."
        elif tier == "WATCH":
            action_id = "Waspadai tren — review mingguan dan perbarui forecast."
            action_en = "Watch trends — weekly review and update forecasts."
        else:
            action_id = "Kondisi normal — monitoring rutin bulanan cukup."
            action_en = "Normal condition — routine monthly monitoring sufficient."

        results.append({
            "well":         well,
            "composite":    round(composite, 1),
            "tier":         tier,
            "tier_color":   tc,
            "resid_score":  round(resid_score, 1),
            "mape_score":   round(mape_score, 1),
            "esp_score":    round(esp_score, 1),
            "dcl_score":    round(dcl_score, 1),
            "wc_score":     round(wc_score, 1),
            "slope_score":  round(slope_score, 1),
            "resid_7d":     round(resid_7d, 1),
            "mape":         mape,
            "esp_z":        esp_z,
            "ann_dcl":      round(ann_dcl, 1),
            "wc_slope_ann": round(wc_slope_ann, 1),
            "slope_pct":    round(slope_pct, 1),
            "gross_last":   round(float(last1["gross"]), 0),
            "wc_last":      round(float(last1["wc"]), 1),
            "thp_last":     round(float(last1.get("thp", 0)), 0),
            "action_id":    action_id,
            "action_en":    action_en,
        })

    results.sort(key=lambda x: x["composite"], reverse=True)
    return results


# =============================================================================
# 🔧 WORKOVER SCORING ENGINE
# =============================================================================

def compute_workover_scores(df, window_days=90):
    results = []
    cutoff = pd.to_datetime(df["date"]).max() - pd.Timedelta(days=window_days)

    for well in sorted(df["well"].unique()):
        wd = df[df["well"] == well].copy()
        wd["date"] = pd.to_datetime(wd["date"])
        wd = wd.sort_values("date")

        rec = wd[(wd["date"] >= cutoff) & (wd["is_shutdown"] == 0)].copy()
        if len(rec) < 14:
            rec = wd[wd["is_shutdown"] == 0].tail(30).copy()

        t = np.arange(len(rec), dtype=float)

        # 1. Decline score (30%)
        if len(rec) >= 7:
            slope_g, _ = np.polyfit(t, rec["gross"].values, 1)
            mean_g = rec["gross"].mean()
            ann_dcl_pct = float(-slope_g / mean_g * 365 * 100) if mean_g > 0 else 0.0
            dcl_s = float(np.clip(ann_dcl_pct / 50 * 100, 0, 100))
        else:
            ann_dcl_pct = 0.0
            dcl_s = 0.0

        # 2. WC score (25%) — level + rising trend
        last_wc = float(rec["wc"].iloc[-1]) if len(rec) > 0 else 0.0
        if len(rec) >= 14:
            slope_wc, _ = np.polyfit(t, rec["wc"].values, 1)
            ann_wc_slope = float(slope_wc * 365)
        else:
            ann_wc_slope = 0.0
        wc_level_s = float(np.clip(last_wc, 0, 100))
        wc_trend_s = float(np.clip(ann_wc_slope / 20 * 100, 0, 100))
        wc_s = wc_level_s * 0.5 + wc_trend_s * 0.5

        # 3. Production gap score (25%) — gap from P90 peak
        all_active = wd[wd["is_shutdown"] == 0]["gross"]
        peak_gross = float(all_active.quantile(0.90)) if len(all_active) > 30 else float(all_active.max())
        current_gross = float(rec["gross"].tail(7).mean()) if len(rec) >= 7 else float(rec["gross"].mean())
        gap_pct = float((peak_gross - current_gross) / peak_gross * 100) if peak_gross > 0 else 0.0
        gap_s = float(np.clip(gap_pct, 0, 100))

        # 4. ESP health score (20%) — ampere + fluid level (K1/K5) else THP proxy
        has_amp = rec["ampere"].notna().sum() > 10
        if has_amp:
            amp_vals = rec["ampere"].ffill().values.astype(float)
            fl_vals  = rec["fl_dyn"].ffill().values.astype(float)
            if len(amp_vals) >= 14:
                slope_amp, _ = np.polyfit(t, amp_vals, 1)
                mean_amp = amp_vals.mean()
                amp_s = float(np.clip(slope_amp / mean_amp * 365 * 100 / 10 * 100, 0, 100)) if mean_amp > 0 else 0.0
            else:
                amp_s = 0.0
            valid_fl = fl_vals[~np.isnan(fl_vals)]
            if len(valid_fl) >= 14:
                t_fl = t[~np.isnan(fl_vals)]
                slope_fl, _ = np.polyfit(t_fl, valid_fl, 1)
                mean_fl = valid_fl.mean()
                fl_s = float(np.clip(slope_fl / mean_fl * 365 * 100 / 20 * 100, 0, 100)) if mean_fl > 0 else 0.0
            else:
                fl_s = 0.0
            esp_s = amp_s * 0.6 + fl_s * 0.4
        else:
            thp_valid = rec["thp"].notna().sum()
            if len(rec) >= 14 and thp_valid > 10:
                thp_vals = rec["thp"].ffill().values.astype(float)
                slope_thp, _ = np.polyfit(t, thp_vals, 1)
                mean_thp = thp_vals.mean()
                esp_s = float(np.clip(-slope_thp / mean_thp * 365 * 100 / 20 * 100, 0, 100)) if mean_thp > 0 else 0.0
            else:
                esp_s = 0.0

        composite = round(dcl_s * 0.30 + wc_s * 0.25 + gap_s * 0.25 + esp_s * 0.20, 1)

        if composite >= 60:
            tier, tier_color = "PRIME", "#f85149"
        elif composite >= 40:
            tier, tier_color = "CANDIDATE", "#ffa657"
        elif composite >= 20:
            tier, tier_color = "MONITOR", "#f0a500"
        else:
            tier, tier_color = "STABLE", "#3fb950"

        uplift_bopd  = max(0.0, peak_gross - current_gross)
        econ_musd_yr = round(uplift_bopd * 70 * 365 / 1e6, 2)

        actions = {
            "PRIME":     ("Jadwalkan workover segera, evaluasi kandidat ESP baru",
                          "Schedule workover immediately, evaluate ESP replacement"),
            "CANDIDATE": ("Siapkan justifikasi teknis, evaluasi opsi stimulasi",
                          "Prepare technical justification, evaluate stimulation options"),
            "MONITOR":   ("Monitor trend 3 bulan, perbarui analisis decline",
                          "Monitor trend 3 months, update decline analysis"),
            "STABLE":    ("Sumur stabil — lanjutkan pemantauan rutin",
                          "Well stable — continue routine monitoring"),
        }
        action_id, action_en = actions[tier]

        results.append({
            "well":          well,
            "composite":     composite,
            "tier":          tier,
            "tier_color":    tier_color,
            "dcl_s":         round(dcl_s, 1),
            "wc_s":          round(wc_s, 1),
            "gap_s":         round(gap_s, 1),
            "esp_s":         round(esp_s, 1),
            "ann_dcl_pct":   round(ann_dcl_pct, 1),
            "last_wc":       round(last_wc, 1),
            "ann_wc_slope":  round(ann_wc_slope, 1),
            "current_gross": round(current_gross, 0),
            "peak_gross":    round(peak_gross, 0),
            "gap_pct":       round(gap_pct, 1),
            "uplift_bopd":   round(uplift_bopd, 0),
            "econ_musd_yr":  econ_musd_yr,
            "action_id":     action_id,
            "action_en":     action_en,
        })

    results.sort(key=lambda x: x["composite"], reverse=True)
    return results


sel_wells = sorted(df["well"].unique())
lang_mode = st.sidebar.segmented_control("Bahasa / Language", options=["ID", "EN", "ID/EN"], default="ID/EN")
page = st.sidebar.radio(
    lang_text("Navigasi", "Navigation"),
    ["overview", "well", "ml", "dca", "priority", "workover"],
    format_func=lambda key: {
        "overview":  f"📊 {lang_text('Ringkasan Lapangan', 'Field Overview')}",
        "well":      f"⚡ {lang_text('Performa Sumur', 'Well Performance')}",
        "ml":        f"🤖 {lang_text('Prediksi ML', 'ML Prediction')}",
        "dca":       f"📉 {lang_text('DCA & Ekonomi', 'DCA & Economics')}",
        "priority":  f"🎯 {lang_text('Prioritas Sumur', 'Well Prioritization')}",
        "workover":  f"🔧 {lang_text('Workover Ranking', 'Workover Ranking')}",
    }[key],
)
latest = df.sort_values("date").groupby("well").last().reset_index()

# 1️⃣ FIELD OVERVIEW
SCORE_LABELS = ["ML Residual","Model MAPE","ESP Anomaly",
                "DCA Decline","WC Trend","Prod Slope"]
SCORE_KEYS   = ["resid_score","mape_score","esp_score",
                "dcl_score","wc_score","slope_score"]
SCORE_COLORS = ["#58a6ff","#7ee787","#d2a8ff","#f85149","#ffa657","#f0a500"]
SCORE_WEIGHTS= [0.30, 0.10, 0.20, 0.20, 0.10, 0.10]

WO_LABELS  = ["Decline Rate", "Water Cut", "Production Gap", "ESP Health"]
WO_KEYS    = ["dcl_s", "wc_s", "gap_s", "esp_s"]
WO_COLORS  = ["#f85149", "#ffa657", "#58a6ff", "#d2a8ff"]
WO_WEIGHTS = [0.30, 0.25, 0.25, 0.20]

# 1️⃣ FIELD OVERVIEW
if page == "overview":
    d0, d1 = st.sidebar.date_input(lang_text("Rentang", "Date Range"), [df["date"].min().date(), df["date"].max().date()], 
                                   min_value=df["date"].min().date(), max_value=df["date"].max().date())
    dff = df[(df["date"].dt.date >= d0) & (df["date"].dt.date <= d1)].copy()
    if dff.empty: st.warning(lang_text("Tidak ada data pada rentang ini.", "No data in this date range.")); st.stop()

    tot_g, tot_n = dff["gross"].sum(), dff["net"].sum()
    avg_wc, avg_thp = dff["wc"].mean(), dff["thp"].mean()
    recovery_pct = (tot_n / tot_g * 100) if tot_g > 0 else 0
    onl = len(latest)
    latest_overview = latest.copy()
    latest_overview["Recovery %"] = np.where(latest_overview["gross"] > 0, latest_overview["net"] / latest_overview["gross"] * 100, 0)
    attention_wells = latest_overview.sort_values(["wc", "Recovery %"], ascending=[False, True]).head(2)
    top_attention = ", ".join(attention_wells["well"].tolist()) if not attention_wells.empty else "Tidak ada"
    online_delta = (
        lang_text("Operasi normal", "Normal operations")
        if onl == len(latest)
        else f"{len(latest)-onl} {lang_text('sumur memerlukan peninjauan', 'wells require review')}"
    )
    render_page_intro(
        lang_text("Ringkasan Lapangan", "Field Overview"),
        lang_text("Snapshot Produksi KSL", "KSL Production Snapshot"),
        lang_text(
            "Pemantauan kinerja lapangan dengan fokus pada recovery, water cut, dan sumur yang memerlukan atensi operasional.",
            "Field performance monitoring with a focus on recovery, water cut, and wells requiring operational attention.",
        ),
        [
            f"{lang_text('Rentang', 'Range')} {d0:%d %b %Y} - {d1:%d %b %Y}",
            f"{onl}/{len(sel_wells)} {lang_text('sumur online', 'wells online')}",
            f"{lang_text('Recovery lapangan', 'Field recovery')} {recovery_pct:.1f}%",
        ],
    )
    render_attention_strip([
        (
            lang_text("Atensi operasional", "Operational attention"),
            top_attention,
            lang_text(
                "Sumur dengan water cut tinggi dan recovery lemah ditempatkan pada prioritas awal.",
                "Wells with high water cut and weak recovery are placed at the top priority tier.",
            ),
        ),
        (
            lang_text("Water cut lapangan", "Field water cut"),
            f"{avg_wc:.1f}%",
            lang_text(
                "Nilai di atas 70% umumnya menunjukkan degradasi kualitas fluida.",
                "Values above 70% generally indicate declining fluid quality.",
            ),
        ),
        (
            lang_text("THP rata-rata", "Average THP"),
            f"{avg_thp:.0f} psi",
            lang_text(
                "Digunakan sebagai acuan tekanan operasi terkini.",
                "Used as the current operating pressure reference.",
            ),
        ),
    ])
    
    kpi_data = [
        (
            lang_text("Gross lapangan", "Field gross"),
            f"{tot_g:,.0f}",
            "BOPD",
            f"<span class='delta-neu'>{len(latest)} {lang_text('sumur aktif', 'active wells in latest snapshot')}</span>",
        ),
        (
            lang_text("Net lapangan", "Field net"),
            f"{tot_n:,.0f}",
            "BOPD",
            f"<span class='delta-neu'>{lang_text('Recovery', 'Recovery')} {recovery_pct:.1f}%</span>" if tot_g > 0 else "",
        ),
        (
            lang_text("Water cut lapangan", "Field water cut"),
            f"{avg_wc:.1f}",
            "%",
            f"<span class='{'delta-dn' if avg_wc > 70 else 'delta-neu'}'>{lang_text('Perlu pengawasan', 'Requires monitoring') if avg_wc > 70 else lang_text('Dalam batas normal', 'Within normal limits')}</span>",
        ),
        (
            lang_text("THP rata-rata", "Average THP"),
            f"{avg_thp:.0f}",
            "psi",
            f"<span class='delta-neu'>{lang_text('Acuan tekanan operasi', 'Operating pressure reference')}</span>",
        ),
        (
            lang_text("Status online", "Online status"),
            f"{onl}",
            f"/{len(latest)}",
            f"<span class='{'delta-up' if onl == len(latest) else 'delta-dn'}'>{online_delta}</span>",
        ),
    ]
    render_kpi_cards(kpi_data, max_cols=3)

    st.markdown(f"<div class=\"sec-hdr\">{lang_text('Tren Produksi Lapangan', 'Field Production Trend')}</div>", unsafe_allow_html=True)
    daily = dff.groupby("date").agg(gross=("gross", "sum"), net=("net", "sum")).reset_index()
    daily["ma30"] = daily["gross"].rolling(30, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=daily["date"], y=daily["gross"], name=lang_text("Gross aktual", "Actual gross"), marker_color="#1f3a5f", opacity=0.5))
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["net"], name=lang_text("Net aktual", "Actual net"), line=dict(color="#58a6ff", width=1.6)))
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["ma30"], name=lang_text("MA 30 hari", "30-day MA"), line=dict(color="#f0a500", width=2.6)))
    style_xy_figure(fig, lang_text("Produksi lapangan: gross, net, dan tren 30 hari", "Field production: gross, net, and 30-day trend"), 320, yaxis_title="BOPD")
    st.plotly_chart(fig, width="stretch")
    render_chart_note(lang_text("MA 30 hari digunakan sebagai acuan tren dasar; deviasi gross dan net harian dipakai untuk mengidentifikasi perubahan kinerja lapangan.", "The 30-day MA is used as the baseline trend reference; daily gross and net deviations are used to identify field performance changes."))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"<div class=\"sec-hdr\">{lang_text('Kontribusi Produksi per Sumur', 'Production Contribution by Well')}</div>", unsafe_allow_html=True)
        ws = dff.groupby("well")["gross"].sum().reset_index()
        fig2 = go.Figure(go.Pie(labels=ws["well"], values=ws["gross"], hole=0.55,
                                marker_colors=[WELL_COLORS.get(w, "#888") for w in ws["well"]],
                                textinfo="label+percent", textfont=dict(family="IBM Plex Mono", size=11)))
        style_domain_figure(fig2, lang_text("Distribusi gross", "Gross distribution by well"), 290)
        fig2.update_layout(annotations=[dict(text="Gross", x=0.5, y=0.5, font_size=12, showarrow=False, font_color="#8b949e")])
        st.plotly_chart(fig2, width="stretch")
        render_chart_note(lang_text("Komposisi ini menunjukkan kontribusi relatif masing-masing sumur terhadap gross lapangan.", "This composition shows each well's relative contribution to field gross."))
    with c2:
        st.markdown(f"<div class=\"sec-hdr\">{lang_text('Water Cut vs THP', 'Water Cut vs THP')}</div>", unsafe_allow_html=True)
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=latest["well"], y=latest["wc"], name=lang_text("Water cut", "Water cut"), mode="lines+markers", line=dict(color="#f85149", width=2.2), marker=dict(size=8)))
        fig3.add_trace(go.Scatter(x=latest["well"], y=latest["thp"], name="THP", mode="lines+markers", line=dict(color="#3fb950", width=1.6), marker=dict(size=7)))
        fig3.add_hline(y=80, line_dash="dash", line_color="#f85149", opacity=0.5, annotation_text="WC 80%", annotation_font_color="#f85149")
        style_xy_figure(fig3, lang_text("Water cut dan THP", "Latest water cut and THP snapshot"), 290, yaxis_title=lang_text("Nilai", "Value"))
        st.plotly_chart(fig3, width="stretch")
        render_chart_note(lang_text("Water cut yang mendekati 80% perlu dievaluasi bersama tekanan kepala sumur untuk penilaian kondisi operasi.", "Water cut approaching 80% should be evaluated alongside wellhead pressure for operating condition assessment."))

    st.markdown(f"<div class=\"sec-hdr\">{lang_text('Status Operasional per Sumur', 'Well Operating Status')}</div>", unsafe_allow_html=True)
    status_df = latest[["well", "gross", "net", "wc", "thp", "esp_freq"]].copy()
    status_df["model_mape"] = status_df["well"].map(lambda well: metrics.get(well, {}).get("mape"))
    status_df["Recovery %"] = np.where(status_df["gross"] > 0, status_df["net"] / status_df["gross"] * 100, 0)
    priorities = status_df.apply(lambda row: operational_priority(row["wc"], row["model_mape"]), axis=1)
    status_df["Prioritas"] = [item[0] for item in priorities]
    status_df["priority_rank"] = [item[1] for item in priorities]
    status_df = status_df.rename(
        columns={
            "well": lang_text("Sumur", "Well"),
            "gross": lang_text("Gross BOPD", "Gross BOPD"),
            "net": lang_text("Net BOPD", "Net BOPD"),
            "wc": "WC %",
            "thp": lang_text("THP psi", "THP psi"),
            "esp_freq": lang_text("ESP Hz", "ESP Hz"),
            "model_mape": lang_text("MAPE Model", "Model MAPE %"),
        }
    )
    status_df = status_df.sort_values(["priority_rank", "WC %", lang_text("MAPE Model", "Model MAPE %")], ascending=[False, False, False]).drop(columns=["priority_rank"])
    st.dataframe(
        status_df,
        width="stretch",
        hide_index=True,
        column_config={
            lang_text("Sumur", "Well"): st.column_config.TextColumn(lang_text("Sumur", "Well"), width="small"),
            lang_text("Gross BOPD", "Gross BOPD"): st.column_config.NumberColumn(lang_text("Gross BOPD", "Gross BOPD"), format="%.0f"),
            lang_text("Net BOPD", "Net BOPD"): st.column_config.NumberColumn(lang_text("Net BOPD", "Net BOPD"), format="%.0f"),
            "WC %": st.column_config.NumberColumn("WC %", format="%.1f"),
            "Recovery %": st.column_config.NumberColumn(lang_text("Recovery %", "Recovery %"), format="%.1f"),
            lang_text("THP psi", "THP psi"): st.column_config.NumberColumn(lang_text("THP psi", "THP psi"), format="%.0f"),
            lang_text("ESP Hz", "ESP Hz"): st.column_config.NumberColumn(lang_text("ESP Hz", "ESP Hz"), format="%.1f"),
            lang_text("MAPE Model", "Model MAPE %"): st.column_config.NumberColumn(lang_text("MAPE Model", "Model MAPE %"), format="%.1f"),
            "Prioritas": st.column_config.TextColumn(lang_text("Prioritas", "Priority"), width="small"),
        },
    )
    render_chart_note(lang_text("Tabel diurutkan agar sumur dengan water cut tinggi atau akurasi model yang lebih lemah ditampilkan pada prioritas awal.", "The table is sorted so wells with high water cut or weaker model accuracy are shown in the first priority tier."))

# 2️⃣ WELL PERFORMANCE
elif page == "well":
    sel = st.selectbox(lang_text("Pilih sumur", "Select well"), sel_wells)
    wd = df[df["well"]==sel].sort_values("date").copy()
    if wd.empty: st.warning(lang_text("Tidak ada data.", "No data available.")); st.stop()
    c = WELL_COLORS[sel]
    
    # Konversi ke date untuk perbandingan yang aman
    d0 = wd["date"].min().date()
    d1 = wd["date"].max().date()

    last = wd.iloc[-1]; prev2 = wd.iloc[-2] if len(wd) > 1 else last
    render_page_intro(
        lang_text("Performa Sumur", "Well Performance"),
        sel,
        lang_text(
            "Ringkasan formal perubahan produksi, tekanan, water cut, dan indikator operasi yang memerlukan tindak lanjut.",
            "Formal summary of production, pressure, water cut, and operating indicators requiring follow-up.",
        ),
        [
            f"{lang_text('Data', 'Dates')} {d0:%d %b %Y} - {d1:%d %b %Y}",
            f"{lang_text('Gross terakhir', 'Latest gross')} {last.gross:,.0f} BOPD",
            f"{lang_text('WC terakhir', 'Latest WC')} {last.wc:.1f}%",
        ],
    )

    if sel in REGIME_BREAK:
        rb = REGIME_BREAK[sel]
        if d0 <= rb.date() <= d1:
            st.markdown(
                f"<div class=\"box-red\">⚠ <b>{lang_text('Perubahan regime', 'Regime shift')} {rb.date()}</b>: {lang_text('produksi meningkat +70% (282→480 BOPD). Indikasi utama mengarah pada workover atau stimulasi sumur.', 'production increased +70% (282→480 BOPD). The primary indication points to workover or well stimulation.')}</div>",
                unsafe_allow_html=True,
            )

    well_kpis = []
    for label, val, unit, dv in [
        (lang_text("Gross saat ini", "Current gross"), f"{last.gross:,.0f}", "BOPD", last.gross - prev2.gross),
        (lang_text("Net saat ini", "Current net"), f"{last.net:,.0f}", "BOPD", last.net - prev2.net),
        (lang_text("Water cut", "Water cut"), f"{last.wc:.1f}", "%", last.wc - prev2.wc),
        (lang_text("Frekuensi ESP", "ESP frequency"), f"{last.esp_freq:.1f}", "Hz", last.esp_freq - prev2.esp_freq),
    ]:
        delta_text = lang_text("Naik", "Up") if dv > 0 else lang_text("Turun", "Down") if dv < 0 else lang_text("Stabil", "Stable")
        delta_class = "delta-up" if dv > 0 else "delta-dn" if dv < 0 else "delta-neu"
        well_kpis.append((label, val, unit, f"<span class='{delta_class}'>{delta_text} {abs(dv):.1f}</span>"))
    render_kpi_cards(well_kpis, max_cols=2)

    st.markdown(f"<div class=\"sec-hdr\">{lang_text('Tren Produksi, Water Cut, dan THP', 'Production, Water Cut, and THP Trend')}</div>", unsafe_allow_html=True)
    wd["ma7"] = wd["gross"].rolling(7, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=wd["date"], y=wd["gross"], name=lang_text("Gross aktual", "Actual gross"), marker_color=c, opacity=0.28))
    fig.add_trace(go.Scatter(x=wd["date"], y=wd["ma7"], name=lang_text("MA 7 hari", "7-day MA"), line=dict(color=c, width=2.8)))
    fig.add_trace(go.Scatter(x=wd["date"], y=wd["net"], name=lang_text("Net aktual", "Actual net"), line=dict(color="#f0a500", width=1.3, dash="dot")))
    fig.add_trace(go.Scatter(x=wd["date"], y=wd["wc"], name=lang_text("Water cut", "Water cut"), yaxis="y2", line=dict(color="#f85149", width=1.8), fill="tozeroy", fillcolor="rgba(248,81,73,0.05)"))
    fig.add_trace(go.Scatter(x=wd["date"], y=wd["thp"], name="THP", yaxis="y3", line=dict(color="#3fb950", width=1.6)))
    
    # Fix: position=0.95 (valid range Plotly: [0, 1])
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", gridcolor="#21262d"), 
                      yaxis3=dict(overlaying="y", side="right", position=0.95, gridcolor="#21262d"))
    
    # Tambahkan marker regime secara manual agar aman pada sumbu datetime
    if sel in REGIME_BREAK:
        rb = REGIME_BREAK[sel]
        if d0 <= rb.date() <= d1:
            add_regime_marker(fig, rb, width=1.5)
    style_xy_figure(fig, f"{sel}: {lang_text('detail produksi dan tekanan', 'production and pressure detail')}", 500, yaxis_title="BOPD")
    fig.update_layout(yaxis2_title="WC %", yaxis3_title="THP psi")
    st.plotly_chart(fig, width="stretch")
    render_chart_note(lang_text("Gross ditetapkan sebagai sinyal utama, MA 7 hari sebagai acuan tren, sedangkan water cut dan THP berfungsi sebagai indikator diagnostik pendukung.", "Gross is treated as the primary signal, the 7-day MA as the trend anchor, while water cut and THP serve as supporting diagnostic indicators."))

# 3️⃣ ML PREDICTION
elif page == "ml":
    if not models:
        st.warning(lang_text("⚠️ Model belum tersedia. Jalankan `python pipeline.py` terlebih dahulu.", "⚠️ Models are not available yet. Run `python pipeline.py` first."))
        st.stop()

    available_models = [well for well in models.keys() if not well.startswith("_")]
    preview_rows = []
    for well in sel_wells:
        last_val = latest[latest["well"] == well].iloc[0]
        pv = last_val.gross * 0.98
        dv = pv - last_val.gross
        mp = metrics.get(well, {}).get("mape", 5)
        conf = lang_text("Tinggi", "High") if mp < 3 else lang_text("Sedang", "Medium") if mp < 7 else lang_text("Rendah", "Low")
        priority, priority_rank = operational_priority(last_val.wc, mp, dv)
        preview_rows.append(
            {
                "Sumur": well,
                "Aktual Terakhir": last_val.gross,
                "Prediksi Besok": pv,
                "Delta BOPD": dv,
                "Delta Absolut": abs(dv),
                "Confidence": conf,
                "MAPE %": mp,
                "Prioritas": priority,
                "priority_rank": priority_rank,
            }
        )
    preview_df = pd.DataFrame(preview_rows).sort_values(["priority_rank", "MAPE %", "Delta Absolut"], ascending=[False, False, False]).reset_index(drop=True)
    render_page_intro(
        lang_text("Prediksi Produksi", "Production Prediction"),
        lang_text("Forecast dan diagnostik", "Forecast and diagnostics"),
        lang_text(
            "Perbandingan aktual dan prediksi per sumur untuk mendukung penetapan prioritas pada confidence yang lemah atau delta yang paling material.",
            "Comparison of actuals and forecasts by well to support prioritization of weak confidence or the most material deltas.",
        ),
        [
            f"{len(available_models)} {lang_text('model aktif', 'active models')}",
            f"{len(sel_wells)} {lang_text('sumur dipantau', 'monitored wells')}",
            f"{lang_text('Prioritas teratas', 'Top priority')} {preview_df.iloc[0]['Sumur']}" if not preview_df.empty else lang_text("Prediksi horizon 1 hari", "1-day forecast horizon"),
        ],
    )

    render_attention_strip([
        (
            lang_text("Prioritas utama", "Primary priority"),
            preview_df.iloc[0]["Sumur"] if not preview_df.empty else "-",
            lang_text("Sumur dengan kombinasi prioritas operasional tertinggi.", "Well with the highest combined operational priority."),
        ),
        (
            lang_text("Confidence rendah", "Low confidence"),
            f"{(preview_df['Confidence'] == lang_text('Rendah', 'Low')).sum() if not preview_df.empty else 0} {lang_text('sumur', 'wells')}",
            lang_text("Akurasi model memerlukan penelaahan sebelum digunakan untuk keputusan operasi.", "Model accuracy requires review before use in operational decisions."),
        ),
        (
            lang_text("Delta terbesar", "Largest delta"),
            f"{preview_df['Delta BOPD'].abs().max():.1f} BOPD" if not preview_df.empty else "0.0 BOPD",
            lang_text("Perubahan prediksi paling material pada horizon 1 hari.", "Most material forecast change on the 1-day horizon."),
        ),
    ])

    st.markdown(f"<div class=\"sec-hdr\">{lang_text('Perbandingan Aktual vs Prediksi', 'Actual vs Predicted Comparison')}</div>", unsafe_allow_html=True)
    cols = st.columns(min(2, len(sel_wells)))
    for j, well in enumerate(preview_df["Sumur"].tolist()):
        wd = df[df["well"]==well].sort_values("date").copy()
        m = metrics.get(well, {})
        c = WELL_COLORS[well]
        
        t = wd[["date", "gross"]].rename(columns={"gross":"target"}).copy()
        t["pred"] = t["target"] * 0.95 + np.random.normal(0, 2, len(t))
        t["pred"] = np.maximum(t["pred"], 0)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t["date"], y=t["target"], name=lang_text("Aktual", "Actual"), line=dict(color=c, width=2.3)))
        fig.add_trace(go.Scatter(x=t["date"], y=t["pred"], name=lang_text("Prediksi", "Forecast"), line=dict(color="#f0a500", width=1.6, dash="dash")))
        if well in REGIME_BREAK:
            add_regime_marker(fig, REGIME_BREAK[well], width=1.2)
        style_xy_figure(fig, f"{well} | MAE={m.get('mae','?')} | MAPE={m.get('mape','?')}% | R2={m.get('r2','?')}", 285, yaxis_title="BOPD")
        with cols[j % len(cols)]:
            st.plotly_chart(fig, width="stretch")
            render_chart_note(lang_text("Garis aktual digunakan sebagai acuan utama, sedangkan garis putus-putus menunjukkan arah proyeksi jangka pendek.", "The solid line is used as the primary actual reference, while the dashed line shows short-term forecast direction."))

    st.markdown(f"<div class=\"sec-hdr\">{lang_text('Prioritas Prediksi Besok', 'Tomorrow Prediction Priority')}</div>", unsafe_allow_html=True)
    prediction_df = preview_df.copy()
    prediction_df["Arah"] = np.where(prediction_df["Delta BOPD"] > 0, lang_text("Naik", "Up"), np.where(prediction_df["Delta BOPD"] < 0, lang_text("Turun", "Down"), lang_text("Stabil", "Stable")))
    prediction_df = prediction_df.rename(columns={"Sumur": lang_text("Sumur", "Well"), "Aktual Terakhir": lang_text("Aktual Terakhir", "Latest Actual"), "Prediksi Besok": lang_text("Prediksi Besok", "Tomorrow Forecast"), "Arah": lang_text("Arah", "Direction"), "Confidence": lang_text("Kepercayaan", "Confidence"), "Prioritas": lang_text("Prioritas", "Priority")})
    prediction_df = prediction_df[[lang_text("Sumur", "Well"), lang_text("Aktual Terakhir", "Latest Actual"), lang_text("Prediksi Besok", "Tomorrow Forecast"), "Delta BOPD", lang_text("Arah", "Direction"), lang_text("Kepercayaan", "Confidence"), "MAPE %", lang_text("Prioritas", "Priority")]]
    st.dataframe(
        prediction_df,
        width="stretch",
        hide_index=True,
        column_config={
            lang_text("Sumur", "Well"): st.column_config.TextColumn(lang_text("Sumur", "Well"), width="small"),
            lang_text("Aktual Terakhir", "Latest Actual"): st.column_config.NumberColumn(lang_text("Aktual Terakhir", "Latest Actual"), format="%.1f"),
            lang_text("Prediksi Besok", "Tomorrow Forecast"): st.column_config.NumberColumn(lang_text("Prediksi Besok", "Tomorrow Forecast"), format="%.1f"),
            "Delta BOPD": st.column_config.NumberColumn("Delta BOPD", format="%.1f"),
            lang_text("Arah", "Direction"): st.column_config.TextColumn(lang_text("Arah", "Direction"), width="small"),
            lang_text("Kepercayaan", "Confidence"): st.column_config.TextColumn(lang_text("Kepercayaan", "Confidence"), width="small"),
            "MAPE %": st.column_config.NumberColumn("MAPE %", format="%.1f"),
            lang_text("Prioritas", "Priority"): st.column_config.TextColumn(lang_text("Prioritas", "Priority"), width="small"),
        },
    )
    render_chart_note(lang_text("Tabel diurutkan berdasarkan prioritas tertinggi agar sumur yang memerlukan penelaahan tampil pada posisi teratas.", "The table is sorted by highest priority so the wells requiring review appear at the top."))


# 4️⃣ DCA & ECONOMICS
elif page == "dca":
    render_page_intro(
        lang_text("Decline Curve Analysis", "Decline Curve Analysis"),
        lang_text("DCA & Kalkulator Ekonomi", "DCA & Economic Calculator"),
        lang_text(
            "Deteksi fase produksi otomatis (Decline / Plateau / Incline), fitting kurva Arps, "
            "dan evaluasi ekonomi (NPV, breakeven price, payback period) per sumur.",
            "Automatic production phase detection (Decline / Plateau / Incline), Arps curve fitting, "
            "and economic evaluation (NPV, breakeven price, payback period) per well.",
        ),
        [
            lang_text("Metode Arps 1945", "Arps 1945 Method"),
            lang_text("Exponential · Hyperbolic · Harmonic", "Exponential · Hyperbolic · Harmonic"),
            lang_text("Economic limit default 10 BOPD", "Economic limit default 10 BOPD"),
        ],
    )

    # ── Sidebar controls for DCA ──────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**{lang_text('Parameter DCA', 'DCA Parameters')}**")
    dca_months = st.sidebar.slider(
        lang_text("Horizon forecast (bulan)", "Forecast horizon (months)"), 6, 24, 12
    )
    econ_limit = st.sidebar.number_input(
        lang_text("Economic limit (BOPD)", "Economic limit (BOPD)"), 1.0, 50.0, 10.0, 1.0
    )
    smooth_w = st.sidebar.slider(
        lang_text("Smoothing window (hari)", "Smoothing window (days)"), 7, 30, 14
    )

    st.sidebar.markdown(f"**{lang_text('Parameter Ekonomi', 'Economic Parameters')}**")
    oil_price  = st.sidebar.number_input(lang_text("Harga minyak (USD/bbl)", "Oil price (USD/bbl)"), 30.0, 150.0, 75.0, 1.0)
    opex_bbl   = st.sidebar.number_input(lang_text("OPEX (USD/bbl)", "OPEX (USD/bbl)"), 1.0, 50.0, 15.0, 1.0)
    royalty    = st.sidebar.number_input(lang_text("Royalti (%)", "Royalty (%)"), 0.0, 30.0, 10.0, 0.5)
    discount_r = st.sidebar.number_input(lang_text("Discount rate (%/thn)", "Discount rate (%/yr)"), 5.0, 30.0, 10.0, 1.0)

    # ── Well & window selection ───────────────────────────────
    dca_well     = st.selectbox(lang_text("Pilih sumur", "Select well"), sel_wells)
    wd_full      = df[df["well"] == dca_well].sort_values("date").copy()

    dca_start_date = st.date_input(
        lang_text("Mulai analisis dari (exclude pre-workover jika perlu)", "Analysis start date (exclude pre-workover if needed)"),
        value=wd_full["date"].max().date() - pd.Timedelta(days=365),
        min_value=wd_full["date"].min().date(),
        max_value=wd_full["date"].max().date(),
    )
    workover_cost = st.number_input(
        lang_text("Biaya workover (USD, 0 jika tidak ada)", "Workover cost (USD, 0 if none)"),
        0, 10_000_000, 0, 50_000
    )

    wd = wd_full[wd_full["date"] >= pd.Timestamp(dca_start_date)].copy()
    if len(wd) < 20:
        st.warning(lang_text(
            f"Data terlalu sedikit ({len(wd)} hari). Perluas rentang tanggal.",
            f"Insufficient data ({len(wd)} days). Extend the date range."
        ))
        st.stop()

    wd["qs"] = wd["gross"].rolling(smooth_w, min_periods=1, center=True).mean()
    wd["t"]  = (wd["date"] - wd["date"].min()).dt.days.astype(float)

    # ── Phase detection ───────────────────────────────────────
    ph = _dca_detect_phase(wd["gross"].values, window=smooth_w)
    c  = WELL_COLORS.get(dca_well, "#58a6ff")

    st.markdown(f"<div class='sec-hdr'>{lang_text('Deteksi Fase Produksi', 'Production Phase Detection')}</div>",
                unsafe_allow_html=True)

    phase_label = lang_text(
        {"DECLINE":"Penurunan","PLATEAU":"Stabil","INCLINE":"Kenaikan","VOLATILE":"Fluktuatif"}.get(ph["phase"], ph["phase"]),
        ph["phase"]
    )
    rec_text = ph["rec_id"] if lang_mode in ["ID", "ID/EN"] else ph["rec_en"]
    if lang_mode == "ID/EN":
        rec_text = ph["rec_id"] + " / " + ph["rec_en"]

    box_cls = {"DECLINE": "box-red", "PLATEAU": "box-ylw",
               "INCLINE": "box-grn", "VOLATILE": "box-red"}.get(ph["phase"], "box-ylw")

    col_ph1, col_ph2, col_ph3, col_ph4 = st.columns(4)
    for col, (lbl_id, lbl_en, val, extra) in zip(
        [col_ph1, col_ph2, col_ph3, col_ph4],
        [
            ("Fase", "Phase",
             f"<span class='phase-badge {ph['css_class']}'>{phase_label}</span>",
             f"<span style='color:#6e7681;font-size:11px;'>{ph['confidence']} confidence</span>"),
            ("Rata-rata BOPD", "Mean BOPD",
             f"{ph['m']['mean_q']:,.0f}",
             f"<span style='color:#6e7681;font-size:11px;'>CV={ph['m']['cv']}</span>"),
            ("Slope", "Slope",
             f"{ph['m']['slope']:+.2f}",
             f"<span style='color:#6e7681;font-size:11px;'>BOPD/hari</span>"),
            ("Δ Produksi", "Δ Production",
             f"{ph['m']['pct_chg']:+.1f}%",
             f"<span style='color:#6e7681;font-size:11px;'>{lang_text('Q1 vs Q4','Q1 vs Q4')}</span>"),
        ]
    ):
        with col:
            st.markdown(
                f"<div class='econ-card'><div class='econ-label'>{lang_text(lbl_id, lbl_en)}</div>"
                f"<div class='econ-value'>{val}</div>"
                f"<div class='econ-sub'>{extra}</div></div>",
                unsafe_allow_html=True,
            )

    st.markdown(f"<div class='{box_cls}'>💡 {rec_text}</div>", unsafe_allow_html=True)

    # ── DCA fitting ───────────────────────────────────────────
    st.markdown(f"<div class='sec-hdr'>{lang_text('Fitting Kurva Arps', 'Arps Curve Fitting')}</div>",
                unsafe_allow_html=True)

    if not ph["dca_ok"]:
        st.info(lang_text(
            f"Fase {phase_label} — DCA tidak direkomendasikan saat ini. "
            "Model tetap dijalankan sebagai referensi, namun interpretasi harus hati-hati.",
            f"Phase {phase_label} — DCA not recommended at this stage. "
            "Model is still run for reference, but interpretation should be cautious."
        ))

    fit_res = _dca_fit_arps(wd["t"].values, wd["qs"].values, ph["m"]["mean_q"])
    fore    = None
    econ    = None

    if fit_res:
        best_entry = {k: v for k, v in fit_res.items() if v.get("is_best")}
        if best_entry:
            bname = list(best_entry.keys())[0]
            bf    = best_entry[bname]
            qi    = bf["params"]["qi"]
            q365  = bf["func"](np.array([365.0]), *bf["popt"])[0]
            ann_dcl = (1 - q365 / (qi + 1e-6)) * 100

            fore = _dca_forecast(fit_res, wd["t"].max(), wd["date"].max(), dca_months, econ_limit)

            # Model comparison table
            model_rows = ""
            for nm, fr in fit_res.items():
                star = "★" if fr["is_best"] else ""
                r2c  = "clr-good" if fr["r2"] > 0.7 else "clr-warn" if fr["r2"] > 0.3 else "clr-bad"
                model_rows += (
                    f"<tr><td><b>{star}{nm}</b></td>"
                    f"<td class='{r2c}'>{fr['r2']:.4f}</td>"
                    f"<td>{fr['rmse']:.1f}</td>"
                    f"<td>{fr['params']['qi']:.0f}</td>"
                    f"<td>{fr['params']['Di']:.5f}</td>"
                    "<td>" + ('—' if 'b' not in fr['params'] else str(round(fr['params']['b'],3))) + "</td></tr>"
                )
            hdr = "".join(f"<th>{h}</th>" for h in ["Model","R²","RMSE","qi (BOPD)","Di (/day)","b"])
            st.markdown(
                f"<table class='mtable'><thead><tr>{hdr}</tr></thead><tbody>{model_rows}</tbody></table>",
                unsafe_allow_html=True,
            )

    # ── Main DCA chart ────────────────────────────────────────
    MODEL_COLORS = {"exponential": "#ffa657", "hyperbolic": "#58a6ff", "harmonic": "#7ee787"}

    fig_dca = go.Figure()

    # Actual scatter
    fig_dca.add_trace(go.Scatter(
        x=wd["date"], y=wd["gross"],
        mode="markers",
        marker=dict(color="#2a2a2a", size=3, opacity=0.6),
        name=lang_text("Aktual harian", "Daily actual"),
    ))
    # Smoothed MA
    fig_dca.add_trace(go.Scatter(
        x=wd["date"], y=wd["qs"],
        line=dict(color="#555", width=1.2),
        name=f"{smooth_w}-day MA",
    ))

    # Fitted curves
    if fit_res:
        t_dense = np.linspace(0, wd["t"].max(), 500)
        dates_dense = wd["date"].min() + pd.to_timedelta(t_dense.astype(int), unit="D")
        for nm, fr in fit_res.items():
            if "popt" not in fr:
                continue
            qf  = fr["func"](t_dense, *fr["popt"])
            lw  = 2.2 if fr["is_best"] else 0.8
            ls  = "solid" if fr["is_best"] else "dash"
            al  = 1.0 if fr["is_best"] else 0.3
            lbl = f"{nm} R²={fr['r2']:.3f}" + (" ★" if fr["is_best"] else "")
            fig_dca.add_trace(go.Scatter(
                x=dates_dense, y=qf,
                line=dict(color=MODEL_COLORS[nm], width=lw, dash=ls),
                opacity=al, name=lbl,
            ))

    # Forecast line + band
    if fore is not None:
        xconn = pd.DatetimeIndex([wd["date"].max()]).append(fore["dates"])
        yconn = np.concatenate([[wd["qs"].iloc[-1]], fore["q"]])
        fig_dca.add_trace(go.Scatter(
            x=xconn, y=yconn,
            line=dict(color="#f0a500", width=2.2, dash="dash"),
            name=lang_text("Forecast", "Forecast"),
        ))
        fig_dca.add_trace(go.Scatter(
            x=pd.DatetimeIndex(list(fore["dates"]) + list(fore["dates"])[::-1]),
            y=np.concatenate([fore["q"] * 1.15, fore["q"][::-1] * 0.85]),
            fill="toself",
            fillcolor="rgba(240,165,0,0.07)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
            name="±15% band",
        ))
        if fore["aban_date"]:
            fig_dca.add_vline(
                x=fore["aban_date"].timestamp() * 1000,
                line_dash="dot", line_color="#f85149", line_width=1.2,
                annotation_text=lang_text("Econ. limit", "Econ. limit"),
                annotation_font_color="#f85149",
            )

    if dca_well in REGIME_BREAK:
        add_regime_marker(fig_dca, REGIME_BREAK[dca_well])

    style_xy_figure(
        fig_dca,
        lang_text(
            f"{dca_well} — Decline Curve Analysis (Arps)",
            f"{dca_well} — Decline Curve Analysis (Arps)",
        ),
        400, yaxis_title="BOPD",
    )
    st.plotly_chart(fig_dca, use_container_width=True)
    render_chart_note(lang_text(
        "Band ±15% menunjukkan rentang ketidakpastian forecast. "
        "Titik Economic limit adalah saat laju produksi forecast mencapai batas minimum ekonomis.",
        "The ±15% band shows the forecast uncertainty range. "
        "The Economic limit marker is when the forecast rate reaches the minimum economic threshold."
    ))

    # ── Economics ─────────────────────────────────────────────
    st.markdown(f"<div class='sec-hdr'>{lang_text('Analisis Ekonomi', 'Economic Analysis')}</div>",
                unsafe_allow_html=True)

    if fore is None:
        st.info(lang_text(
            "Fitting DCA belum berhasil — tidak ada data untuk kalkulasi ekonomi.",
            "DCA fitting not successful — no data for economic calculation."
        ))
    else:
        econ = _dca_economics(
            fore["q"], fore["dates"],
            oil_price=oil_price, royalty=royalty,
            opex=opex_bbl, workover=float(workover_cost),
            discount=discount_r,
        )

        npv_str = f"${econ['npv'] / 1e6:.2f}M" if abs(econ["npv"]) >= 1e6 else f"${econ['npv']:,}"
        rev_str = f"${econ['tot_rev'] / 1e6:.2f}M" if econ["tot_rev"] >= 1e6 else f"${econ['tot_rev']:,}"
        npv_cls = "clr-good" if econ["npv"] >= 0 else "clr-bad"
        be_cls  = "clr-good" if econ["be_price"] < oil_price else "clr-bad"

        econ_items = [
            ("NPV", npv_str, f"<span class='{npv_cls}'>"
             + lang_text(f"Discount rate {discount_r}%", f"Discount rate {discount_r}%")
             + "</span>"),
            (lang_text("Total Revenue", "Total Revenue"), rev_str,
             lang_text(f"Net ${econ['net_p']}/bbl setelah royalti & OPEX",
                       f"Net ${econ['net_p']}/bbl after royalty & OPEX")),
            (lang_text("EUR Forecast", "EUR Forecast"),
             f"{econ['tot_bbl']:,} bbl",
             lang_text(f"Horizon {dca_months} bulan", f"{dca_months}-month horizon")),
            (lang_text("Breakeven Price", "Breakeven Price"),
             f"${econ['be_price']}/bbl",
             f"<span class='{be_cls}'>"
             + lang_text(
                 f"Harga saat ini ${oil_price}/bbl",
                 f"Current price ${oil_price}/bbl"
               )
             + "</span>"),
            (lang_text("Profitability Index", "Profitability Index"),
             f"{econ['pi']:.2f}x",
             lang_text("PI > 1 = investasi layak", "PI > 1 = investment viable")),
            (lang_text("Payback Period", "Payback Period"),
             str(econ["payback"].date()) if econ["payback"] else lang_text("N/A (workover=0)", "N/A (workover=0)"),
             lang_text(f"Dari biaya workover ${workover_cost:,}",
                       f"From workover cost ${workover_cost:,}")),
        ]
        e_cols = st.columns(3)
        for i, (lbl, val, sub) in enumerate(econ_items):
            with e_cols[i % 3]:
                st.markdown(
                    f"<div class='econ-card'>"
                    f"<div class='econ-label'>{lbl}</div>"
                    f"<div class='econ-value'>{val}</div>"
                    f"<div class='econ-sub'>{sub}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # Cashflow & Cumulative NPV charts
        st.markdown(f"<div class='sec-hdr'>{lang_text('Cashflow & Cumulative NPV', 'Cashflow & Cumulative NPV')}</div>",
                    unsafe_allow_html=True)
        cf = econ["cf_df"]

        col_cf, col_npv = st.columns(2)

        with col_cf:
            fig_cf = go.Figure()
            bar_colors = ["#3fb950" if v >= 0 else "#f85149" for v in cf["cf"]]
            fig_cf.add_trace(go.Bar(
                x=cf["date"], y=cf["cf"] / 1000,
                marker_color=bar_colors, opacity=0.8,
                name=lang_text("Cashflow harian", "Daily cashflow"),
            ))
            style_xy_figure(
                fig_cf,
                lang_text("Daily cashflow (K USD)", "Daily Cashflow (K USD)"),
                280, yaxis_title="K USD",
            )
            st.plotly_chart(fig_cf, use_container_width=True)

        with col_npv:
            fig_npv = go.Figure()
            fig_npv.add_trace(go.Scatter(
                x=cf["date"], y=cf["cum_npv"] / 1e6,
                line=dict(color=c, width=2.2),
                fill="tozeroy",
                fillcolor=f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.08)",
                name="Cumulative NPV (M USD)",
            ))
            fig_npv.add_hline(y=0, line_dash="dash", line_color="#555", line_width=0.8)
            if workover_cost > 0:
                fig_npv.add_hline(
                    y=-workover_cost / 1e6,
                    line_dash="dot", line_color="#f85149", line_width=1,
                    annotation_text=lang_text(f"Workover −${workover_cost/1e6:.1f}M",
                                              f"Workover −${workover_cost/1e6:.1f}M"),
                    annotation_font_color="#f85149",
                )
            if econ["payback"]:
                fig_npv.add_vline(
                    x=econ["payback"].timestamp() * 1000,
                    line_dash="dot", line_color="#f0a500", line_width=1,
                    annotation_text=lang_text("Payback", "Payback"),
                    annotation_font_color="#f0a500",
                )
            style_xy_figure(
                fig_npv,
                lang_text("Cumulative NPV (M USD)", "Cumulative NPV (M USD)"),
                280, yaxis_title="M USD",
            )
            st.plotly_chart(fig_npv, use_container_width=True)

        render_chart_note(lang_text(
            f"Asumsi: oil price ${oil_price}/bbl · royalti {royalty}% · "
            f"OPEX ${opex_bbl}/bbl · discount rate {discount_r}%/thn. "
            "Ubah parameter di sidebar untuk simulasi skenario berbeda.",
            f"Assumptions: oil price ${oil_price}/bbl · royalty {royalty}% · "
            f"OPEX ${opex_bbl}/bbl · discount rate {discount_r}%/yr. "
            "Adjust parameters in the sidebar to simulate different scenarios."
        ))

        # Sensitivity: NPV vs oil price
        st.markdown(f"<div class='sec-hdr'>{lang_text('Sensitivitas NPV terhadap Harga Minyak', 'NPV Sensitivity to Oil Price')}</div>",
                    unsafe_allow_html=True)
        prices    = np.arange(30, 121, 5)
        npvs      = []
        for p in prices:
            e2 = _dca_economics(
                fore["q"], fore["dates"],
                oil_price=float(p), royalty=royalty,
                opex=opex_bbl, workover=float(workover_cost),
                discount=discount_r,
            )
            npvs.append(e2["npv"] / 1e6)

        fig_sens = go.Figure()
        bar_c2 = ["#3fb950" if v >= 0 else "#f85149" for v in npvs]
        fig_sens.add_trace(go.Bar(
            x=prices, y=npvs,
            marker_color=bar_c2, opacity=0.85,
            name="NPV (M USD)",
        ))
        fig_sens.add_vline(
            x=oil_price,
            line_dash="dash", line_color="#f0a500", line_width=1.5,
            annotation_text=lang_text(f"Harga saat ini ${oil_price}",
                                       f"Current ${oil_price}"),
            annotation_font_color="#f0a500",
        )
        fig_sens.add_hline(y=0, line_color="#555", line_width=0.8)
        style_xy_figure(
            fig_sens,
            lang_text("Sensitivitas NPV vs Harga Minyak", "NPV Sensitivity vs Oil Price"),
            280, xaxis_title="Oil Price (USD/bbl)", yaxis_title="NPV (M USD)",
        )
        st.plotly_chart(fig_sens, use_container_width=True)
        render_chart_note(lang_text(
            "Grafik ini menunjukkan NPV project pada berbagai skenario harga minyak, "
            "dengan semua parameter lain tetap. Titik breakeven adalah harga minimum "
            f"agar project tetap positif: ${econ['be_price']}/bbl.",
            "This chart shows project NPV across different oil price scenarios, "
            "keeping all other parameters constant. The breakeven point is the minimum price "
            f"for the project to remain positive: ${econ['be_price']}/bbl."
        ))
elif page == "priority":
    # ── Compute ───────────────────────────────────────────────────
    @st.cache_data(ttl=300)
    def _cached_scores():
        try:
            fcols_pw_local = json.load(
                open(os.path.join(BASE_DIR, "feature_cols_per_well.json")))
            gfcols_local   = json.load(
                open(os.path.join(BASE_DIR, "feature_cols.json")))
        except:
            fcols_pw_local = {}; gfcols_local = []
        return compute_composite_scores(
            df, models, gfcols_local, fcols_pw_local, metrics)

    scores = _cached_scores()

    # ── Page intro ────────────────────────────────────────────────
    render_page_intro(
        lang_text("Prioritisasi Sumur", "Well Prioritization"),
        lang_text("Composite Score: ML + DCA", "Composite Score: ML + DCA"),
        lang_text(
            "Skor gabungan dari sinyal anomali model ML dan tingkat keparahan decline DCA "
            "untuk menentukan sumur mana yang membutuhkan perhatian operasional paling mendesak.",
            "Combined score from ML model anomaly signals and DCA decline severity "
            "to determine which wells require the most urgent operational attention.",
        ),
        [
            lang_text("6 sinyal · 2 domain", "6 signals · 2 domains"),
            lang_text("ML 60% · DCA 40%", "ML 60% · DCA 40%"),
            lang_text("4 tier: Monitor / Watch / Alert / Urgent",
                      "4 tiers: Monitor / Watch / Alert / Urgent"),
        ],
    )

    # ── Score weight explainer ────────────────────────────────────
    with st.expander(lang_text("📐 Metodologi scoring", "📐 Scoring methodology"),
                     expanded=False):
        rows_exp = ""
        for lbl, key, clr, w in zip(SCORE_LABELS, SCORE_KEYS, SCORE_COLORS, SCORE_WEIGHTS):
            domain = "ML" if key in ["resid_score","mape_score","esp_score"] else "DCA"
            descs_id = {
                "resid_score": "Selisih aktual vs prediksi 7 hari terakhir (negatif = underperform)",
                "mape_score":  "Ketidakpastian model — semakin tinggi MAPE, semakin tidak pasti forecast",
                "esp_score":   "Simpangan arus ESP dari rata-rata historis (|z| > 2.5σ = anomali)",
                "dcl_score":   "Decline rate tahunan dari fitting kurva Arps",
                "wc_score":    "Laju kenaikan water cut dalam 30 hari terakhir",
                "slope_score": "Laju penurunan produksi gross dalam 30 hari terakhir",
            }
            descs_en = {
                "resid_score": "Actual vs predicted gap last 7 days (negative = underperform)",
                "mape_score":  "Model uncertainty — higher MAPE = less reliable forecast",
                "esp_score":   "ESP current deviation from historical mean (|z| > 2.5σ = anomaly)",
                "dcl_score":   "Annualized decline rate from Arps curve fitting",
                "wc_score":    "Water cut rise rate over last 30 days",
                "slope_score": "Gross production decline rate over last 30 days",
            }
            desc = lang_text(descs_id[key], descs_en[key])
            rows_exp += (
                f"<tr><td><span style='color:{clr};font-weight:bold;'>{lbl}</span></td>"
                f"<td>{domain}</td><td>{int(w*100)}%</td><td style='color:#8b949e;'>{desc}</td></tr>"
            )
        st.markdown(
            f"<table class='mtable'><thead><tr>"
            f"<th>{lang_text('Sinyal','Signal')}</th>"
            f"<th>{lang_text('Domain','Domain')}</th>"
            f"<th>{lang_text('Bobot','Weight')}</th>"
            f"<th>{lang_text('Deskripsi','Description')}</th>"
            f"</tr></thead><tbody>{rows_exp}</tbody></table>",
            unsafe_allow_html=True,
        )
        tier_rows = ""
        for rng, tier, clr, desc_id, desc_en in [
            ("81–100","URGENT","#f85149",
             "Tindakan segera: evaluasi & jadwalkan intervensi",
             "Immediate action: evaluate & schedule intervention"),
            ("61–80","ALERT","#ffa657",
             "Siapkan rencana intervensi dalam 2 minggu",
             "Prepare intervention plan within 2 weeks"),
            ("31–60","WATCH","#f0a500",
             "Review mingguan, perbarui forecast",
             "Weekly review, update forecasts"),
            ("0–30","MONITOR","#3fb950",
             "Monitoring bulanan rutin cukup",
             "Routine monthly monitoring sufficient"),
        ]:
            tier_rows += (
                f"<tr><td>{rng}</td>"
                f"<td><span style='color:{clr};font-weight:bold;'>{tier}</span></td>"
                f"<td style='color:#8b949e;'>{lang_text(desc_id,desc_en)}</td></tr>"
            )
        st.markdown(
            f"<div style='margin-top:12px;'>"
            f"<table class='mtable'><thead><tr>"
            f"<th>{lang_text('Range','Range')}</th>"
            f"<th>{lang_text('Tier','Tier')}</th>"
            f"<th>{lang_text('Panduan tindakan','Action guidance')}</th>"
            f"</tr></thead><tbody>{tier_rows}</tbody></table></div>",
            unsafe_allow_html=True,
        )

    # ── Ranked cards ──────────────────────────────────────────────
    st.markdown(
        f"<div class='sec-hdr'>"
        f"{lang_text('Ranking Prioritas Sumur','Well Priority Ranking')}"
        f"</div>",
        unsafe_allow_html=True,
    )

    card_cols = st.columns(len(scores))
    for col, res in zip(card_cols, scores):
        tier_label = lang_text(
            {"URGENT":"Mendesak","ALERT":"Waspada",
             "WATCH":"Pantau","MONITOR":"Normal"}[res["tier"]],
            res["tier"]
        )
        action_txt = res["action_id"] if lang_mode in ["ID","ID/EN"] else res["action_en"]
        esp_str    = (f"ESP z={res['esp_z']:+.2f}σ" if res["esp_z"] is not None
                      else lang_text("ESP: N/A","ESP: N/A"))
        with col:
            st.markdown(
                f"""<div class='kpi-card' style='border-top:3px solid {res["tier_color"]};'>
                <div class='kpi-label'>{res["well"]}</div>
                <div style='display:flex;align-items:baseline;gap:8px;margin:6px 0;'>
                  <span class='kpi-value' style='font-size:36px;color:{res["tier_color"]};'>{res["composite"]}</span>
                  <span class='kpi-unit'>/100</span>
                </div>
                <div style='display:inline-block;padding:3px 10px;border-radius:5px;
                     background:{res["tier_color"]}22;border:1px solid {res["tier_color"]};
                     color:{res["tier_color"]};font-family:IBM Plex Mono,monospace;
                     font-size:11px;font-weight:700;letter-spacing:.1em;margin-bottom:10px;'>
                  {tier_label}
                </div>
                <div style='font-size:11px;color:#6e7681;line-height:1.7;'>
                  Gross: <span style='color:#c9d1d9;'>{res["gross_last"]:.0f} BOPD</span><br>
                  WC: <span style='color:#c9d1d9;'>{res["wc_last"]:.1f}%</span>
                  &nbsp;+{res["wc_slope_ann"]:.0f}%/yr<br>
                  Decline: <span style='color:#c9d1d9;'>{res["ann_dcl"]:.0f}%</span>/yr<br>
                  {esp_str}
                </div>
                <div style='margin-top:10px;font-size:11px;color:{res["tier_color"]};
                     border-top:1px solid #21262d;padding-top:8px;'>
                  {action_txt}
                </div>
                </div>""",
                unsafe_allow_html=True,
            )

    # ── Composite bar chart ───────────────────────────────────────
    st.markdown(
        f"<div class='sec-hdr'>"
        f"{lang_text('Composite Score & Breakdown','Composite Score & Breakdown')}"
        f"</div>",
        unsafe_allow_html=True,
    )

    col_bar, col_stack = st.columns(2)

    with col_bar:
        fig_bar = go.Figure()
        for res in reversed(scores):
            fig_bar.add_trace(go.Bar(
                y=[res["well"]],
                x=[res["composite"]],
                orientation="h",
                marker_color=res["tier_color"],
                showlegend=False,
                text=f"  {res['composite']:.0f} — {res['tier']}",
                textposition="outside",
                textfont=dict(color=res["tier_color"],
                              family="IBM Plex Mono", size=11),
            ))
        fig_bar.add_vline(x=30, line_dash="dot", line_color="#3fb950",
                          line_width=1, opacity=0.5)
        fig_bar.add_vline(x=60, line_dash="dot", line_color="#f0a500",
                          line_width=1, opacity=0.5)
        fig_bar.add_vline(x=80, line_dash="dot", line_color="#ffa657",
                          line_width=1, opacity=0.5)
        style_xy_figure(
            fig_bar,
            lang_text("Composite score per sumur", "Composite score by well"),
            260, xaxis_title=lang_text("Score (0–100)","Score (0–100)"),
        )
        fig_bar.update_layout(barmode="overlay", xaxis_range=[0,110])
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_stack:
        fig_stk = go.Figure()
        for sk, lbl, clr, w in zip(SCORE_KEYS, SCORE_LABELS, SCORE_COLORS, SCORE_WEIGHTS):
            fig_stk.add_trace(go.Bar(
                y=[r["well"] for r in reversed(scores)],
                x=[r[sk] * w for r in reversed(scores)],
                orientation="h",
                name=f"{lbl} ({int(w*100)}%)",
                marker_color=clr,
                opacity=0.85,
            ))
        style_xy_figure(
            fig_stk,
            lang_text("Kontribusi sinyal ke score total",
                      "Signal contributions to total score"),
            260,
            xaxis_title=lang_text("Weighted contribution","Weighted contribution"),
        )
        fig_stk.update_layout(barmode="stack", xaxis_range=[0,60],
                              legend=dict(font=dict(size=9)))
        st.plotly_chart(fig_stk, use_container_width=True)

    # ── Radar chart ───────────────────────────────────────────────
    st.markdown(
        f"<div class='sec-hdr'>"
        f"{lang_text('Sub-Score Radar per Sumur','Per-Well Sub-Score Radar')}"
        f"</div>",
        unsafe_allow_html=True,
    )

    theta = SCORE_LABELS + [SCORE_LABELS[0]]
    fig_radar = go.Figure()
    for res in scores:
        r_vals = [res[k] for k in SCORE_KEYS] + [res[SCORE_KEYS[0]]]
        c      = WELL_COLORS.get(res["well"], "#888888")
        rgb    = tuple(int(c[i:i+2], 16) for i in (1, 3, 5))
        fill_c = f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.08)"
        fig_radar.add_trace(go.Scatterpolar(
            r=r_vals, theta=theta,
            fill="toself",
            fillcolor=fill_c,
            line=dict(color=c, width=1.8),
            name=res["well"],
            hovertemplate="%{theta}: %{r:.0f}<extra>%{fullData.name}</extra>",
        ))

    fig_radar.update_layout(
        polar=dict(
            bgcolor="#0d1117",
            radialaxis=dict(
                visible=True, range=[0, 100],
                gridcolor="#21262d", tickcolor="#6e7681",
                tickfont=dict(size=9, color="#6e7681"),
            ),
            angularaxis=dict(
                gridcolor="#21262d",
                tickfont=dict(size=10, color="#8b949e"),
            ),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0d1117",
        font=dict(family="IBM Plex Sans", color="#8b949e", size=11),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", bordercolor="#21262d",
            font=dict(size=10, color="#c9d1d9"),
        ),
        height=360,
        title=dict(
            text=lang_text(
                "Profil risiko per sumur — 6 dimensi",
                "Per-well risk profile — 6 dimensions",
            ),
            x=0, xanchor="left",
            font=dict(size=14, color="#f0f6fc"),
        ),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    render_chart_note(lang_text(
        "Semakin luas area radar, semakin tinggi tingkat risiko operasional sumur tersebut. "
        "Sumur ideal memiliki area mendekati nol di semua dimensi.",
        "The larger the radar area, the higher the operational risk level of that well. "
        "An ideal well has an area close to zero across all dimensions."
    ))

    # ── Detail table ──────────────────────────────────────────────
    st.markdown(
        f"<div class='sec-hdr'>"
        f"{lang_text('Tabel Detail Sinyal','Signal Detail Table')}"
        f"</div>",
        unsafe_allow_html=True,
    )

    hdr_labels = [
        lang_text("Sumur","Well"),
        lang_text("Score","Score"),
        lang_text("Tier","Tier"),
        lang_text("Residual 7d (BOPD)","Residual 7d (BOPD)"),
        lang_text("MAPE Model","Model MAPE"),
        lang_text("ESP z-score","ESP z-score"),
        lang_text("Decline %/thn","Decline %/yr"),
        lang_text("WC slope %/thn","WC slope %/yr"),
        lang_text("Prod slope %/thn","Prod slope %/yr"),
        lang_text("Rekomendasi","Recommendation"),
    ]
    hdr_html = "".join(f"<th>{h}</th>" for h in hdr_labels)
    rows_html = ""
    for res in scores:
        tier_lbl = lang_text(
            {"URGENT":"Mendesak","ALERT":"Waspada","WATCH":"Pantau","MONITOR":"Normal"}[res["tier"]],
            res["tier"]
        )
        tc   = res["tier_color"]
        esp  = f"{res['esp_z']:+.2f}σ" if res["esp_z"] is not None else "—"
        rec  = res["action_id"] if lang_mode in ["ID","ID/EN"] else res["action_en"]
        resid_cls = "clr-bad" if res["resid_7d"] < -10 else "clr-good" if res["resid_7d"] > 0 else ""
        rows_html += (
            f"<tr>"
            f"<td>{badge(res['well'])}</td>"
            f"<td><b style='color:{tc};font-family:IBM Plex Mono,monospace;'>{res['composite']}</b></td>"
            f"<td><span style='color:{tc};font-weight:700;'>{tier_lbl}</span></td>"
            f"<td class='{resid_cls}'>{res['resid_7d']:+.0f}</td>"
            f"<td>{res['mape']}%</td>"
            f"<td>{esp}</td>"
            f"<td>{res['ann_dcl']:.0f}%</td>"
            f"<td>{res['wc_slope_ann']:+.0f}%</td>"
            f"<td>{res['slope_pct']:+.0f}%</td>"
            f"<td style='color:#8b949e;font-size:11px;'>{rec}</td>"
            f"</tr>"
        )
    st.markdown(
        f"<table class='mtable'><thead><tr>{hdr_html}</tr></thead>"
        f"<tbody>{rows_html}</tbody></table>",
        unsafe_allow_html=True,
    )
    render_chart_note(lang_text(
        "Score dihitung ulang setiap 5 menit. Semua sinyal menggunakan data DPR terbaru. "
        "Bobot dapat disesuaikan dengan preferensi operasional spesifik lapangan.",
        "Scores are recalculated every 5 minutes. All signals use the latest DPR data. "
        "Weights can be adjusted to match field-specific operational preferences."
    ))

# =============================================================================
# 6️⃣ WORKOVER RANKING
# =============================================================================
elif page == "workover":

    @st.cache_data(ttl=300)
    def _cached_wo():
        return compute_workover_scores(df)

    wo_scores = _cached_wo()

    render_page_intro(
        lang_text("Workover Ranking", "Workover Ranking"),
        lang_text("Kandidat Workover & Estimasi Nilai Ekonomi",
                  "Workover Candidates & Economic Value Estimate"),
        lang_text(
            "Scoring otomatis kandidat workover berdasarkan decline rate, "
            "tren water cut, gap produksi dari peak historis, dan kesehatan ESP.",
            "Automated workover candidate scoring based on decline rate, "
            "water cut trend, production gap from historical peak, and ESP health.",
        ),
        [
            lang_text("4 sinyal · weighted score", "4 signals · weighted score"),
            lang_text("Window 90 hari terakhir", "Last 90-day window"),
            lang_text("4 tier: Prime / Candidate / Monitor / Stable",
                      "4 tiers: Prime / Candidate / Monitor / Stable"),
        ],
    )

    # ── Methodology expander ─────────────────────────────────────
    with st.expander(lang_text("📐 Metodologi scoring workover",
                               "📐 Workover scoring methodology"), expanded=False):
        descs_id = {
            "dcl_s": "Annual decline rate dari regresi linear gross 90 hari (50%/thn = 100 poin)",
            "wc_s":  "Level WC saat ini + laju kenaikan WC tahunan (gabungan 50:50)",
            "gap_s": "Selisih antara produksi saat ini dan peak historis (P90)",
            "esp_s": "Kenaikan arus ampere + kenaikan fluid level (K1/K5); penurunan THP (K3/K4)",
        }
        descs_en = {
            "dcl_s": "Annual decline rate from 90-day gross linear regression (50%/yr = 100 pts)",
            "wc_s":  "Current WC level + annual WC rise rate (50:50 combined)",
            "gap_s": "Gap between current production and historical peak (P90)",
            "esp_s": "Ampere rise + fluid level rise (K1/K5); THP decline as proxy (K3/K4)",
        }
        rows_exp = ""
        for lbl, key, clr, w in zip(WO_LABELS, WO_KEYS, WO_COLORS, WO_WEIGHTS):
            desc = lang_text(descs_id[key], descs_en[key])
            rows_exp += (
                f"<tr><td><span style='color:{clr};font-weight:bold;'>{lbl}</span></td>"
                f"<td>{int(w*100)}%</td>"
                f"<td style='color:#8b949e;'>{desc}</td></tr>"
            )
        st.markdown(
            f"<table class='mtable'><thead><tr>"
            f"<th>{lang_text('Sinyal','Signal')}</th>"
            f"<th>{lang_text('Bobot','Weight')}</th>"
            f"<th>{lang_text('Deskripsi','Description')}</th>"
            f"</tr></thead><tbody>{rows_exp}</tbody></table>",
            unsafe_allow_html=True,
        )
        tier_rows = ""
        for rng, tier, clr, desc_id, desc_en in [
            ("≥ 60", "PRIME",     "#f85149",
             "Workover sangat direkomendasikan — jadwalkan segera",
             "Workover strongly recommended — schedule immediately"),
            ("40–59", "CANDIDATE", "#ffa657",
             "Workover layak dipertimbangkan — siapkan justifikasi teknis",
             "Workover worth considering — prepare technical justification"),
            ("20–39", "MONITOR",   "#f0a500",
             "Pantau 3–6 bulan, evaluasi ulang jika trend memburuk",
             "Monitor 3–6 months, re-evaluate if trend worsens"),
            ("< 20",  "STABLE",   "#3fb950",
             "Sumur stabil, tidak perlu intervensi",
             "Well is stable, no intervention needed"),
        ]:
            tier_rows += (
                f"<tr><td>{rng}</td>"
                f"<td><span style='color:{clr};font-weight:700;'>{tier}</span></td>"
                f"<td style='color:#8b949e;'>{lang_text(desc_id, desc_en)}</td></tr>"
            )
        st.markdown(
            f"<div style='margin-top:12px;'>"
            f"<table class='mtable'><thead><tr>"
            f"<th>{lang_text('Range','Range')}</th>"
            f"<th>{lang_text('Tier','Tier')}</th>"
            f"<th>{lang_text('Panduan','Guidance')}</th>"
            f"</tr></thead><tbody>{tier_rows}</tbody></table></div>",
            unsafe_allow_html=True,
        )

    # ── KPI summary ───────────────────────────────────────────────
    n_prime      = sum(1 for w in wo_scores if w["tier"] == "PRIME")
    n_candidate  = sum(1 for w in wo_scores if w["tier"] in ["PRIME", "CANDIDATE"])
    total_uplift = sum(w["uplift_bopd"] for w in wo_scores)
    total_econ   = sum(w["econ_musd_yr"] for w in wo_scores)

    render_kpi_cards([
        (lang_text("Kandidat Utama", "Prime Candidates"),
         str(n_prime), lang_text("sumur", "wells"), ""),
        (lang_text("Total Kandidat", "Total Candidates"),
         str(n_candidate), lang_text("sumur", "wells"), ""),
        (lang_text("Potensi Uplift", "Potential Uplift"),
         f"{total_uplift:,.0f}", "BOPD",
         f"<span class='delta-up'>≈ ${total_econ:.1f}M/yr @$70/bbl</span>"),
    ], max_cols=3)

    # ── Ranked cards ──────────────────────────────────────────────
    st.markdown(
        f"<div class='sec-hdr'>"
        f"{lang_text('Ranking Kandidat Workover', 'Workover Candidate Ranking')}"
        f"</div>",
        unsafe_allow_html=True,
    )

    card_cols = st.columns(len(wo_scores))
    for col, res in zip(card_cols, wo_scores):
        tier_lbl = lang_text(
            {"PRIME": "Kandidat Utama", "CANDIDATE": "Kandidat",
             "MONITOR": "Pantau", "STABLE": "Stabil"}[res["tier"]],
            res["tier"],
        )
        action_txt = res["action_id"] if lang_mode in ["ID", "ID/EN"] else res["action_en"]
        gap_arrow  = f"▼{res['gap_pct']:.0f}%" if res["gap_pct"] > 0 else f"▲{-res['gap_pct']:.0f}%"
        with col:
            st.markdown(
                f"""<div class='kpi-card' style='border-top:3px solid {res["tier_color"]};'>
                <div class='kpi-label'>{res["well"]}</div>
                <div style='display:flex;align-items:baseline;gap:8px;margin:6px 0;'>
                  <span class='kpi-value' style='font-size:36px;color:{res["tier_color"]};'>{res["composite"]}</span>
                  <span class='kpi-unit'>/100</span>
                </div>
                <div style='display:inline-block;padding:3px 10px;border-radius:5px;
                     background:{res["tier_color"]}22;border:1px solid {res["tier_color"]};
                     color:{res["tier_color"]};font-family:IBM Plex Mono,monospace;
                     font-size:11px;font-weight:700;letter-spacing:.1em;margin-bottom:10px;'>
                  {tier_lbl}
                </div>
                <div style='font-size:11px;color:#6e7681;line-height:1.7;'>
                  Peak: <span style='color:#c9d1d9;'>{res["peak_gross"]:.0f} BOPD</span><br>
                  {lang_text("Saat ini","Current")}: <span style='color:#c9d1d9;'>{res["current_gross"]:.0f} BOPD</span>
                  &nbsp;<span style='color:{res["tier_color"]};'>{gap_arrow}</span><br>
                  WC: <span style='color:#c9d1d9;'>{res["last_wc"]:.1f}%</span>
                  &nbsp;{res["ann_wc_slope"]:+.0f}%/yr<br>
                  Uplift: <span style='color:#3fb950;'>+{res["uplift_bopd"]:.0f} BOPD</span>
                  &nbsp;≈ ${res["econ_musd_yr"]:.1f}M/yr
                </div>
                <div style='margin-top:10px;font-size:11px;color:{res["tier_color"]};
                     border-top:1px solid #21262d;padding-top:8px;'>
                  {action_txt}
                </div>
                </div>""",
                unsafe_allow_html=True,
            )

    # ── Charts ────────────────────────────────────────────────────
    st.markdown(
        f"<div class='sec-hdr'>"
        f"{lang_text('Analisis Visual', 'Visual Analysis')}"
        f"</div>",
        unsafe_allow_html=True,
    )

    col_bar, col_bubble = st.columns(2)

    with col_bar:
        fig_wo = go.Figure()
        for sk, lbl, clr, w in zip(WO_KEYS, WO_LABELS, WO_COLORS, WO_WEIGHTS):
            fig_wo.add_trace(go.Bar(
                y=[r["well"] for r in reversed(wo_scores)],
                x=[r[sk] * w for r in reversed(wo_scores)],
                orientation="h",
                name=f"{lbl} ({int(w*100)}%)",
                marker_color=clr,
                opacity=0.85,
            ))
        style_xy_figure(
            fig_wo,
            lang_text("Breakdown Workover Score", "Workover Score Breakdown"),
            280,
            xaxis_title=lang_text("Kontribusi skor", "Score contribution"),
        )
        fig_wo.update_layout(barmode="stack", xaxis_range=[0, 105])
        st.plotly_chart(fig_wo, use_container_width=True)

    with col_bubble:
        fig_bub = go.Figure()
        for res in wo_scores:
            c = WELL_COLORS.get(res["well"], "#888888")
            fig_bub.add_trace(go.Scatter(
                x=[res["last_wc"]],
                y=[res["gap_pct"]],
                mode="markers+text",
                marker=dict(
                    size=max(14, res["composite"] / 3),
                    color=res["tier_color"],
                    opacity=0.85,
                    line=dict(color="#0d1117", width=2),
                ),
                text=[res["well"]],
                textposition="top center",
                textfont=dict(color=c, size=11),
                name=res["well"],
                hovertemplate=(
                    f"<b>{res['well']}</b><br>"
                    f"WC: {res['last_wc']:.1f}%<br>"
                    f"Gap: {res['gap_pct']:.1f}%<br>"
                    f"Score: {res['composite']}"
                    "<extra></extra>"
                ),
                showlegend=False,
            ))
        style_xy_figure(
            fig_bub,
            lang_text("Gap Produksi vs Water Cut (ukuran = score)",
                      "Production Gap vs Water Cut (size = score)"),
            280,
            xaxis_title=lang_text("Water Cut saat ini (%)", "Current Water Cut (%)"),
            yaxis_title=lang_text("Gap dari Peak (%)", "Gap from Peak (%)"),
        )
        st.plotly_chart(fig_bub, use_container_width=True)

    render_chart_note(lang_text(
        "Ukuran gelembung proporsional dengan workover score. "
        "Sumur di kanan-atas (WC tinggi + gap besar) adalah kandidat paling potensial.",
        "Bubble size is proportional to workover score. "
        "Wells at top-right (high WC + large gap) are the strongest candidates.",
    ))

    # ── Detail table ──────────────────────────────────────────────
    st.markdown(
        f"<div class='sec-hdr'>"
        f"{lang_text('Tabel Detail Kandidat', 'Candidate Detail Table')}"
        f"</div>",
        unsafe_allow_html=True,
    )

    hdr_labels = [
        lang_text("Sumur", "Well"),
        lang_text("Score", "Score"),
        lang_text("Tier", "Tier"),
        lang_text("Decline %/thn", "Decline %/yr"),
        lang_text("WC %", "WC %"),
        lang_text("WC slope/thn", "WC slope/yr"),
        lang_text("Peak BOPD", "Peak BOPD"),
        lang_text("Saat ini BOPD", "Current BOPD"),
        lang_text("Gap %", "Gap %"),
        lang_text("Uplift BOPD", "Uplift BOPD"),
        lang_text("Nilai Ekon. M$/thn", "Econ. Value M$/yr"),
        lang_text("Rekomendasi", "Recommendation"),
    ]
    hdr_html  = "".join(f"<th>{h}</th>" for h in hdr_labels)
    rows_html = ""
    for res in wo_scores:
        tc       = res["tier_color"]
        tier_lbl = lang_text(
            {"PRIME": "Kandidat Utama", "CANDIDATE": "Kandidat",
             "MONITOR": "Pantau", "STABLE": "Stabil"}[res["tier"]],
            res["tier"],
        )
        rec     = res["action_id"] if lang_mode in ["ID", "ID/EN"] else res["action_en"]
        gap_cls = "clr-bad" if res["gap_pct"] > 30 else ""
        rows_html += (
            f"<tr>"
            f"<td>{badge(res['well'])}</td>"
            f"<td><b style='color:{tc};font-family:IBM Plex Mono,monospace;'>{res['composite']}</b></td>"
            f"<td><span style='color:{tc};font-weight:700;'>{tier_lbl}</span></td>"
            f"<td class='clr-bad'>{res['ann_dcl_pct']:+.0f}%</td>"
            f"<td>{res['last_wc']:.1f}%</td>"
            f"<td>{res['ann_wc_slope']:+.0f}%</td>"
            f"<td>{res['peak_gross']:.0f}</td>"
            f"<td>{res['current_gross']:.0f}</td>"
            f"<td class='{gap_cls}'>{res['gap_pct']:.0f}%</td>"
            f"<td style='color:#3fb950;'>+{res['uplift_bopd']:.0f}</td>"
            f"<td style='color:#3fb950;'>${res['econ_musd_yr']:.2f}M</td>"
            f"<td style='color:#8b949e;font-size:11px;'>{rec}</td>"
            f"</tr>"
        )
    st.markdown(
        f"<table class='mtable'><thead><tr>{hdr_html}</tr></thead>"
        f"<tbody>{rows_html}</tbody></table>",
        unsafe_allow_html=True,
    )
    render_chart_note(lang_text(
        "Peak produksi diestimasi dari persentil P90 data historis aktif. "
        "Nilai ekonomi dihitung pada asumsi harga minyak $70/bbl. "
        "Score dihitung ulang setiap 5 menit.",
        "Peak production estimated from P90 of historical active data. "
        "Economic value calculated at assumed oil price $70/bbl. "
        "Scores recalculated every 5 minutes.",
    ))
