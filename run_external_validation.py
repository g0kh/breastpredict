"""
External validation of METABRIC-trained DRFI models on hospital cohort.
Produces sonuclar/report_external.html with embedded figures + tables/.
"""
from __future__ import annotations

import base64
import json
import warnings
from datetime import datetime
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import brier_score, concordance_index_ipcw, cumulative_dynamic_auc
from sksurv.util import Surv

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "sonuclar"
EXT_OUT = OUT / "external"
TAB = EXT_OUT / "tables"
FIG = EXT_OUT / "figures"
for d in (OUT, EXT_OUT, TAB, FIG):
    d.mkdir(parents=True, exist_ok=True)

Y = 365.25
RANDOM_STATE = 42
N_BOOT = 400
S5 = ROOT / "brca_metabric" / "NIHMS1520488-supplement-Supp_Table_5.txt"
DB = ROOT / "database.xlsx"
EXT_XLSX = OUT / "External_Validasyon_Veri_Sablonu.xlsx"

FEAT_COLS = [
    "age",
    "log_size",
    "ln",
    "grade",
    "er",
    "pr",
    "her2",
    "post",
    "lobular",
    "ct",
    "ht",
    "pr_miss",
    "her2_miss",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fig_b64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def save_fig(fig, name: str) -> str:
    fig.savefig(FIG / name, dpi=160, bbox_inches="tight", facecolor="white")
    return fig_b64(fig)


def df_html(df: pd.DataFrame, digits: int = 3) -> str:
    if df is None or len(df) == 0:
        return "<p><em>Yok</em></p>"
    show = df.copy()
    for c in show.columns:
        if pd.api.types.is_float_dtype(show[c]):
            show[c] = show[c].map(lambda x: f"{x:.{digits}f}" if pd.notna(x) else "—")
    return show.to_html(index=False, classes="tbl", border=0, escape=True)


def yes_no(v):
    s = str(v).strip().lower()
    if s in ("evet", "yes", "1", "true"):
        return 1.0
    if s in ("hayir", "hayır", "no", "0", "false"):
        return 0.0
    return np.nan


def pos_neg(v):
    s = str(v).strip().lower()
    if s in ("pozitif", "positive", "pos"):
        return 1.0
    if s in ("negatif", "negative", "neg"):
        return 0.0
    return np.nan  # bilinmiyor / NA


def meno_post(v):
    s = str(v).strip().lower()
    if s == "post":
        return 1.0
    if s == "pre":
        return 0.0
    return np.nan


def imp_scale_fit_transform(Xtr: pd.DataFrame, Xte: pd.DataFrame):
    im = IterativeImputer(max_iter=15, random_state=RANDOM_STATE).fit(Xtr)
    a, b = im.transform(Xtr), im.transform(Xte)
    for j, c in enumerate(Xtr.columns):
        if c in ("pr", "her2", "post", "lobular", "ct", "ht", "pr_miss", "her2_miss", "er"):
            a[:, j] = np.clip(np.round(a[:, j]), 0, 1)
            b[:, j] = np.clip(np.round(b[:, j]), 0, 1)
        if c == "grade":
            a[:, j] = np.clip(np.round(a[:, j]), 1, 3)
            b[:, j] = np.clip(np.round(b[:, j]), 1, 3)
    sc = StandardScaler().fit(a)
    return sc.transform(a), sc.transform(b), im, sc


def risk_from_surv_fns(fns, t):
    return np.array([1.0 - float(f(t)) for f in fns])


def cap_test(ytr, yte):
    yte = yte.copy()
    cens = ytr["time"][ytr["event"] == 0]
    if len(cens) == 0:
        return yte
    cap = float(cens.max() - 1)
    over = yte["time"] > cap
    yte["event"][over] = False
    yte["time"][over] = cap
    return yte


def event_by_horizon(yS, t):
    ev = yS["event"].astype(bool)
    tm = yS["time"].astype(float)
    lab = np.full(len(yS), np.nan)
    lab[(tm <= t) & ev] = 1
    lab[tm > t] = 0
    return lab


def calibration_slope(y_bin, p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    if len(np.unique(y_bin)) < 2:
        return np.nan
    lp = np.log(p / (1 - p)).reshape(-1, 1)
    try:
        return float(LogisticRegression(C=1e6, max_iter=3000).fit(lp, y_bin).coef_[0, 0])
    except Exception:
        return np.nan


def dca_net_benefit(y_bin, p, thresholds):
    n = len(y_bin)
    out = []
    for pt in thresholds:
        pred = (p >= pt).astype(int)
        tp = ((pred == 1) & (y_bin == 1)).sum()
        fp = ((pred == 1) & (y_bin == 0)).sum()
        out.append(tp / n - fp / n * (pt / (1 - pt)))
    return np.array(out)


class ERStratCox:
    def fit(self, X, y, er):
        self.m = {}
        for g in (0, 1):
            idx = er == g
            if idx.sum() >= 30 and y[idx]["event"].sum() >= 5:
                self.m[g] = CoxPHSurvivalAnalysis(alpha=1.0).fit(X[idx], y[idx])
            else:
                self.m[g] = None
        self.fallback = CoxPHSurvivalAnalysis(alpha=1.0).fit(X, y)
        return self

    def risk_at(self, X, er, t):
        out = np.zeros(len(X))
        for g in (0, 1):
            idx = er == g
            if not idx.any():
                continue
            model = self.m.get(g) or self.fallback
            fns = model.predict_survival_function(X[idx])
            out[idx] = risk_from_surv_fns(fns, t)
        return out


# ---------------------------------------------------------------------------
# Load development + external
# ---------------------------------------------------------------------------

def load_development():
    s5 = pd.read_csv(S5, sep="\t")
    db = pd.read_excel(DB, sheet_name="hastalar")
    s = s5.copy()
    s = s[(s["T"] > 0) & s["Death"].notna() & (s["Stage"] != 4)]
    s = s[~s["Histological.Type"].isin(["BENIGN", "PHYL", "DCIS"])]
    s = s.merge(db[["hasta_id", "pr", "her2"]], left_on="METABRIC.ID", right_on="hasta_id", how="left")

    event = ((s["DR"] == 1) | (s["DeathBreast"] == 1)).astype(bool).values
    time = np.where(s["DR"] == 1, s["TDR"], s["T"]).astype(float)
    time = np.where(time <= 0, 1.0, time)

    h = s["Histological.Type"].fillna("")
    F = pd.DataFrame(
        {
            "age": s["Age.At.Diagnosis"],
            "log_size": np.log(pd.to_numeric(s["Size"], errors="coerce").clip(lower=1)),
            "ln": np.log1p(pd.to_numeric(s["Lymph.Nodes.Positive"], errors="coerce").fillna(0).clip(upper=20)),
            "grade": pd.to_numeric(s["Grade"], errors="coerce"),
            "er": (s["ER.Status"] == "pos").astype(float),
            "pr": s["pr"].map({"Positive": 1.0, "Negative": 0.0}),
            "her2": s["her2"].map({"Positive": 1.0, "Negative": 0.0}),
            "post": s["Inferred.Menopausal.State"].map({"post": 1.0, "pre": 0.0}),
            "lobular": h.str.contains("ILC").astype(float),
            "ct": (s["CT"] != "NO/NA").astype(float),
            "ht": (s["HT"] != "NO/NA").astype(float),
        }
    )
    F["pr_miss"] = F["pr"].isna().astype(float)
    F["her2_miss"] = F["her2"].isna().astype(float)
    F = F[FEAT_COLS]
    npi = pd.to_numeric(s["NPI"], errors="coerce").values
    yS = Surv.from_arrays(event, time)
    return F.reset_index(drop=True), yS, npi, s.reset_index(drop=True)


def load_external():
    raw = pd.read_excel(EXT_XLSX, sheet_name="2_Veri_Girisi", header=None)
    header = list(raw.iloc[1].values)
    df = raw.iloc[3:].copy()
    df.columns = header
    df = df.dropna(subset=["hasta_kodu"])
    df = df[df["hasta_kodu"].astype(str).str.match(r"^EXT-", na=False)].copy()
    df = df.reset_index(drop=True)

    # exclusions: stage 4, non-invasive if any
    evre = pd.to_numeric(df["evre"], errors="coerce")
    hist = df["histoloji"].astype(str)
    keep = (evre != 4) & (~hist.str.upper().isin(["DCIS", "BENIGN", "PHYL"]))
    dropped = int((~keep).sum())
    df = df.loc[keep].reset_index(drop=True)
    hist = df["histoloji"].astype(str)

    age = pd.to_numeric(df["yas_tani"], errors="coerce")
    size = pd.to_numeric(df["tumor_boyutu_mm"], errors="coerce")
    ln = pd.to_numeric(df["pozitif_lenf_nodu_sayisi"], errors="coerce").fillna(0).clip(upper=20)
    grade = pd.to_numeric(df["grade"], errors="coerce")
    er = df["ER"].map(pos_neg)
    pr = df["PR"].map(pos_neg)
    her2 = df["HER2"].map(pos_neg)
    post = df["menopoz"].map(meno_post)
    lobular = hist.str.upper().str.contains("ILC").astype(float)
    ct = df["kemoterapi_verildi"].map(yes_no)
    ht = df["hormonoterapi_verildi"].map(yes_no)

    F = pd.DataFrame(
        {
            "age": age,
            "log_size": np.log(size.clip(lower=1)),
            "ln": np.log1p(ln),
            "grade": grade,
            "er": er,
            "pr": pr,
            "her2": her2,
            "post": post,
            "lobular": lobular,
            "ct": ct,
            "ht": ht,
        }
    )
    F["pr_miss"] = F["pr"].isna().astype(float)
    F["her2_miss"] = F["her2"].isna().astype(float)
    F = F[FEAT_COLS]

    event = pd.to_numeric(df["DRFI_olay"], errors="coerce").fillna(0).astype(int).astype(bool).values
    time = pd.to_numeric(df["DRFI_sure_gun"], errors="coerce").astype(float).values
    time = np.where(~np.isfinite(time) | (time <= 0), 1.0, time)
    yS = Surv.from_arrays(event, time)

    npi = pd.to_numeric(df["NPI"], errors="coerce").values
    meta = df.copy()
    meta["evre_num"] = pd.to_numeric(df["evre"], errors="coerce")
    meta["time_y"] = time / Y
    meta["event"] = event.astype(int)

    # site labels
    site_raw = df["ilk_uzak_bolge"].astype(str).str.strip().str.lower()
    site_map = {
        "kemik": "bone_first",
        "viseral": "visceral_first",
        "visceral": "visceral_first",
        "beyin_sss": "cns_first",
        "beyin": "cns_first",
        "kemik+viseral": "bone_first",
        "diger": "other_first",
        "nan": np.nan,
    }
    meta["site_cause"] = site_raw.map(lambda x: site_map.get(x, "other_first" if x and x != "nan" else np.nan))
    met_flag = df["ilk_uzak_metastaz_oldu"].map(yes_no)
    meta["distant_met"] = met_flag
    meta["site_time"] = pd.to_numeric(df["ilk_uzak_metastaz_gun"], errors="coerce")

    return F, yS, npi, meta, int(dropped)


# ---------------------------------------------------------------------------
# Train on development, score external
# ---------------------------------------------------------------------------

def train_and_score(F_tr, y_tr, npi_tr, F_te, y_te, npi_te):
    Xtr, Xte, im, sc = imp_scale_fit_transform(F_tr, F_te)
    er_tr = np.nan_to_num(F_tr["er"].values, nan=0).astype(int)
    er_te = np.nan_to_num(F_te["er"].values, nan=0).astype(int)

    # NPI
    med = np.nanmedian(npi_tr)
    a = np.nan_to_num(npi_tr, nan=med).reshape(-1, 1)
    b = np.nan_to_num(npi_te, nan=med).reshape(-1, 1)
    npi_sc = StandardScaler().fit(a)
    npi_tr_s, npi_te_s = npi_sc.transform(a), npi_sc.transform(b)

    models = {}
    models["NPI_Cox"] = CoxPHSurvivalAnalysis(alpha=1.0).fit(npi_tr_s, y_tr)
    models["Cox"] = CoxPHSurvivalAnalysis(alpha=1.0).fit(Xtr, y_tr)
    models["Cox_ER_ayri"] = ERStratCox().fit(Xtr, y_tr, er_tr)
    models["RSF"] = RandomSurvivalForest(
        n_estimators=300,
        min_samples_leaf=15,
        max_features="sqrt",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    ).fit(Xtr, y_tr)

    y_te_c = cap_test(y_tr, y_te)
    tmax = min(float(y_tr["time"].max()), float(y_te_c["time"].max()))
    horizons = [1 * Y, 2 * Y, 3 * Y, 4 * Y, 5 * Y, 10 * Y]
    horizons = [t for t in horizons if t < tmax]

    risks = {}
    for name, model in models.items():
        risks[name] = {}
        for t in horizons:
            if name == "NPI_Cox":
                risks[name][t] = risk_from_surv_fns(model.predict_survival_function(npi_te_s), t)
            elif name == "Cox_ER_ayri":
                risks[name][t] = model.risk_at(Xte, er_te, t)
            else:
                risks[name][t] = risk_from_surv_fns(model.predict_survival_function(Xte), t)

    return models, risks, horizons, y_te_c, Xtr, Xte, npi_tr_s, npi_te_s


def eval_external(y_tr, y_te, risks, horizons):
    rows = []
    for name, by_t in risks.items():
        row = {"model": name}
        for t in horizons:
            p = by_t[t]
            yr = int(round(t / Y))
            try:
                row[f"AUC{yr}y"] = float(cumulative_dynamic_auc(y_tr, y_te, p, [t])[0][0])
            except Exception:
                row[f"AUC{yr}y"] = np.nan
            try:
                row[f"Brier{yr}y"] = float(brier_score(y_tr, y_te, 1 - p, [t])[1][0])
            except Exception:
                row[f"Brier{yr}y"] = np.nan
        # Uno C at longest horizon available among {5,10}
        for tau in [10 * Y, 5 * Y]:
            if tau in by_t:
                try:
                    row[f"UnoC_{int(round(tau/Y))}y"] = float(
                        concordance_index_ipcw(y_tr, y_te, by_t[tau], tau=tau)[0]
                    )
                except Exception:
                    row[f"UnoC_{int(round(tau/Y))}y"] = np.nan
                break
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_external_metrics(y_tr, y_te, risks, t, n_boot=N_BOOT):
    """Patient bootstrap on external test set for CI of tdAUC / Brier."""
    rng = np.random.RandomState(RANDOM_STATE)
    n = len(y_te)
    out = {}
    for name, by_t in risks.items():
        if t not in by_t:
            continue
        p0 = by_t[t]
        aucs, briers = [], []
        for _ in range(n_boot):
            idx = rng.choice(n, size=n, replace=True)
            yb = y_te[idx]
            pb = p0[idx]
            # need at least one event in bootstrap sample
            if yb["event"].sum() < 2:
                continue
            try:
                aucs.append(float(cumulative_dynamic_auc(y_tr, yb, pb, [t])[0][0]))
            except Exception:
                pass
            try:
                briers.append(float(brier_score(y_tr, yb, 1 - pb, [t])[1][0]))
            except Exception:
                pass
        def ci(a):
            a = np.asarray(a, dtype=float)
            a = a[np.isfinite(a)]
            if len(a) < 20:
                return np.nan, np.nan, np.nan
            return float(np.mean(a)), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))
        am, alo, ahi = ci(aucs)
        bm, blo, bhi = ci(briers)
        out[name] = {
            "AUC_mean": am,
            "AUC_lo": alo,
            "AUC_hi": ahi,
            "Brier_mean": bm,
            "Brier_lo": blo,
            "Brier_hi": bhi,
            "n_boot_auc": len(aucs),
            "n_boot_brier": len(briers),
        }
    return out


def platt_recalibrate(y_bin, p_train_like, p_apply):
    """Fit logistic on logit(p) using external observed labels; apply."""
    mask = np.isfinite(y_bin) & np.isfinite(p_train_like)
    if mask.sum() < 20 or len(np.unique(y_bin[mask])) < 2:
        return p_apply, np.nan, np.nan
    p = np.clip(p_train_like[mask], 1e-4, 1 - 1e-4)
    lp = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=3000).fit(lp, y_bin[mask].astype(int))
    pa = np.clip(p_apply, 1e-4, 1 - 1e-4)
    lpa = np.log(pa / (1 - pa)).reshape(-1, 1)
    return lr.predict_proba(lpa)[:, 1], float(lr.coef_[0, 0]), float(lr.intercept_[0])


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_cohort_compare(meta_ext, F_ext, F_dev, y_ext, y_dev):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.ravel()
    # age
    axes[0].hist(F_dev["age"].dropna(), bins=25, alpha=0.55, color="#1f5a4e", label="METABRIC", density=True)
    axes[0].hist(F_ext["age"].dropna(), bins=15, alpha=0.65, color="#8b3a3a", label="External", density=True)
    axes[0].set_title("Yaş")
    axes[0].legend(fontsize=8)
    # size
    axes[1].hist(np.exp(F_dev["log_size"]), bins=25, alpha=0.55, color="#1f5a4e", density=True)
    axes[1].hist(np.exp(F_ext["log_size"]), bins=15, alpha=0.65, color="#8b3a3a", density=True)
    axes[1].set_title("Tümör boyutu (mm)")
    # LN
    axes[2].hist(np.expm1(F_dev["ln"]), bins=20, alpha=0.55, color="#1f5a4e", density=True)
    axes[2].hist(np.expm1(F_ext["ln"]), bins=12, alpha=0.65, color="#8b3a3a", density=True)
    axes[2].set_title("Pozitif LN")
    # ER/PR/HER2
    def rate(series):
        return float(np.nanmean(series))

    cats = ["ER+", "PR+", "HER2+", "Postmeno", "CT", "HT"]
    dev_r = [
        rate(F_dev["er"]),
        rate(F_dev["pr"]),
        rate(F_dev["her2"]),
        rate(F_dev["post"]),
        rate(F_dev["ct"]),
        rate(F_dev["ht"]),
    ]
    ext_r = [
        rate(F_ext["er"]),
        rate(F_ext["pr"]),
        rate(F_ext["her2"]),
        rate(F_ext["post"]),
        rate(F_ext["ct"]),
        rate(F_ext["ht"]),
    ]
    x = np.arange(len(cats))
    axes[3].bar(x - 0.18, dev_r, 0.35, color="#1f5a4e", label="METABRIC")
    axes[3].bar(x + 0.18, ext_r, 0.35, color="#8b3a3a", label="External")
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(cats, rotation=30, ha="right")
    axes[3].set_ylim(0, 1)
    axes[3].set_title("Oranlar")
    axes[3].legend(fontsize=8)
    # follow-up
    axes[4].hist(y_dev["time"] / Y, bins=30, alpha=0.55, color="#1f5a4e", density=True)
    axes[4].hist(y_ext["time"] / Y, bins=15, alpha=0.65, color="#8b3a3a", density=True)
    axes[4].set_title("Takip / olay süresi (yıl)")
    # KM-like event cumulative crude
    for ax_i, (lab, ys, col) in enumerate(
        [
            ("METABRIC", y_dev, "#1f5a4e"),
            ("External", y_ext, "#8b3a3a"),
        ]
    ):
        pass
    tgrid = np.linspace(0, min(12, y_ext["time"].max() / Y), 50)
    for ys, col, lab in [(y_dev, "#1f5a4e", "METABRIC"), (y_ext, "#8b3a3a", "External")]:
        cum = []
        for t in tgrid:
            labv = event_by_horizon(ys, t * Y)
            m = np.isfinite(labv)
            cum.append(np.nan if m.sum() == 0 else labv[m].mean())
        axes[5].plot(tgrid, cum, color=col, lw=2, label=lab)
    axes[5].set_title("Kümülatif olay oranı (naive)")
    axes[5].set_xlabel("Yıl")
    axes[5].legend(fontsize=8)
    fig.suptitle("Geliştirme vs external kohort profili", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return save_fig(fig, "ext_cohort_compare.png")


def plot_auc_bars(metrics_df, boot_ci, title, fname, col="AUC5y"):
    d = metrics_df.dropna(subset=[col]).sort_values(col)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh(d["model"], d[col], color="#1f5a4e")
    for i, (_, r) in enumerate(d.iterrows()):
        m = r["model"]
        if m in boot_ci and np.isfinite(boot_ci[m].get("AUC_lo", np.nan)):
            ax.plot([boot_ci[m]["AUC_lo"], boot_ci[m]["AUC_hi"]], [i, i], color="#222", lw=2)
    ax.axvline(0.5, color="#aaa", ls="--", lw=1)
    ax.set_xlabel(col)
    ax.set_xlim(0.4, 0.95)
    ax.set_title(title)
    return save_fig(fig, fname)


def plot_tdauc_curves(metrics_df, horizons):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    years = sorted({int(round(t / Y)) for t in horizons})
    for _, r in metrics_df.iterrows():
        xs, ys = [], []
        for yr in years:
            c = f"AUC{yr}y"
            if c in r and pd.notna(r[c]):
                xs.append(yr)
                ys.append(r[c])
        if xs:
            ax.plot(xs, ys, "o-", lw=2, label=r["model"])
    ax.axhline(0.5, color="#aaa", ls="--")
    ax.set_xlabel("Ufuk (yıl)")
    ax.set_ylabel("Time-dependent AUC")
    ax.set_title("External tdAUC — zaman ufukları")
    ax.legend(fontsize=8)
    ax.set_ylim(0.4, 0.95)
    return save_fig(fig, "ext_tdauc_horizons.png")


def plot_calibration(yS, p, title, fname, t=5 * Y, n_bins=5):
    lab = event_by_horizon(yS, t)
    mask = np.isfinite(lab) & np.isfinite(p)
    if mask.sum() < 25:
        return "", np.nan
    pp = p[mask]
    yy = lab[mask].astype(int)
    try:
        bins = pd.qcut(pp, q=min(n_bins, max(3, mask.sum() // 8)), duplicates="drop")
    except Exception:
        return "", np.nan
    g = pd.DataFrame({"p": pp, "y": yy, "bin": bins}).groupby("bin", observed=True).agg(
        pred=("p", "mean"), obs=("y", "mean"), n=("y", "size")
    )
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 1], [0, 1], "--", color="#aaa")
    ax.plot(g["pred"], g["obs"], "o-", color="#8b3a3a", ms=8)
    for _, r in g.iterrows():
        ax.text(r["pred"], r["obs"], str(int(r["n"])), fontsize=8)
    slope = calibration_slope(yy, pp)
    ax.text(0.05, 0.92, f"kalibrasyon eğimi ≈ {slope:.2f}", transform=ax.transAxes, fontsize=10)
    ax.set_xlabel(f"Tahmin edilen {int(t/Y)}y DRFI riski")
    ax.set_ylabel("Gözlenen olay oranı")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return save_fig(fig, fname), slope


def plot_dca(yS, risk_dict, fname, t=5 * Y):
    lab = event_by_horizon(yS, t)
    thresholds = np.arange(0.05, 0.55, 0.01)
    fig, ax = plt.subplots(figsize=(7.2, 5))
    mask0 = np.isfinite(lab)
    y = lab[mask0].astype(int)
    prev = y.mean() if len(y) else 0
    nb_all = prev - (1 - prev) * (thresholds / (1 - thresholds))
    ax.plot(thresholds, nb_all, "--", color="#888", label="Hepsini tedavi et")
    ax.plot(thresholds, np.zeros_like(thresholds), ":", color="#888", label="Hiçbirini tedavi etme")
    for name, p in risk_dict.items():
        mask = mask0 & np.isfinite(p)
        if mask.sum() < 20:
            continue
        nb = dca_net_benefit(lab[mask].astype(int), p[mask], thresholds)
        ax.plot(thresholds, nb, lw=2, label=name)
    ax.set_xlabel(f"Eşik olasılık ({int(t/Y)}y DRFI)")
    ax.set_ylabel("Net fayda")
    ax.set_title("Karar eğrisi analizi (external)")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, max(0.25, prev * 1.2))
    return save_fig(fig, fname)


def plot_risk_by_outcome(meta, risks5, fname):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    models = list(risks5.keys())
    data = []
    for m in models:
        for i, ev in enumerate(meta["event"].values):
            data.append({"model": m, "risk": risks5[m][i], "DRFI": "Olay" if ev == 1 else "Sansür"})
    d = pd.DataFrame(data)
    sns.boxplot(data=d, x="model", y="risk", hue="DRFI", ax=ax, palette=["#1f5a4e", "#8b3a3a"])
    ax.set_ylabel("Tahmini 5y DRFI riski")
    ax.set_title("External: tahmin riski — olay vs sansür")
    ax.legend(title="")
    plt.xticks(rotation=15)
    return save_fig(fig, fname)


def plot_site_pie(meta):
    sub = meta[meta["distant_met"] == 1].copy()
    if len(sub) == 0:
        return ""
    counts = sub["site_cause"].fillna("unknown").value_counts()
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.pie(counts.values, labels=counts.index, autopct="%1.0f%%", startangle=90)
    ax.set_title(f"İlk uzak bölge (n={len(sub)} metastaz)")
    return save_fig(fig, "ext_site_pie.png")


def plot_forest_internal_vs_external(internal_auc5, ext_metrics, boot5):
    """Compare internal CV AUC5y vs external."""
    rows = []
    for _, r in ext_metrics.iterrows():
        m = r["model"]
        rows.append(
            {
                "model": m,
                "internal": internal_auc5.get(m, np.nan),
                "external": r.get("AUC5y", np.nan),
                "ext_lo": boot5.get(m, {}).get("AUC_lo", np.nan),
                "ext_hi": boot5.get(m, {}).get("AUC_hi", np.nan),
            }
        )
    d = pd.DataFrame(rows).dropna(subset=["external"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = np.arange(len(d))
    ax.scatter(d["internal"], y + 0.12, s=70, color="#1f5a4e", label="Internal CV 5y tdAUC", zorder=3)
    ax.scatter(d["external"], y - 0.12, s=70, color="#8b3a3a", label="External 5y tdAUC", zorder=3)
    for i, r in d.iterrows():
        ii = list(d.index).index(i)
        if np.isfinite(r["ext_lo"]):
            ax.plot([r["ext_lo"], r["ext_hi"]], [ii - 0.12, ii - 0.12], color="#8b3a3a", lw=2)
    ax.set_yticks(y)
    ax.set_yticklabels(d["model"])
    ax.axvline(0.5, color="#aaa", ls="--")
    ax.set_xlabel("tdAUC (5 yıl)")
    ax.set_title("Internal vs external ayırt edicilik")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0.45, 0.9)
    return save_fig(fig, "ext_vs_internal_auc.png")


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def build_html(ctx: dict) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""<!DOCTYPE html>
<html lang="tr"><head><meta charset="utf-8"/>
<title>BreastPredict — External Validasyon Raporu</title>
<style>
:root {{ --ink:#1a1814; --muted:#5a554c; --card:#fffcf7; --line:#e2dbd0; --bg:#f4f1ec; --brand:#7a3030; --teal:#1f5a4e; }}
body{{font-family:Georgia,"Times New Roman",serif;background:var(--bg);color:var(--ink);margin:0;line-height:1.55}}
.wrap{{max-width:1080px;margin:auto;padding:32px 18px 80px}}
.brand{{color:var(--brand);font-size:11px;letter-spacing:.14em;text-transform:uppercase;font-weight:700;font-family:Segoe UI,Arial,sans-serif}}
h1{{font-size:34px;margin:6px 0 8px;line-height:1.2}}
h2{{font-size:22px;margin:0 0 10px;color:var(--teal);font-family:Segoe UI,Arial,sans-serif}}
h3{{font-size:16px;margin:14px 0 6px;font-family:Segoe UI,Arial,sans-serif}}
p,li{{font-size:15px;color:var(--ink)}}
.lead{{font-size:17px;color:var(--muted);max-width:52rem}}
section{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px 20px 16px;margin:16px 0}}
.tbl{{width:100%;border-collapse:collapse;font-size:13px;font-family:Segoe UI,Arial,sans-serif}}
.tbl th{{background:#f1ebe1;text-align:left;padding:8px;border-bottom:1px solid var(--line)}}
.tbl td{{padding:7px 8px;border-bottom:1px solid var(--line)}}
.plot{{width:100%;max-width:560px;border:1px solid var(--line);border-radius:10px;background:#fff}}
.plots{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px;margin-top:10px}}
.note{{background:#f7efe4;border-left:4px solid var(--brand);padding:12px 14px;color:var(--muted);font-size:14px;margin:10px 0}}
.ok{{background:#eaf2f0;border-left:4px solid var(--teal);padding:12px 14px;color:var(--muted);font-size:14px;margin:10px 0}}
.chip{{display:inline-block;background:#efe8dc;border-radius:999px;padding:4px 11px;margin:2px;font-size:12px;font-family:Segoe UI,Arial,sans-serif}}
.kpi{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:12px 0}}
.kpi .box{{background:#fff;border:1px solid var(--line);border-radius:12px;padding:12px}}
.kpi .n{{font-size:26px;font-weight:700;color:var(--teal);font-family:Segoe UI,Arial,sans-serif}}
.kpi .l{{font-size:12px;color:var(--muted);font-family:Segoe UI,Arial,sans-serif}}
ul{{padding-left:1.2rem}}
code{{background:#efe8dc;padding:1px 5px;border-radius:4px;font-size:13px}}
</style></head><body><div class="wrap">
<div class="brand">BreastPredict · External Validasyon</div>
<h1>Hastane kohortunda DRFI model testi</h1>
<p class="lead">METABRIC üzerinde eğitilen STEEP DRFI sağkalım modelleri (NPI-Cox, Cox, Cox_ER_ayrı, RSF),
doldurulmuş external Excel kohortunda skorlandı. Aşağıda kohort profili, ayırt edicilik, kalibrasyon,
karar eğrisi, rekalibrasyon ve bölge özeti yer alıyor.</p>
<div>
<span class="chip">{now}</span>
<span class="chip">Geliştirme n={ctx['n_dev']} / olay={ctx['n_dev_events']}</span>
<span class="chip">External n={ctx['n_ext']} / olay={ctx['n_ext_events']}</span>
<span class="chip">Ufuklar: 1–5y (+10y mümkünse)</span>
</div>

<section>
<h2>1. Yönetici özeti</h2>
<div class="kpi">
  <div class="box"><div class="n">{ctx['best_auc5']:.2f}</div><div class="l">En iyi external 5y tdAUC<br>({ctx['best_model']})</div></div>
  <div class="box"><div class="n">{ctx['best_unoc']:.2f}</div><div class="l">Uno C (external)</div></div>
  <div class="box"><div class="n">{ctx['n_ext_events']}</div><div class="l">DRFI olay / {ctx['n_ext']} hasta</div></div>
  <div class="box"><div class="n">{ctx['calib_slope']:.2f}</div><div class="l">Kalibrasyon eğimi ({ctx['best_model']}, 5y)</div></div>
</div>
<div class="note">{ctx['executive_note']}</div>
<p><b>Önerilen kullanım:</b> {ctx['recommendation']}</p>
</section>

<section>
<h2>2. External kohort nasıl kuruldu?</h2>
<p>Kaynak dosya: <code>sonuclar/External_Validasyon_Veri_Sablonu.xlsx</code> → sayfa <code>2_Veri_Girisi</code>.</p>
<ul>
<li>n = <b>{ctx['n_ext']}</b> hasta (evre 4 / DCIS / BENIGN / PHYL dışlandıktan sonra; düşülen={ctx['n_dropped']})</li>
<li>Tanı yılları: {ctx['year_range']}</li>
<li>DRFI olay tanımı Excel’deki STEEP kuralı ile aynı: uzak metastaz veya meme kanserine bağlı ölüm</li>
<li>Medyan takip/olay süresi: <b>{ctx['median_fu_y']:.1f} yıl</b></li>
<li>Özellikler geliştirme modeli ile hizalandı: yaş, log(boyut), log1p(LN), grade, ER/PR/HER2, menopoz, lobular, CT/HT (+ PR/HER2 missing göstergesi)</li>
</ul>
{df_html(ctx['desc_tbl'], 2)}
<div class="plots">
<img class="plot" src="data:image/png;base64,{ctx['img_cohort']}"/>
</div>
</section>

<section>
<h2>3. Eğitim ve skorlama protokolü</h2>
<ul>
<li>Modeller <b>yalnız METABRIC</b> geliştirme kohortunda (n={ctx['n_dev']}) eğitildi; external hiç fit’e girmedi.</li>
<li>Impute + scale geliştirme setinde fit edilip external’a uygulandı (bilgi sızıntısı yok).</li>
<li>Karşılaştırılan modeller: NPI_Cox (baseline), Cox, Cox_ER_ayrı, RSF.</li>
<li>Birincil metrikler: time-dependent AUC, IPCW Brier, Uno C; ek: kalibrasyon, DCA.</li>
<li>External örneklem küçük olduğu için 5y tdAUC için bootstrap CI ({N_BOOT} tekrar) verildi.</li>
</ul>
<div class="ok">Pre-trastuzumab METABRIC → modern trastuzumab dönemi external’da HER2+ riskinin abartılması beklenir; bu yüzden ham skor + rekalibre skor birlikte raporlandı.</div>
</section>

<section>
<h2>4. Ayırt edicilik (discrimination)</h2>
<h3>4.1 Özet tablo (external)</h3>
{df_html(ctx['metrics'])}
<div class="plots">
<img class="plot" src="data:image/png;base64,{ctx['img_auc5']}"/>
<img class="plot" src="data:image/png;base64,{ctx['img_tdauc']}"/>
</div>
<h3>4.2 Internal CV vs external</h3>
<p>Internal sayılar önceki DRFI pipeline 5-fold özetinden; external bu koşudan.</p>
{df_html(ctx['compare_tbl'])}
<div class="plots">
<img class="plot" src="data:image/png;base64,{ctx['img_vs']}"/>
</div>
<div class="note">{ctx['disc_note']}</div>
</section>

<section>
<h2>5. Kalibrasyon ve rekalibrasyon</h2>
<p>İyi ayırt edicilik, olasılıkların doğru ölçekte olduğu anlamına gelmez. Aşağıda 5 yıllık DRFI riski için
gözlenen vs beklenen oran ve logistic (Platt) rekalibrasyon sonrası eğri yer alıyor.</p>
<div class="plots">
<img class="plot" src="data:image/png;base64,{ctx['img_cal_raw']}"/>
<img class="plot" src="data:image/png;base64,{ctx['img_cal_recal']}"/>
</div>
<p>Ham kalibrasyon eğimi ({ctx['best_model']}): <b>{ctx['calib_slope']:.2f}</b>
(1.0 ideal; &gt;1 fazla yayılım / &lt;1 aşırı güven veya ölçek kayması).</p>
<p>Rekalibrasyon sonrası eğim: <b>{ctx['calib_slope_recal']:.2f}</b>
(Platt a={ctx['platt_a']:.2f}, b={ctx['platt_b']:.2f}).</p>
<div class="note">Rekalibrasyon <b>aynı external örneklemde</b> fit edildi (gösterim amaçlı). Gerçek uygulamada ayrı bir kalibrasyon seti veya merkez-içi güncelleme gerekir; aksi halde iyimser kalır.</div>
</section>

<section>
<h2>6. Karar eğrisi (DCA) ve risk dağılımı</h2>
<div class="plots">
<img class="plot" src="data:image/png;base64,{ctx['img_dca']}"/>
<img class="plot" src="data:image/png;base64,{ctx['img_box']}"/>
</div>
<p>DCA, belirli eşiklerde modelin “hepsini tedavi et / hiçbirini etme” stratejilerine göre net fayda verip vermediğini gösterir.
Küçük external n nedeniyle eğriler gürültülüdür; yönelimi yorumlayın, kesin eşik seçmeyin.</p>
</section>

<section>
<h2>7. Senaryo C — ilk uzak metastaz bölgesi</h2>
<p>External’da uzak metastaz işaretli hasta: <b>{ctx['n_site']}</b>. Dağılım:</p>
{df_html(ctx['site_tbl'], 0)}
<div class="plots">
<img class="plot" src="data:image/png;base64,{ctx['img_site']}"/>
</div>
<div class="note">n küçük olduğu için bölge için ayrı competing-risk AUC hesaplanmadı / güvenilir değil.
Bölge modeli için en az onlarca olay/sınıf gerekir; burada yalnızca tanımlayıcı özet verildi.</div>
</section>

<section>
<h2>8. Hasta düzeyinde skorlar</h2>
<p>İlk 15 satır (tam tablo: <code>sonuclar/external/tables/patient_scores.xlsx</code>):</p>
{df_html(ctx['scores_head'], 3)}
</section>

<section>
<h2>9. Sınırlılıklar ve sonuç</h2>
<ul>
<li>External n={ctx['n_ext']}, olay={ctx['n_ext_events']} — metrik CI’ları geniştir; tek merkezli keşifsel validasyon.</li>
<li>METABRIC pre-trastuzumab; external tanı yılları {ctx['year_range']} (trastuzumab dönemi) → HER2 etkisi kayabilir.</li>
<li>CTS5 yapılamadı (Ki-67 yok).</li>
<li>Binary accuracy / F1 raporlanmadı; survival metrikleri kullanıldı.</li>
<li>Rekalibrasyon aynı sette gösterildi; prospektif kullanımda ayrı kalibrasyon şart.</li>
</ul>
<div class="ok"><b>Sonuç:</b> {ctx['conclusion']}</div>
<p style="color:#5a554c;font-size:13px;margin-top:18px">Üretim: <code>run_external_validation.py</code> · seed {RANDOM_STATE} · {now}</p>
</section>
</div></body></html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading development cohort...")
    F_dev, y_dev, npi_dev, s_dev = load_development()
    print(f"  n={len(F_dev)} events={int(y_dev['event'].sum())}")

    print("Loading external cohort...")
    F_ext, y_ext, npi_ext, meta, n_dropped = load_external()
    print(f"  n={len(F_ext)} events={int(y_ext['event'].sum())} dropped={n_dropped}")

    print("Training on METABRIC and scoring external...")
    models, risks, horizons, y_ext_c, Xtr, Xte, npi_tr_s, npi_te_s = train_and_score(
        F_dev, y_dev, npi_dev, F_ext, y_ext, npi_ext
    )
    print("  horizons (years):", [round(t / Y, 1) for t in horizons])

    metrics = eval_external(y_dev, y_ext_c, risks, horizons)
    metrics.to_excel(TAB / "external_metrics.xlsx", index=False)
    print(metrics)

    t5 = 5 * Y
    boot5 = bootstrap_external_metrics(y_dev, y_ext_c, risks, t5) if t5 in horizons else {}
    boot_rows = []
    for m, d in boot5.items():
        boot_rows.append({"model": m, "horizon": "5y", **d})
    pd.DataFrame(boot_rows).to_excel(TAB / "external_bootstrap_5y.xlsx", index=False)

    # patient scores
    score_cols = {"hasta_kodu": meta["hasta_kodu"], "DRFI_olay": meta["event"], "DRFI_sure_gun": meta["time_y"] * Y}
    for name in risks:
        if t5 in risks[name]:
            score_cols[f"{name}_risk5y"] = risks[name][t5]
        if 10 * Y in risks[name]:
            score_cols[f"{name}_risk10y"] = risks[name][10 * Y]
    scores = pd.DataFrame(score_cols)
    scores.to_excel(TAB / "patient_scores.xlsx", index=False)

    # pick best by AUC5y
    if "AUC5y" in metrics.columns and metrics["AUC5y"].notna().any():
        best_row = metrics.sort_values("AUC5y", ascending=False).iloc[0]
        best_model = best_row["model"]
        best_auc5 = float(best_row["AUC5y"])
    else:
        best_model = "Cox"
        best_auc5 = float("nan")

    unoc_col = "UnoC_10y" if "UnoC_10y" in metrics.columns else "UnoC_5y"
    if unoc_col not in metrics.columns:
        # find any UnoC
        unoc_cols = [c for c in metrics.columns if c.startswith("UnoC_")]
        unoc_col = unoc_cols[0] if unoc_cols else None
    best_unoc = float(metrics.loc[metrics["model"] == best_model, unoc_col].iloc[0]) if unoc_col else float("nan")

    # calibration raw + recal for best
    p_best = risks[best_model][t5] if t5 in risks[best_model] else risks[best_model][horizons[-1]]
    t_cal = t5 if t5 in risks[best_model] else horizons[-1]
    img_cal_raw, slope_raw = plot_calibration(
        y_ext, p_best, f"Kalibrasyon (ham) — {best_model}", "ext_calib_raw.png", t=t_cal
    )
    lab5 = event_by_horizon(y_ext, t_cal)
    p_recal, platt_a, platt_b = platt_recalibrate(lab5, p_best, p_best)
    img_cal_recal, slope_recal = plot_calibration(
        y_ext, p_recal, f"Kalibrasyon (rekalibre) — {best_model}", "ext_calib_recal.png", t=t_cal
    )

    # plots
    img_cohort = plot_cohort_compare(meta, F_ext, F_dev, y_ext, y_dev)
    img_auc5 = plot_auc_bars(metrics, boot5, "External 5y tdAUC (+ bootstrap CI)", "ext_auc5y.png", col="AUC5y")
    img_tdauc = plot_tdauc_curves(metrics, horizons)
    risk5 = {m: risks[m][t5] for m in risks if t5 in risks[m]}
    img_dca = plot_dca(y_ext, risk5, "ext_dca.png", t=t5 if t5 in horizons else horizons[-1])
    img_box = plot_risk_by_outcome(meta, risk5, "ext_risk_box.png")
    img_site = plot_site_pie(meta)

    # internal reference from previous summary.json if present
    internal_auc5 = {"NPI_Cox": 0.690, "Cox": 0.714, "Cox_ER_ayri": 0.729, "RSF": 0.732}
    sum_path = OUT / "summary.json"
    if sum_path.exists():
        try:
            js = json.loads(sum_path.read_text(encoding="utf-8"))
            for rec in js.get("metrics_main", []):
                internal_auc5[rec["model"]] = rec.get("AUC5y", internal_auc5.get(rec["model"]))
        except Exception:
            pass

    img_vs = plot_forest_internal_vs_external(internal_auc5, metrics, boot5)

    compare_rows = []
    for _, r in metrics.iterrows():
        m = r["model"]
        compare_rows.append(
            {
                "model": m,
                "internal_AUC5y": internal_auc5.get(m, np.nan),
                "external_AUC5y": r.get("AUC5y", np.nan),
                "external_AUC5y_boot_lo": boot5.get(m, {}).get("AUC_lo", np.nan),
                "external_AUC5y_boot_hi": boot5.get(m, {}).get("AUC_hi", np.nan),
                "delta_ext_minus_int": (r.get("AUC5y", np.nan) - internal_auc5.get(m, np.nan))
                if pd.notna(r.get("AUC5y", np.nan))
                else np.nan,
            }
        )
    compare_tbl = pd.DataFrame(compare_rows)
    compare_tbl.to_excel(TAB / "internal_vs_external.xlsx", index=False)

    # descriptives
    years = pd.to_datetime(meta["tani_tarihi"], errors="coerce").dt.year
    year_range = f"{int(years.min())}–{int(years.max())}" if years.notna().any() else "?"
    desc_tbl = pd.DataFrame(
        [
            {"özellik": "n", "değer": len(F_ext)},
            {"özellik": "DRFI olay", "değer": int(y_ext["event"].sum())},
            {"özellik": "olay oranı", "değer": float(y_ext["event"].mean())},
            {"özellik": "medyan süre (yıl)", "değer": float(np.median(y_ext["time"] / Y))},
            {"özellik": "ER+", "değer": float(np.nanmean(F_ext["er"]))},
            {"özellik": "PR+", "değer": float(np.nanmean(F_ext["pr"]))},
            {"özellik": "HER2+", "değer": float(np.nanmean(F_ext["her2"]))},
            {"özellik": "postmenopoz", "değer": float(np.nanmean(F_ext["post"]))},
            {"özellik": "KT oranı", "değer": float(np.nanmean(F_ext["ct"]))},
            {"özellik": "HT oranı", "değer": float(np.nanmean(F_ext["ht"]))},
            {"özellik": "medyan yaş", "değer": float(np.nanmedian(F_ext["age"]))},
            {"özellik": "medyan boyut mm", "değer": float(np.nanmedian(np.exp(F_ext["log_size"])))},
        ]
    )
    desc_tbl.to_excel(TAB / "external_descriptives.xlsx", index=False)

    site_counts = meta.loc[meta["distant_met"] == 1, "site_cause"].fillna("unknown").value_counts()
    site_tbl = site_counts.rename_axis("site_cause").reset_index(name="n")
    site_tbl.to_excel(TAB / "external_site_counts.xlsx", index=False)

    # narrative
    delta = best_auc5 - internal_auc5.get(best_model, best_auc5)
    if np.isfinite(best_auc5) and best_auc5 >= 0.70:
        exec_note = (
            f"External kohortta {best_model} 5y tdAUC = {best_auc5:.2f}. "
            "Ayırt edicilik klinik olarak kullanılabilir aralıkta görünüyor; ancak olay sayısı sınırlı "
            f"({int(y_ext['event'].sum())}) ve CI geniştir. Kalibrasyon mutlaka kontrol edilmeli."
        )
    elif np.isfinite(best_auc5) and best_auc5 >= 0.60:
        exec_note = (
            f"External 5y tdAUC ({best_model}) = {best_auc5:.2f}: zayıf–orta ayırt edicilik. "
            "Model sinyali var ama tek başına klinik karar için yetersiz; rekalibrasyon ve daha büyük kohort gerekir."
        )
    else:
        exec_note = (
            f"External ayırt edicilik düşük veya hesaplanamadı (en iyi 5y tdAUC ≈ {best_auc5}). "
            "Küçük örneklem / dönem kayması / etiket gürültüsü olabilir; sonuçlar keşifseldir."
        )

    if np.isfinite(slope_raw) and abs(slope_raw - 1) > 0.35:
        recommendation = (
            f"{best_model} skorlarını merkezde Platt/intercept rekalibrasyonu ile güncelleyin; "
            "ham olasılıkları olduğu gibi eşiklemeyin. NPI_Cox baseline ile birlikte raporlayın."
        )
    else:
        recommendation = (
            f"Öncelikli aday: {best_model} (+ NPI_Cox baseline). "
            "Klinik kullanımdan önce ayrı bir kalibrasyon seti veya zamansal validasyon planlayın."
        )

    if np.isfinite(best_auc5):
        conclusion = (
            f"METABRIC’te eğitilen modeller external n={len(F_ext)} kohortta test edildi. "
            f"En iyi 5y tdAUC {best_model} ile {best_auc5:.2f} "
            f"(internal CV’ye göre Δ={delta:+.2f}). "
            "Bu, tek merkezli ön validasyondur; yayın/ klinik uygulama için daha fazla olay ve "
            "prospektif veya çok merkezli tekrar gerekir."
        )
    else:
        conclusion = "External metrikler güvenilir hesaplanamadı; kohort boyutu / olay sayısı yetersiz olabilir."

    disc_note = (
        "tdAUC, belirli bir zamanda hâlâ risk altında olanlar arasında olay/olmayan ayırt ediciliğidir. "
        "Accuracy değildir. Bootstrap CI’lar external hastalardan örneklenerek üretildi."
    )

    ctx = {
        "n_dev": len(F_dev),
        "n_dev_events": int(y_dev["event"].sum()),
        "n_ext": len(F_ext),
        "n_ext_events": int(y_ext["event"].sum()),
        "n_dropped": n_dropped,
        "n_site": int((meta["distant_met"] == 1).sum()),
        "median_fu_y": float(np.median(y_ext["time"] / Y)),
        "year_range": year_range,
        "best_model": best_model,
        "best_auc5": best_auc5 if np.isfinite(best_auc5) else float("nan"),
        "best_unoc": best_unoc if np.isfinite(best_unoc) else float("nan"),
        "calib_slope": slope_raw if np.isfinite(slope_raw) else float("nan"),
        "calib_slope_recal": slope_recal if np.isfinite(slope_recal) else float("nan"),
        "platt_a": platt_a if np.isfinite(platt_a) else float("nan"),
        "platt_b": platt_b if np.isfinite(platt_b) else float("nan"),
        "executive_note": exec_note,
        "recommendation": recommendation,
        "conclusion": conclusion,
        "disc_note": disc_note,
        "desc_tbl": desc_tbl,
        "metrics": metrics,
        "compare_tbl": compare_tbl,
        "site_tbl": site_tbl,
        "scores_head": scores.head(15),
        "img_cohort": img_cohort,
        "img_auc5": img_auc5,
        "img_tdauc": img_tdauc,
        "img_vs": img_vs,
        "img_cal_raw": img_cal_raw,
        "img_cal_recal": img_cal_recal,
        "img_dca": img_dca,
        "img_box": img_box,
        "img_site": img_site,
    }

    html = build_html(ctx)
    report_path = OUT / "report_external.html"
    report_path.write_text(html, encoding="utf-8")

    summary = {
        "framework": "external_validation_STEEP_DRFI",
        "n_development": len(F_dev),
        "n_external": len(F_ext),
        "n_external_events": int(y_ext["event"].sum()),
        "best_model_AUC5y": best_model,
        "external_metrics": metrics.to_dict(orient="records"),
        "bootstrap_5y": boot5,
        "calibration_slope_raw": slope_raw,
        "calibration_slope_recal": slope_recal,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "report": str(report_path),
    }
    (EXT_OUT / "summary_external.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print("\nDone.")
    print("Report:", report_path)
    print(json.dumps({"best": best_model, "AUC5y": best_auc5, "UnoC": best_unoc, "calib_slope": slope_raw}, indent=2))


if __name__ == "__main__":
    main()
