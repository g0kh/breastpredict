"""
DRFI survival + competing-risk site pipeline (TRIPOD-style internal validation).
Outputs → sonuclar/
Based on demo_duzeltmeler.py methodological fixes.
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
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import brier_score, concordance_index_ipcw, cumulative_dynamic_auc
from sksurv.util import Surv

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "sonuclar"
TAB = OUT / "tables"
FIG = OUT / "figures"
for d in (OUT, TAB, FIG):
    d.mkdir(parents=True, exist_ok=True)

Y = 365.25
RANDOM_STATE = 42
N_SPLITS = 5
N_BOOT = 120  # CI bootstrap on OOF predictions (runtime-friendly)
S5 = ROOT / "brca_metabric" / "NIHMS1520488-supplement-Supp_Table_5.txt"
S7 = ROOT / "brca_metabric" / "NIHMS1520488-supplement-Supp_Table_7.txt"
DB = ROOT / "database.xlsx"
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


def fig_b64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def save_fig(fig, name: str) -> str:
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight", facecolor="white")
    return fig_b64(fig)


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------

def load_raw():
    s5 = pd.read_csv(S5, sep="\t")
    db = pd.read_excel(DB, sheet_name="hastalar")
    return s5, db


def cohort_flow(s5: pd.DataFrame) -> pd.DataFrame:
    rows = [{"step": "raw_Table5", "n": len(s5)}]
    m = s5["T"] > 0
    rows.append({"step": "T>0", "n": int(m.sum()), "dropped": int((~m).sum())})
    m2 = m & s5["Death"].notna()
    rows.append({"step": "Death_notna", "n": int(m2.sum()), "dropped": int((m & ~s5["Death"].notna()).sum())})
    m3 = m2 & (s5["Stage"] != 4)
    rows.append({"step": "exclude_Stage4", "n": int(m3.sum()), "dropped": int((m2 & (s5["Stage"] == 4)).sum())})
    bad_hist = s5["Histological.Type"].isin(["BENIGN", "PHYL", "DCIS"])
    m4 = m3 & ~bad_hist
    rows.append({"step": "exclude_BENIGN_PHYL_DCIS", "n": int(m4.sum()), "dropped": int((m3 & bad_hist).sum())})
    return pd.DataFrame(rows)


def build_analysis_frame(s5: pd.DataFrame, db: pd.DataFrame, require_complete_rec: bool = False):
    s = s5.copy()
    s = s[(s["T"] > 0) & s["Death"].notna() & (s["Stage"] != 4)]
    s = s[~s["Histological.Type"].isin(["BENIGN", "PHYL", "DCIS"])]
    if require_complete_rec:
        s = s[s["Complete.Rec.History"] == "YES"]
    s = s.merge(db[["hasta_id", "pr", "her2"]], left_on="METABRIC.ID", right_on="hasta_id", how="left")

    # STEEP DRFI
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

    meta = pd.DataFrame(
        {
            "id": s["METABRIC.ID"].values,
            "cohort": s["Cohort"].values,
            "npi": pd.to_numeric(s["NPI"], errors="coerce").values,
            "complete_rec": s["Complete.Rec.History"].values,
            "dr": s["DR"].values,
            "death_breast": s["DeathBreast"].values,
            "death": s["Death"].values,
            "tdr": s["TDR"].values,
            "t": s["T"].values,
        }
    )
    yS = Surv.from_arrays(event, time)
    return s.reset_index(drop=True), F.reset_index(drop=True), yS, meta.reset_index(drop=True)


def first_site_labels(s_clean: pd.DataFrame) -> pd.DataFrame:
    """First distant site from Table 7; competing causes bone/visceral/cns."""
    t7 = pd.read_csv(S7, sep="\t")
    dist = t7[t7["TYPE.RELAPSE"].astype(str).str.upper() == "DISTANT"].copy()
    dist["TIME.RELAPSE"] = pd.to_numeric(dist["TIME.RELAPSE"], errors="coerce")
    dist = dist.dropna(subset=["TIME.RELAPSE"])
    # earliest distant row per patient
    dist = dist.sort_values("TIME.RELAPSE").groupby("METABRIC.ID", as_index=False).first()

    def classify(site):
        u = str(site).upper()
        has_cns = any(x in u for x in ["BRAIN", "CNS", "MENING"])
        has_bone = "BONE" in u
        has_visc = any(
            x in u
            for x in ["LIVER", "LUNG", "PLEURA", "PERITONE", "ASCITES", "ADRENAL", "MEDIAST"]
        )
        # priority if multi: cns > bone > visceral > other→visceral bucket as non-cns non-bone
        if has_cns:
            return "cns_first"
        if has_bone and not has_visc:
            return "bone_first"
        if has_visc and not has_bone:
            return "visceral_first"
        if has_bone and has_visc:
            return "bone_first"  # bone-inclusive multi counted bone-first clinically common
        return "other_first"

    dist["site_cause"] = dist["SITE"].map(classify)
    m = s_clean[["METABRIC.ID", "DR", "TDR", "T"]].merge(
        dist[["METABRIC.ID", "SITE", "TIME.RELAPSE", "site_cause"]],
        on="METABRIC.ID",
        how="left",
    )
    # competing event time: first distant site time if DR else censor at T
    cause = []
    etime = []
    for _, r in m.iterrows():
        if r["DR"] == 1 and pd.notna(r["site_cause"]) and r["site_cause"] != "other_first":
            cause.append(r["site_cause"])
            etime.append(float(r["TIME.RELAPSE"] if pd.notna(r["TIME.RELAPSE"]) else r["TDR"]))
        elif r["DR"] == 1:
            cause.append("other_first")
            etime.append(float(r["TDR"]))
        else:
            cause.append("censor")
            etime.append(float(r["T"]))
    m["cause"] = cause
    m["etime"] = np.maximum(etime, 1.0)
    return m


# ---------------------------------------------------------------------------
# Preprocess / models
# ---------------------------------------------------------------------------

def imp_scale(Xtr: pd.DataFrame, Xte: pd.DataFrame):
    im = IterativeImputer(max_iter=10, random_state=RANDOM_STATE).fit(Xtr)
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
            out[idx] = [1 - f(t) for f in fns]
        return out


def risk_from_surv_fns(fns, t):
    return np.array([1 - f(t) for f in fns])


def cap_test(ytr, yte):
    yte = yte.copy()
    cens = ytr["time"][ytr["event"] == 0]
    if len(cens) == 0:
        return yte
    cap = cens.max() - 1
    over = yte["time"] > cap
    yte["event"][over] = False
    yte["time"][over] = cap
    return yte


def npi_matrix(npi_tr, npi_te):
    # single-feature; impute median
    med = np.nanmedian(npi_tr)
    a = np.nan_to_num(npi_tr, nan=med).reshape(-1, 1)
    b = np.nan_to_num(npi_te, nan=med).reshape(-1, 1)
    sc = StandardScaler().fit(a)
    return sc.transform(a), sc.transform(b)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def eval_fold_predictions(ytr, yte, risk_by_t: dict, times):
    row = {}
    for t in times:
        p = risk_by_t[t]
        try:
            row[f"AUC{int(round(t/Y))}y"] = float(cumulative_dynamic_auc(ytr, yte, p, [t])[0][0])
        except Exception:
            row[f"AUC{int(round(t/Y))}y"] = np.nan
        try:
            # brier_score expects survival probs
            row[f"Brier{int(round(t/Y))}y"] = float(brier_score(ytr, yte, 1 - p, [t])[1][0])
        except Exception:
            row[f"Brier{int(round(t/Y))}y"] = np.nan
    try:
        tau = times[-1]
        row[f"UnoC_{int(round(tau/Y))}y"] = float(
            concordance_index_ipcw(ytr, yte, risk_by_t[tau], tau=tau)[0]
        )
    except Exception:
        row[f"UnoC_{int(round(times[-1]/Y))}y"] = np.nan
    return row


def calibration_slope(y_event_at_t, p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    # binary proxy among those informative; caller passes event by t among FU>=t or event earlier
    if len(np.unique(y_event_at_t)) < 2:
        return np.nan
    lp = np.log(p / (1 - p)).reshape(-1, 1)
    try:
        return float(LogisticRegression(C=1e6, max_iter=2000).fit(lp, y_event_at_t).coef_[0, 0])
    except Exception:
        return np.nan


def dca_net_benefit(y_bin, p, thresholds):
    n = len(y_bin)
    out = []
    for pt in thresholds:
        pred = (p >= pt).astype(int)
        tp = ((pred == 1) & (y_bin == 1)).sum()
        fp = ((pred == 1) & (y_bin == 0)).sum()
        nb = tp / n - fp / n * (pt / (1 - pt))
        out.append(nb)
    return np.array(out)


def event_by_horizon(yS, t):
    """1 if event by t, 0 if censored/survived beyond t; NaN if censored before t."""
    ev = yS["event"].astype(bool)
    tm = yS["time"].astype(float)
    lab = np.full(len(yS), np.nan)
    lab[(tm <= t) & ev] = 1
    lab[tm > t] = 0
    return lab


def bootstrap_ci(values_list, n_boot=N_BOOT, seed=RANDOM_STATE):
    """values_list: list of dicts per fold → overall mean; bootstrap over folds' pooled? 
    Better: bootstrap patient-level OOF. Here we bootstrap the fold-level means for simplicity
    when only fold rows exist; for OOF arrays use patient bootstrap."""
    arr = np.asarray(values_list, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.RandomState(seed)
    means = []
    for _ in range(n_boot):
        samp = rng.choice(arr, size=len(arr), replace=True)
        means.append(samp.mean())
    return float(np.mean(arr)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def bootstrap_metric_from_oof(yS, oof_risk, t, metric="auc", n_boot=N_BOOT):
    rng = np.random.RandomState(RANDOM_STATE)
    n = len(yS)
    stats = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        ys = yS[idx]
        p = oof_risk[idx]
        # need train ref for IPCW — use full as approx (optimistic); prefer split
        try:
            if metric == "auc":
                # use same sample as train/test approx for bootstrap stability
                half = n // 2
                tr, te = idx[:half], idx[half:]
                if len(tr) < 50 or len(te) < 50:
                    continue
                val = cumulative_dynamic_auc(yS[tr], yS[te], oof_risk[te], [t])[0][0]
            elif metric == "brier":
                half = n // 2
                tr, te = idx[:half], idx[half:]
                val = brier_score(yS[tr], yS[te], 1 - oof_risk[te], [t])[1][0]
            else:
                continue
            if np.isfinite(val):
                stats.append(val)
        except Exception:
            continue
    if not stats:
        return np.nan, np.nan, np.nan
    return float(np.mean(stats)), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


# ---------------------------------------------------------------------------
# Main CV for DRFI
# ---------------------------------------------------------------------------

def run_drfi_cv(F, yS, meta, label="main"):
    print(f"\n=== DRFI CV [{label}] n={len(F)} events={int(yS['event'].sum())} ===")
    kf = KFold(N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    models = ["NPI_Cox", "Cox", "Cox_ER_ayri", "RSF"]
    fold_rows = []
    # OOF storage for 5y risk
    oof = {m: np.full(len(F), np.nan) for m in models}
    oof10 = {m: np.full(len(F), np.nan) for m in models}

    for fold, (tr, te) in enumerate(kf.split(F)):
        print(f"  fold {fold+1}/{N_SPLITS}")
        Xtr, Xte, _, _ = imp_scale(F.iloc[tr], F.iloc[te])
        ytr, yte = yS[tr], cap_test(yS[tr], yS[te])
        er_tr = F["er"].values[tr]
        er_te = F["er"].values[te]
        # fix nan er → 0 after noting
        er_tr = np.nan_to_num(er_tr, nan=0).astype(int)
        er_te = np.nan_to_num(er_te, nan=0).astype(int)

        tmax = min(ytr["time"].max(), yte["time"].max())
        times = np.array([5 * Y, 10 * Y])
        times = times[times < tmax]
        if len(times) == 0:
            continue

        preds = {}

        # NPI baseline
        npi_tr, npi_te = npi_matrix(meta["npi"].values[tr], meta["npi"].values[te])
        npi_cox = CoxPHSurvivalAnalysis(alpha=1.0).fit(npi_tr, ytr)
        preds["NPI_Cox"] = {
            t: risk_from_surv_fns(npi_cox.predict_survival_function(npi_te), t) for t in times
        }

        cox = CoxPHSurvivalAnalysis(alpha=1.0).fit(Xtr, ytr)
        preds["Cox"] = {t: risk_from_surv_fns(cox.predict_survival_function(Xte), t) for t in times}

        ec = ERStratCox().fit(Xtr, ytr, er_tr)
        preds["Cox_ER_ayri"] = {t: ec.risk_at(Xte, er_te, t) for t in times}

        rsf = RandomSurvivalForest(
            n_estimators=300,
            min_samples_leaf=15,
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ).fit(Xtr, ytr)
        preds["RSF"] = {
            t: risk_from_surv_fns(rsf.predict_survival_function(Xte), t) for t in times
        }

        for mname, pt in preds.items():
            row = eval_fold_predictions(ytr, yte, pt, times)
            row.update({"fold": fold, "model": mname, "analysis": label})
            fold_rows.append(row)
            if 5 * Y in pt:
                oof[mname][te] = pt[5 * Y]
            if 10 * Y in pt:
                oof10[mname][te] = pt[10 * Y]

    metrics_df = pd.DataFrame(fold_rows)
    summary_rows = []
    for m in models:
        sub = metrics_df[metrics_df["model"] == m]
        if sub.empty:
            continue
        rec = {"model": m, "analysis": label}
        for c in sub.columns:
            if c in ("fold", "model", "analysis"):
                continue
            mean, lo, hi = bootstrap_ci(sub[c].values)
            rec[c] = mean
            rec[f"{c}_lo"] = lo
            rec[f"{c}_hi"] = hi
        summary_rows.append(rec)
    summary = pd.DataFrame(summary_rows)
    return metrics_df, summary, oof, oof10


def cohort_loo(F, yS, meta, label="cohort_loo"):
    print(f"\n=== Cohort LOO [{label}] ===")
    cohorts = sorted(pd.unique(meta["cohort"]))
    models = ["Cox", "RSF"]
    rows = []
    for c in cohorts:
        te = np.where(meta["cohort"].values == c)[0]
        tr = np.where(meta["cohort"].values != c)[0]
        if len(te) < 40 or len(tr) < 100:
            continue
        print(f"  holdout cohort {c}: n_te={len(te)}")
        Xtr, Xte, _, _ = imp_scale(F.iloc[tr], F.iloc[te])
        ytr, yte = yS[tr], cap_test(yS[tr], yS[te])
        times = np.array([5 * Y, 10 * Y])
        times = times[times < min(ytr["time"].max(), yte["time"].max())]
        if len(times) == 0:
            continue
        cox = CoxPHSurvivalAnalysis(alpha=1.0).fit(Xtr, ytr)
        rsf = RandomSurvivalForest(
            n_estimators=250, min_samples_leaf=15, max_features="sqrt", n_jobs=-1, random_state=RANDOM_STATE
        ).fit(Xtr, ytr)
        preds = {
            "Cox": {t: risk_from_surv_fns(cox.predict_survival_function(Xte), t) for t in times},
            "RSF": {t: risk_from_surv_fns(rsf.predict_survival_function(Xte), t) for t in times},
        }
        for mname, pt in preds.items():
            row = eval_fold_predictions(ytr, yte, pt, times)
            row.update({"holdout_cohort": c, "model": mname, "analysis": label})
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Site competing risks (cause-specific Cox + CIF approx)
# ---------------------------------------------------------------------------

def run_site_competing(F, s_clean, meta):
    print("\n=== Site competing-risk (cause-specific Cox) ===")
    sites = first_site_labels(s_clean)
    causes = ["bone_first", "visceral_first", "cns_first"]
    # align index with F
    assert len(sites) == len(F)

    kf = KFold(N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_cif = {c: np.full(len(F), np.nan) for c in causes}
    rows = []

    for fold, (tr, te) in enumerate(kf.split(F)):
        print(f"  site fold {fold+1}")
        Xtr, Xte, _, _ = imp_scale(F.iloc[tr], F.iloc[te])
        t5 = 5 * Y
        for cause in causes:
            # cause-specific: event=1 if this cause, else 0 (competing treated as censor)
            ev_tr = (sites["cause"].values[tr] == cause)
            ev_te = (sites["cause"].values[te] == cause)
            tm_tr = sites["etime"].values[tr]
            tm_te = sites["etime"].values[te]
            ytr = Surv.from_arrays(ev_tr, tm_tr)
            yte = Surv.from_arrays(ev_te, tm_te)
            yte = cap_test(ytr, yte)
            if ytr["event"].sum() < 8:
                continue
            try:
                cox = CoxPHSurvivalAnalysis(alpha=1.0).fit(Xtr, ytr)
                risk = risk_from_surv_fns(cox.predict_survival_function(Xte), t5)
                oof_cif[cause][te] = risk
                # tdAUC for this cause
                auc = cumulative_dynamic_auc(ytr, yte, risk, [t5])[0][0]
                rows.append(
                    {
                        "fold": fold,
                        "cause": cause,
                        "n_events_train": int(ytr["event"].sum()),
                        "AUC5y": float(auc),
                        "mean_pred_cif5y": float(np.mean(risk)),
                    }
                )
            except Exception as e:
                print(f"    skip {cause}: {e}")

    site_metrics = pd.DataFrame(rows)
    site_summary = (
        site_metrics.groupby("cause")
        .agg(AUC5y_mean=("AUC5y", "mean"), AUC5y_std=("AUC5y", "std"), n_folds=("AUC5y", "count"))
        .reset_index()
        if len(site_metrics)
        else pd.DataFrame()
    )
    cause_counts = sites["cause"].value_counts().rename_axis("cause").reset_index(name="n")
    return site_metrics, site_summary, oof_cif, cause_counts, sites


# ---------------------------------------------------------------------------
# Plots + HTML
# ---------------------------------------------------------------------------

def plot_metrics_bars(summary: pd.DataFrame, col: str, title: str, fname: str):
    if summary.empty or col not in summary.columns:
        return ""
    d = summary.dropna(subset=[col]).sort_values(col)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(d["model"], d[col], color="#1f5a4e")
    if f"{col}_lo" in d.columns:
        for i, (_, r) in enumerate(d.iterrows()):
            ax.plot([r[f"{col}_lo"], r[f"{col}_hi"]], [i, i], color="#333", lw=2)
    ax.set_xlabel(col)
    ax.set_title(title)
    return save_fig(fig, fname)


def plot_calibration(yS, oof_p, title, fname, t=5 * Y):
    lab = event_by_horizon(yS, t)
    mask = np.isfinite(lab) & np.isfinite(oof_p)
    if mask.sum() < 50:
        return ""
    p = oof_p[mask]
    y = lab[mask].astype(int)
    # quantile bins
    try:
        bins = pd.qcut(p, q=8, duplicates="drop")
    except Exception:
        return ""
    df = pd.DataFrame({"p": p, "y": y, "bin": bins})
    g = df.groupby("bin", observed=True).agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size"))
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "--", color="#aaa")
    ax.plot(g["pred"], g["obs"], "o-", color="#8b3a3a")
    for _, r in g.iterrows():
        ax.text(r["pred"], r["obs"], str(int(r["n"])), fontsize=7)
    ax.set_xlabel("Predicted 5y DRFI risk")
    ax.set_ylabel("Observed event-by-5y rate")
    ax.set_title(title)
    slope = calibration_slope(y, p)
    ax.text(0.05, 0.9, f"calib slope≈{slope:.2f}", transform=ax.transAxes)
    return save_fig(fig, fname)


def plot_dca(yS, oof_dict, fname, t=5 * Y):
    lab = event_by_horizon(yS, t)
    thresholds = np.arange(0.05, 0.61, 0.01)
    fig, ax = plt.subplots(figsize=(7, 5))
    # treat all / none
    mask0 = np.isfinite(lab)
    y = lab[mask0].astype(int)
    prev = y.mean()
    nb_all = prev - (1 - prev) * (thresholds / (1 - thresholds))
    ax.plot(thresholds, nb_all, "--", color="#888", label="Treat all")
    ax.plot(thresholds, np.zeros_like(thresholds), ":", color="#888", label="Treat none")
    for name, p in oof_dict.items():
        mask = mask0 & np.isfinite(p)
        if mask.sum() < 50:
            continue
        nb = dca_net_benefit(lab[mask].astype(int), p[mask], thresholds)
        ax.plot(thresholds, nb, lw=2, label=name)
    ax.set_xlabel("Threshold probability (5y DRFI)")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision curve analysis (5y)")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, max(0.2, prev * 1.1))
    return save_fig(fig, fname)


def plot_site_bars(cause_counts, site_summary, fname):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(cause_counts["cause"], cause_counts["n"], color="#34495e")
    axes[0].set_title("Cause counts")
    axes[0].tick_params(axis="x", rotation=30)
    if not site_summary.empty:
        axes[1].bar(site_summary["cause"], site_summary["AUC5y_mean"], color="#2f5d50")
        axes[1].set_title("Cause-specific Cox tdAUC 5y")
        axes[1].tick_params(axis="x", rotation=30)
        axes[1].set_ylim(0.4, 0.9)
    return save_fig(fig, fname)


def df_html(df, digits=3):
    if df is None or len(df) == 0:
        return "<p><em>Yok</em></p>"
    show = df.copy()
    for c in show.columns:
        if pd.api.types.is_float_dtype(show[c]):
            show[c] = show[c].map(lambda x: f"{x:.{digits}f}" if pd.notna(x) else "")
    return show.to_html(index=False, classes="tbl", border=0, escape=True)


def build_html(ctx):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""<!DOCTYPE html>
<html lang="tr"><head><meta charset="utf-8"/>
<title>BreastPredict DRFI — TRIPOD-style report</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;background:#f4f1ec;color:#1a1814;margin:0}}
.wrap{{max-width:1100px;margin:auto;padding:28px 18px 70px}}
h1{{font-size:32px;margin:8px 0}}
.brand{{color:#7a3030;font-size:12px;letter-spacing:.12em;text-transform:uppercase;font-weight:700}}
section{{background:#fffcf7;border:1px solid #e2dbd0;border-radius:14px;padding:18px;margin:14px 0}}
.tbl{{width:100%;border-collapse:collapse;font-size:12.5px}}
.tbl th{{background:#f1ebe1;text-align:left;padding:7px}}
.tbl td{{padding:6px 8px;border-bottom:1px solid #e2dbd0}}
.plot{{width:100%;max-width:520px;border:1px solid #e2dbd0;border-radius:10px}}
.plots{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}}
.note{{background:#f7efe4;border-left:4px solid #7a3030;padding:10px;color:#5a554c;font-size:13px}}
.chip{{display:inline-block;background:#efe8dc;border-radius:999px;padding:3px 10px;margin:2px;font-size:12px}}
</style></head><body><div class="wrap">
<div class="brand">BreastPredict · DRFI Survival Overhaul</div>
<h1>STEEP DRFI sağkalım ve competing-risk raporu</h1>
<p>Internal validation (5-fold + cohort LOO). Binary M1/M2A/M2B kaldırıldı. Metrikler: time-dependent AUC, Uno C, IPCW Brier, kalibrasyon, DCA.</p>
<div><span class="chip">{now}</span><span class="chip">seed {RANDOM_STATE}</span><span class="chip">CTS5: uygulanamadı (Ki-67 yok)</span></div>

<section><h2>1. Kohort akışı</h2>
{df_html(ctx['flow'], 0)}
<div class="note">DRFI olay = uzak nüks (DR) veya meme kanserine bağlı ölüm (DeathBreast). Başka nedenle ölüm sansür. Evre 4 / BENIGN / PHYL / DCIS dışlandı. RT ve cerrahi özellikte yok; CT/HT var-yok.</div>
<p>Ana analiz n={ctx['n_main']}, DRFI olay={ctx['n_events']}. Duyarlılık (Complete.Rec=YES only) n={ctx['n_sens']}.</p>
</section>

<section><h2>2. Ana DRFI model karşılaştırması (5-fold, bootstrap CI)</h2>
{df_html(ctx['summary_main'])}
<div class="plots">
<img class="plot" src="data:image/png;base64,{ctx['img_auc5']}"/>
<img class="plot" src="data:image/png;base64,{ctx['img_unoc']}"/>
</div>
</section>

<section><h2>3. Kalibrasyon ve DCA (OOF 5y)</h2>
<div class="plots">
<img class="plot" src="data:image/png;base64,{ctx['img_cal']}"/>
<img class="plot" src="data:image/png;base64,{ctx['img_dca']}"/>
</div>
</section>

<section><h2>4. Kohort leave-one-out (internal-external)</h2>
{df_html(ctx['cohort_loo'])}
</section>

<section><h2>5. Duyarlılık: Complete.Rec.History=NO dışlandı</h2>
<div class="note">Eksik nüks öyküsü (çoğunlukla Kohort 3) DR oranını şişirir; ana analizde tutuldu, burada dışlandı.</div>
{df_html(ctx['summary_sens'])}
</section>

<section><h2>6. Bölge — competing risks (kemik / visseral / SSS)</h2>
{df_html(ctx['cause_counts'], 0)}
{df_html(ctx['site_summary'])}
<div class="plots"><img class="plot" src="data:image/png;base64,{ctx['img_site']}"/></div>
</section>

<section><h2>7. TRIPOD+AI / sınırlılıklar</h2>
<ul>
<li>Pre-trastuzumab dönem: HER2+ risk overestimate; dış validasyonda rekalibrasyon gerekir.</li>
<li>CTS5 yapılamadı (Ki-67 yok). Baseline: NPI-Cox.</li>
<li>Impute/scale fold içinde; importance tüm veride seçilmedi.</li>
<li>Report1/2 binary AUC iyileşme iddiaları bu çerçevede geçersiz.</li>
<li>External validation yok — cohort LOO yalnızca internal-external benzeri.</li>
</ul>
<p><b>Özellikler:</b> {", ".join(FEAT_COLS)}</p>
</section>
</div></body></html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading raw tables...")
    s5, db = load_raw()
    flow = cohort_flow(s5)
    flow.to_excel(TAB / "cohort_flow.xlsx", index=False)
    print(flow)

    s, F, yS, meta = build_analysis_frame(s5, db, require_complete_rec=False)
    # save analysis extract
    extract = meta.copy()
    extract["event_drfi"] = yS["event"].astype(int)
    extract["time_drfi"] = yS["time"]
    extract.to_excel(TAB / "analysis_ids.xlsx", index=False)
    F.to_excel(TAB / "features_main.xlsx", index=False)

    # descriptive
    desc = pd.DataFrame(
        {
            "metric": [
                "n",
                "drfi_events",
                "dr_only",
                "deathbreast_no_dr",
                "complete_rec_no",
                "pr_missing",
                "her2_missing",
            ],
            "value": [
                len(F),
                int(yS["event"].sum()),
                int(((meta["dr"] == 1)).sum()),
                int(((meta["death_breast"] == 1) & (meta["dr"] == 0)).sum()),
                int((meta["complete_rec"] == "NO").sum()),
                int(F["pr_miss"].sum()),
                int(F["her2_miss"].sum()),
            ],
        }
    )
    desc.to_excel(TAB / "descriptives.xlsx", index=False)

    fold_main, summary_main, oof5, oof10 = run_drfi_cv(F, yS, meta, label="main")
    fold_main.to_excel(TAB / "cv_folds_main.xlsx", index=False)
    summary_main.to_excel(TAB / "cv_summary_main.xlsx", index=False)

    # sensitivity
    s2, F2, yS2, meta2 = build_analysis_frame(s5, db, require_complete_rec=True)
    fold_sens, summary_sens, oof5s, _ = run_drfi_cv(F2, yS2, meta2, label="complete_rec_only")
    fold_sens.to_excel(TAB / "cv_folds_sensitivity.xlsx", index=False)
    summary_sens.to_excel(TAB / "cv_summary_sensitivity.xlsx", index=False)

    # cohort LOO
    loo = cohort_loo(F, yS, meta)
    loo.to_excel(TAB / "cohort_loo.xlsx", index=False)

    # site CR
    site_metrics, site_summary, oof_cif, cause_counts, sites = run_site_competing(F, s, meta)
    site_metrics.to_excel(TAB / "site_folds.xlsx", index=False)
    site_summary.to_excel(TAB / "site_summary.xlsx", index=False)
    cause_counts.to_excel(TAB / "site_cause_counts.xlsx", index=False)
    sites.to_excel(TAB / "site_labels.xlsx", index=False)

    # plots
    img_auc5 = plot_metrics_bars(summary_main, "AUC5y", "5y time-dependent AUC (main)", "auc5y_main.png")
    img_unoc = plot_metrics_bars(summary_main, "UnoC_10y", "Uno C-index @10y (main)", "unoc_main.png")
    # pick best by AUC5y for calib
    best = summary_main.sort_values("AUC5y", ascending=False).iloc[0]["model"]
    img_cal = plot_calibration(yS, oof5[best], f"Calibration 5y — {best}", "calib_5y.png")
    img_dca = plot_dca(yS, {k: oof5[k] for k in ["NPI_Cox", "Cox", "RSF"] if k in oof5}, "dca_5y.png")
    img_site = plot_site_bars(cause_counts, site_summary, "site_competing.png")

    # yearly risk table from best model OOF (distribution)
    risk_years = {}
    # refit full for illustration of mean predicted risk by horizon using Cox OOF 5/10 only
    risk_tbl = pd.DataFrame(
        {
            "model": list(oof5.keys()),
            "mean_oof_risk_5y": [np.nanmean(oof5[m]) for m in oof5],
            "mean_oof_risk_10y": [np.nanmean(oof10[m]) for m in oof10],
        }
    )
    risk_tbl.to_excel(TAB / "mean_predicted_risks.xlsx", index=False)

    ctx = {
        "flow": flow,
        "n_main": len(F),
        "n_events": int(yS["event"].sum()),
        "n_sens": len(F2),
        "summary_main": summary_main,
        "summary_sens": summary_sens,
        "cohort_loo": loo,
        "cause_counts": cause_counts,
        "site_summary": site_summary,
        "img_auc5": img_auc5,
        "img_unoc": img_unoc,
        "img_cal": img_cal,
        "img_dca": img_dca,
        "img_site": img_site,
    }
    html = build_html(ctx)
    (OUT / "report_drfi.html").write_text(html, encoding="utf-8")

    summary_json = {
        "framework": "STEEP_DRFI_survival",
        "n_main": len(F),
        "n_events": int(yS["event"].sum()),
        "n_sensitivity_complete_rec": len(F2),
        "features": FEAT_COLS,
        "excluded": ["Stage4", "BENIGN", "PHYL", "DCIS", "RT_feature", "surgery_feature", "M1", "M2A", "M2B_binary"],
        "best_model_by_AUC5y": best,
        "metrics_main": summary_main.to_dict(orient="records"),
        "cts5": "not_applicable_no_ki67",
        "her2_note": "pre-trastuzumab cohort; recalibration needed for modern external validation",
        "generated": datetime.now().isoformat(timespec="seconds"),
    }
    (OUT / "summary.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")

    readme = """# sonuclar — DRFI pipeline outputs

- `report_drfi.html` — ana TRIPOD-style rapor (tarayıcıda açın)
- `summary.json` — özet metrikler
- `tables/` — kohort akışı, CV, duyarlılık, site competing-risk
- `figures/` — AUC, kalibrasyon, DCA, site

Çalıştırma: `python run_drfi_pipeline.py`
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print("\nDone. Report:", OUT / "report_drfi.html")
    print(json.dumps({k: summary_json[k] for k in ["n_main", "n_events", "best_model_by_AUC5y"]}, indent=2))


if __name__ == "__main__":
    main()
