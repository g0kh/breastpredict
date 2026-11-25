"""
Journal submission figures 1–6 (300 dpi PNG + PDF).
Run: python PMJ/manu/make_journal_figures.py
Exports to PMJ/manu/figures/journal/
Existing M01–M11 files are left untouched.
"""
from __future__ import annotations

import json
import shutil
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
PMJ = Path(__file__).resolve().parents[1]
MANU = Path(__file__).resolve().parent
OUT = MANU / "figures" / "journal"
OUT.mkdir(parents=True, exist_ok=True)
SRC_MANU = MANU / "figures"

sys.path.insert(0, str(PMJ))
import make_manuscript_figures as mm  # noqa: E402

Y = mm.Y
NATURE = mm.NATURE
TEAL_CORAL = mm.TEAL_CORAL
NEUTRAL = mm.NEUTRAL
CHARCOAL = mm.CHARCOAL
SEQ = mm.SEQ
FEATURE_LABELS = mm.FEATURE_LABELS

# Locked model / site colors for submission figures
MODEL_COLOR = {
    "NPI_Cox": "#00468B",
    "Cox": "#ED0000",
    "Cox_ER_ayri": "#42B540",
    "RSF": "#0099B4",
}
MODEL_LABEL = {
    "NPI_Cox": "NPI-Cox baseline",
    "Cox": "Cox proportional hazards",
    "Cox_ER_ayri": "Cox (ER-stratified)",
    "RSF": "Random survival forest",
}
SITE_COLOR = {
    "Bone": "#2A9D8F",
    "Visceral": "#E64B35",
    "CNS": "#F39B7F",
    "Other": "#6C757D",
}
SITE_ORDER = ["CNS", "Visceral", "Bone", "Other"]


def panel_letter(ax, letter: str, x=-0.08, y=1.08):
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        color=CHARCOAL,
        va="top",
        ha="left",
        clip_on=False,
    )


def save_fig(fig, stem: str):
    png = OUT / f"{stem}.png"
    pdf = OUT / f"{stem}.pdf"
    mm._safe_savefig(fig, png, dpi=300, bbox_inches="tight", pad_inches=0.25)
    mm._safe_savefig(fig, pdf, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"  wrote {stem}.png/.pdf")


# ---------------------------------------------------------------------------
# Figure 1 — Cohort case mix
# ---------------------------------------------------------------------------

def figure1_cohort(D):
    F = D["features"].copy()
    ext = D["ext"].copy()

    def pn(series):
        s = series.astype(str).str.strip().str.lower()
        return np.where(
            s.isin(("pozitif", "positive", "pos")),
            1.0,
            np.where(s.isin(("negatif", "negative", "neg")), 0.0, np.nan),
        )

    dev = pd.DataFrame(
        {
            "Age at diagnosis (years)": F["age"],
            "Tumor size (mm)": np.exp(F["log_size"]),
            "Positive lymph nodes": np.expm1(F["ln"]),
            "Histologic grade": F["grade"],
            "ER-positive": F["er"],
            "PR-positive": F["pr"],
            "HER2-positive": F["her2"],
            "Postmenopausal": F["post"],
            "Lobular histology": F["lobular"],
        }
    )
    loc = pd.DataFrame(
        {
            "Age at diagnosis (years)": pd.to_numeric(ext["yas_tani"], errors="coerce"),
            "Tumor size (mm)": pd.to_numeric(ext["tumor_boyutu_mm"], errors="coerce"),
            "Positive lymph nodes": pd.to_numeric(ext["pozitif_lenf_nodu_sayisi"], errors="coerce"),
            "Histologic grade": pd.to_numeric(ext["grade"], errors="coerce"),
            "ER-positive": pn(ext["ER"]),
            "PR-positive": pn(ext["PR"]),
            "HER2-positive": pn(ext["HER2"]),
            "Postmenopausal": ext["menopoz"].astype(str).str.strip().str.lower().map({"post": 1.0, "pre": 0.0}),
            "Lobular histology": ext["histoloji"].astype(str).str.upper().str.contains("ILC").astype(float),
        }
    )

    continuous = ["Age at diagnosis (years)", "Tumor size (mm)", "Positive lymph nodes", "Histologic grade"]
    binary = ["ER-positive", "PR-positive", "HER2-positive", "Postmenopausal", "Lobular histology"]

    fig = plt.figure(figsize=(12.5, 9.0))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1.0], hspace=0.42, wspace=0.35, top=0.90, bottom=0.08)

    # Shared legend above top row
    leg_handles = [
        mpatches.Patch(facecolor=NATURE[0], alpha=0.55, edgecolor=NATURE[0], label="METABRIC (training)"),
        mpatches.Patch(facecolor=TEAL_CORAL[1], alpha=0.55, edgecolor=TEAL_CORAL[1], label="Local center"),
    ]
    fig.legend(
        handles=leg_handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=9.5,
        bbox_to_anchor=(0.5, 0.98),
    )

    for i, col in enumerate(continuous):
        ax = fig.add_subplot(gs[0, i])
        a = dev[col].dropna().astype(float).values
        b = loc[col].dropna().astype(float).values

        if col == "Histologic grade":
            grades = [1, 2, 3]
            w = 0.36
            x = np.arange(len(grades))
            da = [(a == g).mean() if len(a) else 0 for g in grades]
            db = [(b == g).mean() if len(b) else 0 for g in grades]
            ax.bar(x - w / 2, da, w, color=NATURE[0], edgecolor="white")
            ax.bar(x + w / 2, db, w, color=TEAL_CORAL[1], edgecolor="white")
            ax.set_xticks(x)
            ax.set_xticklabels(["1", "2", "3"])
            ax.set_ylabel("Proportion" if i == 0 else "")
            ax.set_ylim(0, max(max(da), max(db), 0.1) * 1.25)
        elif col == "Positive lymph nodes":
            sns.kdeplot(
                a, ax=ax, color=NATURE[0], fill=True, alpha=0.35, linewidth=1.8,
                cut=0, clip=(0, float(np.nanmax(a)) if len(a) else 20),
            )
            sns.kdeplot(
                b, ax=ax, color=TEAL_CORAL[1], fill=True, alpha=0.35, linewidth=1.8,
                cut=0, clip=(0, float(np.nanmax(b)) if len(b) else 20),
            )
            ax.axvline(np.nanmedian(a), color=NATURE[0], ls="--", lw=1.2)
            ax.axvline(np.nanmedian(b), color=TEAL_CORAL[1], ls="--", lw=1.2)
            xmax = max(np.nanpercentile(a, 99), np.nanpercentile(b, 99), 1)
            ax.set_xlim(0, xmax)
            ax.set_ylabel("Density" if i == 0 else "")
        else:
            sns.kdeplot(a, ax=ax, color=NATURE[0], fill=True, alpha=0.35, linewidth=1.8, cut=0)
            sns.kdeplot(b, ax=ax, color=TEAL_CORAL[1], fill=True, alpha=0.35, linewidth=1.8, cut=0)
            ax.axvline(np.nanmedian(a), color=NATURE[0], ls="--", lw=1.2)
            ax.axvline(np.nanmedian(b), color=TEAL_CORAL[1], ls="--", lw=1.2)
            ax.set_ylabel("Density" if i == 0 else "")

        ax.set_title(col, loc="left", fontsize=10, fontweight="bold", color=CHARCOAL, pad=4)
        ax.set_xlabel(col if col != "Histologic grade" else "Grade")
        mm.light_grid(ax)
        sns.despine(ax=ax)
        if i == 0:
            panel_letter(ax, "A", x=-0.18, y=1.12)

    # Bottom categorical bars
    axb = fig.add_subplot(gs[1, :])
    x = np.arange(len(binary))
    w = 0.36
    # nanmean over non-missing values — same basis as Table 1
    dev_rate = [float(np.nanmean(dev[c].astype(float))) for c in binary]
    loc_rate = [float(np.nanmean(loc[c].astype(float))) for c in binary]
    axb.bar(x - w / 2, dev_rate, w, color=NATURE[0], edgecolor="white")
    axb.bar(x + w / 2, loc_rate, w, color=TEAL_CORAL[1], edgecolor="white")
    for i, (d, l) in enumerate(zip(dev_rate, loc_rate)):
        axb.text(i - w / 2, d + 0.02, f"{d:.0%}", ha="center", fontsize=8.5, color=CHARCOAL)
        axb.text(i + w / 2, l + 0.02, f"{l:.0%}", ha="center", fontsize=8.5, color=CHARCOAL)
    axb.set_xticks(x)
    axb.set_xticklabels(binary, fontsize=9)
    axb.set_ylim(0, 1.15)
    axb.set_ylabel("Proportion")
    mm.light_grid(axb)
    sns.despine(ax=axb)
    panel_letter(axb, "B", x=-0.04, y=1.08)

    save_fig(fig, "Fig1_cohort_case_mix")


# ---------------------------------------------------------------------------
# Figure 2 — SHAP
# ---------------------------------------------------------------------------

def figure2_shap(D):
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.util import Surv

    F = D["features"].copy()
    y = Surv.from_arrays(
        D["analysis"]["event_drfi"].astype(bool).values,
        D["analysis"]["time_drfi"].astype(float).values,
    )
    feat_names = list(F.columns)
    im = IterativeImputer(max_iter=10, random_state=42).fit(F)
    X = im.transform(F)
    for j, c in enumerate(feat_names):
        if c in ("pr", "her2", "post", "lobular", "ct", "ht", "pr_miss", "her2_miss", "er"):
            X[:, j] = np.clip(np.round(X[:, j]), 0, 1)
        if c == "grade":
            X[:, j] = np.clip(np.round(X[:, j]), 1, 3)
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    cox = CoxPHSurvivalAnalysis(alpha=1.0).fit(Xs, y)
    coef = cox.coef_
    rng = np.random.RandomState(42)
    idx = rng.choice(len(Xs), size=min(800, len(Xs)), replace=False)
    Xs_s = Xs[idx]
    shap_vals = Xs_s * coef.reshape(1, -1)
    mean_abs = np.abs(shap_vals).mean(axis=0)
    order = np.argsort(mean_abs)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.0), gridspec_kw={"width_ratios": [1.45, 1]})
    ax = axes[0]
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list("seq", SEQ)
    for yi, j in enumerate(order):
        vals = shap_vals[:, j]
        feat = Xs_s[:, j]
        fmin, fmax = np.percentile(feat, [5, 95])
        f = np.clip((feat - fmin) / (fmax - fmin + 1e-9), 0, 1)
        jitter = rng.normal(0, 0.08, size=len(vals))
        ax.scatter(vals, np.full_like(vals, yi) + jitter, c=f, cmap=cmap, s=16, alpha=0.75, linewidths=0)
    nice = [FEATURE_LABELS.get(feat_names[j], feat_names[j]) for j in order]
    ax.axvline(0, color=NEUTRAL, lw=0.8, ls="--")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(nice)
    ax.set_xlabel("SHAP value (impact on log partial hazard)")
    mm.light_grid(ax, "x")
    sns.despine(ax=ax)
    panel_letter(ax, "A", x=-0.22, y=1.05)

    ax2 = axes[1]
    ax2.barh(range(len(order)), mean_abs[order], color=NATURE[0])
    ax2.set_yticks(range(len(order)))
    ax2.set_yticklabels([])
    ax2.set_xlabel("Mean |SHAP|")
    mm.light_grid(ax2, "x")
    sns.despine(ax=ax2)
    panel_letter(ax2, "B", x=-0.08, y=1.05)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[0], fraction=0.035, pad=0.02)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(["0.0\nLow / No", "0.5", "1.0\nHigh / Yes"])
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Feature value", fontsize=9)

    fig.subplots_adjust(wspace=0.22, top=0.92, bottom=0.12)
    save_fig(fig, "Fig2_SHAP_explainability")


# ---------------------------------------------------------------------------
# Figure 3 — Scenario A discrimination
# ---------------------------------------------------------------------------

def figure3_scenario_a(D):
    d = D["int_ext"].copy()
    boot = D["boot"].set_index("model")
    # Order: RSF, Cox_ER, Cox, NPI (external AUC desc; Cox above NPI on internal when tied ext)
    order = ["RSF", "Cox_ER_ayri", "Cox", "NPI_Cox"]
    d = d.set_index("model").loc[order].reset_index()

    scores = D["scores"].copy()
    risk = scores["RSF_risk5y"].astype(float)
    t = scores["DRFI_sure_gun"].astype(float)
    e = scores["DRFI_olay"].astype(int)
    event_5y = ((e == 1) & (t <= 5 * Y)).astype(int)

    fig, axes = plt.subplots(
        1, 2, figsize=(12.0, 4.8), gridspec_kw={"width_ratios": [0.58, 0.42]}
    )

    # --- Panel A: dumbbell ---
    ax = axes[0]
    y = np.arange(len(d))[::-1]  # top = first in order
    # remapped so RSF is at top
    plot_order = list(range(len(d)))
    y_pos = {i: len(d) - 1 - i for i in plot_order}

    for i, (_, r) in enumerate(d.iterrows()):
        yi = y_pos[i]
        m = r["model"]
        ax.plot(
            [r["internal_AUC5y"], r["external_AUC5y"]],
            [yi, yi],
            color="#D0D4DA",
            lw=3.0,
            zorder=1,
        )
        ax.scatter(
            r["internal_AUC5y"],
            yi,
            s=90,
            color=NATURE[0],
            zorder=3,
            label="Internal CV 5y tdAUC" if i == 0 else "",
        )
        ax.scatter(
            r["external_AUC5y"],
            yi,
            s=90,
            color=NATURE[1],
            zorder=3,
            label="External 5y tdAUC" if i == 0 else "",
        )
        if m in boot.index and np.isfinite(boot.loc[m, "AUC_lo"]):
            ax.plot(
                [boot.loc[m, "AUC_lo"], boot.loc[m, "AUC_hi"]],
                [yi, yi],
                color=NATURE[1],
                lw=2.0,
                alpha=0.75,
                zorder=2,
            )

    rsf_yi = y_pos[0]
    ax.axhspan(rsf_yi - 0.35, rsf_yi + 0.35, color="#FFF3CD", alpha=0.55, zorder=0)
    ax.text(
        0.52,
        rsf_yi + 0.38,
        "Best model (RSF)",
        fontsize=8,
        color=NATURE[1],
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    ax.set_yticks([y_pos[i] for i in range(len(d))])
    ax.set_yticklabels([MODEL_LABEL[m] for m in d["model"]])
    ax.axvline(0.5, color=NEUTRAL, ls="--", lw=0.8)
    ax.set_xlabel("5-year time-dependent AUC")
    ax.set_xlim(0.45, 1.00)
    mm.light_grid(ax, "x")
    sns.despine(ax=ax)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.55, 1.18), ncol=2, fontsize=8.5)
    panel_letter(ax, "A", x=-0.28, y=1.12)

    # --- Panel B: risk strip ---
    ax = axes[1]
    rng = np.random.RandomState(42)
    groups = [
        ("Event-free at 5 years\n(n = 66)", event_5y == 0, TEAL_CORAL[0]),
        ("DRFI event by 5 years\n(n = 14)", event_5y == 1, TEAL_CORAL[1]),
    ]
    means = []
    for i, (lab, mask, col) in enumerate(groups):
        vals = risk[mask].dropna().values
        means.append(float(vals.mean()) if len(vals) else np.nan)
        jitter = rng.normal(0, 0.06, size=len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter,
            vals,
            s=28,
            alpha=0.65,
            color=col,
            edgecolors="white",
            linewidths=0.4,
            zorder=2,
        )
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        ax.plot([i - 0.18, i + 0.18], [med, med], color=CHARCOAL, lw=2, zorder=3)
        ax.plot([i, i], [q1, q3], color=CHARCOAL, lw=1.2, zorder=3)
        ax.text(
            i + 0.28,
            means[-1],
            f"Mean: {means[-1]:.2f}",
            va="center",
            ha="left",
            fontsize=8.5,
            color=CHARCOAL,
            fontweight="bold",
        )

    ax.set_xticks([0, 1])
    ax.set_xticklabels([g[0] for g in groups], fontsize=8.5)
    ax.set_ylabel("RSF predicted 5-year DRFI risk")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(-0.55, 1.75)
    mm.light_grid(ax)
    sns.despine(ax=ax)
    panel_letter(ax, "B", x=-0.12, y=1.12)

    fig.subplots_adjust(wspace=0.35, top=0.82, bottom=0.16, left=0.18, right=0.97)
    save_fig(fig, "Fig3_scenario_A_discrimination")


# ---------------------------------------------------------------------------
# Figure 4 — Calibration, DCA, KM
# ---------------------------------------------------------------------------

def _event_by_horizon(time, event, t):
    lab = np.full(len(time), np.nan)
    ev = event.astype(bool)
    lab[(time <= t) & ev] = 1
    lab[time > t] = 0
    return lab


def figure4_usability(D):
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test

    sc = D["scores"].copy()
    time = sc["DRFI_sure_gun"].astype(float).values
    event = sc["DRFI_olay"].astype(int).values
    t5 = 5 * Y
    lab = _event_by_horizon(time, event, t5)

    models = {
        "NPI_Cox": sc["NPI_Cox_risk5y"].astype(float).values,
        "Cox": sc["Cox_risk5y"].astype(float).values,
        "Cox_ER_ayri": sc["Cox_ER_ayri_risk5y"].astype(float).values,
        "RSF": sc["RSF_risk5y"].astype(float).values,
    }

    fig = plt.figure(figsize=(12.5, 9.2))
    gs = GridSpec(2, 2, height_ratios=[1.0, 1.15], hspace=0.38, wspace=0.28, top=0.94, bottom=0.12)

    # --- A Calibration ---
    ax = fig.add_subplot(gs[0, 0])
    ax.plot([0, 1], [0, 1], "--", color="#AAAAAA", lw=1)
    for name, p in models.items():
        mask = np.isfinite(lab) & np.isfinite(p)
        if mask.sum() < 20:
            continue
        pp, yy = p[mask], lab[mask].astype(int)
        try:
            bins = pd.qcut(pp, q=5, duplicates="drop")
        except Exception:
            continue
        g = pd.DataFrame({"p": pp, "y": yy, "bin": bins}).groupby("bin", observed=True).agg(
            pred=("p", "mean"), obs=("y", "mean"), n=("y", "size")
        )
        ax.plot(
            g["pred"],
            g["obs"],
            "o-",
            color=MODEL_COLOR[name],
            label=MODEL_LABEL[name],
            ms=6,
            lw=1.8,
        )
    ax.set_xlabel("Predicted 5y DRFI risk")
    ax.set_ylabel("Observed event rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.text(
        0.98,
        0.05,
        "RSF raw calibration slope = 1.65",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color=CHARCOAL,
        fontweight="bold",
    )
    mm.light_grid(ax)
    sns.despine(ax=ax)
    panel_letter(ax, "A", x=-0.12, y=1.08)

    # --- B DCA ---
    ax = fig.add_subplot(gs[0, 1])
    thresholds = np.arange(0.05, 0.55, 0.01)
    mask0 = np.isfinite(lab)
    yb0 = lab[mask0].astype(int)
    prev = yb0.mean() if len(yb0) else 0
    nb_all = prev - (1 - prev) * (thresholds / (1 - thresholds))
    ax.plot(thresholds, nb_all, "--", color=NEUTRAL, label="Treat all", lw=1.2)
    ax.plot(thresholds, np.zeros_like(thresholds), ":", color=NEUTRAL, label="Treat none", lw=1.2)
    for name, p in models.items():
        mask = mask0 & np.isfinite(p)
        if mask.sum() < 20:
            continue
        n = mask.sum()
        yb = lab[mask].astype(int)
        pb = p[mask]
        nbs = []
        for pt in thresholds:
            pred = (pb >= pt).astype(int)
            tp = ((pred == 1) & (yb == 1)).sum()
            fp = ((pred == 1) & (yb == 0)).sum()
            nbs.append(tp / n - fp / n * (pt / (1 - pt)))
        ax.plot(thresholds, nbs, color=MODEL_COLOR[name], lw=2, label=MODEL_LABEL[name])
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_ylim(-0.15, 0.20)
    mm.light_grid(ax)
    sns.despine(ax=ax)
    ax.legend(
        frameon=False,
        fontsize=7,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )
    panel_letter(ax, "B", x=-0.12, y=1.08)

    # --- C KM ---
    ax = fig.add_subplot(gs[1, :])
    sc2 = D["scores"].copy()
    sc2["risk"] = sc2["RSF_risk5y"].astype(float)
    sc2["time"] = sc2["DRFI_sure_gun"].astype(float) / Y
    sc2["event"] = sc2["DRFI_olay"].astype(int)
    sc2 = sc2.dropna(subset=["risk", "time"])
    sc2["tertile"] = pd.qcut(sc2["risk"], 3, labels=["Low risk", "Intermediate", "High risk"])

    kmf = KaplanMeierFitter()
    colors = [NATURE[2], NATURE[4], NATURE[1]]
    tertile_order = ["Low risk", "Intermediate", "High risk"]
    for col, lab_t in zip(colors, tertile_order):
        sub = sc2[sc2["tertile"] == lab_t]
        if len(sub) < 5:
            continue
        kmf.fit(sub["time"], sub["event"], label=f"{lab_t} (n={len(sub)})")
        kmf.plot_survival_function(ax=ax, color=col, ci_show=True, linewidth=2.2)

    # Log-rank
    try:
        lr = multivariate_logrank_test(sc2["time"], sc2["tertile"].astype(str), sc2["event"])
        p_lr = lr.p_value
        p_txt = f"Log-rank p = {p_lr:.3g}" if p_lr >= 0.001 else "Log-rank p < 0.001"
    except Exception:
        p_txt = "Log-rank p = n/a"
    ax.text(0.98, 0.08, p_txt, transform=ax.transAxes, ha="right", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel("Years from diagnosis")
    ax.set_ylabel("Distant recurrence-free probability")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, 12.5)
    ax.legend(frameon=False, loc="lower left", fontsize=8.5)
    mm.light_grid(ax)
    sns.despine(ax=ax)
    panel_letter(ax, "C", x=-0.05, y=1.06)

    # Number at risk table
    times_risk = [0, 2, 4, 6, 8, 10, 12]
    table_rows = []
    for lab_t in tertile_order:
        sub = sc2[sc2["tertile"] == lab_t]
        row = []
        for t in times_risk:
            # at risk: time >= t
            n_at = int((sub["time"] >= t).sum())
            row.append(str(n_at))
        table_rows.append(row)

    # Draw table below KM
    ax_table = ax.inset_axes([0.0, -0.42, 1.0, 0.28])
    ax_table.set_axis_off()
    col_labels = [str(t) for t in times_risk]
    row_labels = ["Low", "Intermediate", "High"]
    tbl = ax_table.table(
        cellText=table_rows,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.25)
    ax_table.text(0.0, 1.15, "Number at risk", transform=ax_table.transAxes, fontsize=9, fontweight="bold", ha="left")

    fig.subplots_adjust(bottom=0.18)
    save_fig(fig, "Fig4_calibration_DCA_KM")


# ---------------------------------------------------------------------------
# Figure 5 — Scenario B timing
# ---------------------------------------------------------------------------

def figure5_timing(D):
    ext = D["ext_met"].set_index("model")
    models = ["NPI_Cox", "Cox", "Cox_ER_ayri", "RSF"]
    horizons_auc = [("AUC2y", 2), ("AUC5y", 5), ("AUC10y", 10)]
    horizons_b = [("Brier2y", 2), ("Brier5y", 5), ("Brier10y", 10)]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    # Shared legend above
    handles = [
        plt.Line2D([0], [0], color=MODEL_COLOR[m], marker="o", lw=2, label=MODEL_LABEL[m]) for m in models
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=8.5,
        bbox_to_anchor=(0.5, 0.98),
    )

    ax = axes[0]
    for m in models:
        xs = [h for _, h in horizons_auc]
        ys = [ext.loc[m, col] for col, _ in horizons_auc]
        ax.plot(xs, ys, "o-", color=MODEL_COLOR[m], lw=2.2, ms=7)
    # Annotate RSF only
    for col, h in horizons_auc:
        y = ext.loc["RSF", col]
        ax.text(h, y + 0.018, f"{y:.3f}", ha="center", fontsize=8.5, color=MODEL_COLOR["RSF"], fontweight="bold")
    ax.set_xticks([2, 5, 10])
    ax.set_xticklabels(["2-year", "5-year", "10-year"])
    ax.set_xlabel("Prediction horizon")
    ax.set_ylabel("External time-dependent AUC")
    ax.set_ylim(0.55, 1.02)
    mm.light_grid(ax)
    sns.despine(ax=ax)
    panel_letter(ax, "A", x=-0.12, y=1.08)

    ax = axes[1]
    for m in models:
        xs = [h for _, h in horizons_b]
        ys = [ext.loc[m, col] for col, _ in horizons_b]
        ax.plot(xs, ys, "s-", color=MODEL_COLOR[m], lw=2.0, ms=6)
    ax.set_xticks([2, 5, 10])
    ax.set_xticklabels(["2-year", "5-year", "10-year"])
    ax.set_xlabel("Prediction horizon")
    ax.set_ylabel("Brier score (lower is better)")
    ax.set_ylim(0.0, 0.22)
    ax.set_title("Overall prediction error (Brier score)", loc="left", fontsize=11, fontweight="bold", color=CHARCOAL, pad=6)
    mm.light_grid(ax)
    sns.despine(ax=ax)
    panel_letter(ax, "B", x=-0.12, y=1.08)

    # Title for left panel (panel A has no separate title string beyond letter)
    axes[0].set_title("Discrimination over time", loc="left", fontsize=11, fontweight="bold", color=CHARCOAL, pad=6)

    fig.subplots_adjust(top=0.82, bottom=0.14, wspace=0.30)
    save_fig(fig, "Fig5_scenario_B_timing")


# ---------------------------------------------------------------------------
# Figure 6 — Scenario C site
# ---------------------------------------------------------------------------

def figure6_site(D):
    # Heatmap data
    db = D["db"][["hasta_id", "er", "pr", "her2"]].copy()
    sites = D["sites"][["METABRIC.ID", "cause"]].rename(columns={"METABRIC.ID": "hasta_id"})
    m = db.merge(sites, on="hasta_id", how="inner")
    m = m[m["cause"].isin(["bone_first", "visceral_first", "cns_first", "other_first"])]
    m["ER"] = m["er"].map(
        lambda x: "ER+" if str(x).lower() in ("pos", "positive") else ("ER-" if str(x).lower() in ("neg", "negative") else np.nan)
    )
    m["PR"] = m["pr"].map(lambda x: "PR+" if str(x) == "Positive" else ("PR-" if str(x) == "Negative" else np.nan))
    m["HER2"] = m["her2"].map(
        lambda x: "HER2+" if str(x) == "Positive" else ("HER2-" if str(x) == "Negative" else np.nan)
    )
    m["Site"] = m["cause"].map(mm.site_bucket)

    def ph(r):
        er, pr, her2 = r["ER"], r["PR"], r["HER2"]
        if pd.isna(er) or pd.isna(her2):
            return None
        if her2 == "HER2+":
            return "HER2+"
        if er == "ER-" and (pr == "PR-" or pd.isna(pr)):
            return "TNBC-like"
        if er == "ER+":
            return "ER+/HER2-"
        return "Other"

    m["Phenotype"] = m.apply(ph, axis=1)
    m = m.dropna(subset=["Phenotype"])
    ph_counts = m["Phenotype"].value_counts()
    ct = pd.crosstab(m["Phenotype"], m["Site"], normalize="index") * 100
    order_p = ["ER+/HER2-", "HER2+", "TNBC-like", "Other"]
    order_s = ["Bone", "Visceral", "CNS", "Other"]
    ct = ct.reindex([p for p in order_p if p in ct.index])
    ct = ct[[c for c in order_s if c in ct.columns]]
    row_labels = [f"{p} (n = {int(ph_counts.get(p, 0))})" for p in ct.index]

    # Site AUC + external counts
    site = pd.read_excel(ROOT / "sonuclar" / "tables" / "site_summary.xlsx").copy()
    ext_site = pd.read_excel(ROOT / "sonuclar" / "external" / "tables" / "external_site_counts.xlsx").copy()
    label_map = {
        "bone_first": "Bone",
        "visceral_first": "Visceral",
        "cns_first": "CNS",
        "other_first": "Other",
    }
    site["label"] = site["cause"].map(label_map)
    site = site.dropna(subset=["label"])
    # barh: first category at bottom — reverse so CNS is at top
    auc_order = ["Bone", "Visceral", "CNS"]
    site["label"] = pd.Categorical(site["label"], categories=auc_order, ordered=True)
    site = site.sort_values("label")

    ext_site["label"] = ext_site["site_cause"].map(label_map)
    ext_site = ext_site.dropna(subset=["label"])
    count_order = ["Other", "Bone", "Visceral", "CNS"]
    ext_site["label"] = pd.Categorical(ext_site["label"], categories=count_order, ordered=True)
    ext_site = ext_site.sort_values("label")

    fig = plt.figure(figsize=(12.5, 9.0))
    gs = GridSpec(2, 2, height_ratios=[1.05, 1.0], hspace=0.40, wspace=0.32, top=0.94, bottom=0.10)

    # --- A Heatmap ---
    ax = fig.add_subplot(gs[0, :])
    sns.heatmap(
        ct,
        annot=True,
        fmt=".1f",
        cmap="mako",
        ax=ax,
        cbar_kws={"label": "% within phenotype"},
        linewidths=0.5,
        linecolor="white",
        yticklabels=row_labels,
    )
    ax.set_xlabel("First distant site")
    ax.set_ylabel("Receptor phenotype")
    panel_letter(ax, "A", x=-0.06, y=1.08)

    # --- B Internal AUC ---
    ax = fig.add_subplot(gs[1, 0])
    y = np.arange(len(site))
    ax.barh(
        y,
        site["AUC5y_mean"],
        xerr=site["AUC5y_std"],
        color=[SITE_COLOR[l] for l in site["label"]],
        height=0.55,
        error_kw={"ecolor": "#333333", "capsize": 3, "lw": 1.0},
    )
    for i, (_, r) in enumerate(site.iterrows()):
        ax.text(
            0.42,
            i,
            f"{r['AUC5y_mean']:.2f} ± {r['AUC5y_std']:.2f}",
            va="center",
            ha="left",
            fontsize=8.5,
            color="white",
            fontweight="bold",
            zorder=5,
        )
    ax.axvline(0.5, color=NEUTRAL, ls="--", lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(list(site["label"]))
    ax.set_xlim(0.4, 1.05)
    ax.set_xlabel("Cause-specific 5y AUC (CV mean ± SD)")
    ax.set_title(
        "Internal cause-specific 5-year AUC (METABRIC)",
        loc="left",
        fontsize=10.5,
        fontweight="bold",
        color=CHARCOAL,
        pad=6,
    )
    mm.light_grid(ax, "x")
    sns.despine(ax=ax)
    panel_letter(ax, "B", x=-0.14, y=1.10)

    # --- C External counts ---
    ax = fig.add_subplot(gs[1, 1])
    y2 = np.arange(len(ext_site))
    ax.barh(
        y2,
        ext_site["n"],
        color=[SITE_COLOR.get(l, NEUTRAL) for l in ext_site["label"]],
        height=0.55,
    )
    for i, (_, r) in enumerate(ext_site.iterrows()):
        ax.text(r["n"] + 0.2, i, f"n = {int(r['n'])}", va="center", fontsize=9, color=CHARCOAL)
    ax.set_yticks(y2)
    ax.set_yticklabels(list(ext_site["label"]))
    ax.set_xlabel("Patients with this first distant site")
    ax.set_xlim(0, max(ext_site["n"].max() + 3, 14))
    ax.set_title(
        "External validation first-site counts (n = 23)",
        loc="left",
        fontsize=10.5,
        fontweight="bold",
        color=CHARCOAL,
        pad=6,
    )
    mm.light_grid(ax, "x")
    sns.despine(ax=ax)
    panel_letter(ax, "C", x=-0.14, y=1.10)

    save_fig(fig, "Fig6_scenario_C_first_site")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    mm.apply_style()
    print("Loading data...")
    D = mm.load_all()

    print("\n[1/6] Figure 1 cohort...")
    figure1_cohort(D)

    print("\n[2/6] Figure 2 SHAP...")
    figure2_shap(D)

    print("\n[3/6] Figure 3 Scenario A...")
    figure3_scenario_a(D)

    print("\n[4/6] Figure 4 usability...")
    figure4_usability(D)

    print("\n[5/6] Figure 5 timing...")
    figure5_timing(D)

    print("\n[6/6] Figure 6 site...")
    figure6_site(D)

    # Supplementary model card copy
    src = SRC_MANU / "M02_model_card.png"
    if src.exists():
        shutil.copy2(src, OUT / "SuppFig1_model_card.png")
        pdf = SRC_MANU / "M02_model_card.pdf"
        if pdf.exists():
            shutil.copy2(pdf, OUT / "SuppFig1_model_card.pdf")
        print("  copied SuppFig1_model_card")

    expected = [
        "Fig1_cohort_case_mix",
        "Fig2_SHAP_explainability",
        "Fig3_scenario_A_discrimination",
        "Fig4_calibration_DCA_KM",
        "Fig5_scenario_B_timing",
        "Fig6_scenario_C_first_site",
    ]
    print("\nVerify...")
    missing = []
    for stem in expected:
        for ext in (".png", ".pdf"):
            p = OUT / f"{stem}{ext}"
            if not p.exists():
                missing.append(p.name)
            else:
                print(f"  OK {p.name} ({p.stat().st_size // 1024} KB)")
    if missing:
        raise SystemExit(f"Missing: {missing}")
    print(f"\nDone. {len(expected)} figures in {OUT}")


if __name__ == "__main__":
    main()
