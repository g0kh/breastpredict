"""
Manuscript figure pack for oncologists — English Nature style.
Run: python PMJ/manu/make_manu_figures.py
Exports PNG (300 dpi) + PDF to PMJ/manu/figures/
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
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
PMJ = Path(__file__).resolve().parents[1]
MANU = Path(__file__).resolve().parent
OUT = MANU / "figures"
OUT.mkdir(parents=True, exist_ok=True)
SRC_FIG = PMJ / "figures"

sys.path.insert(0, str(PMJ))
import make_manuscript_figures as mm  # noqa: E402

NATURE = mm.NATURE
LANCET = mm.LANCET
TEAL_CORAL = mm.TEAL_CORAL
NEUTRAL = mm.NEUTRAL
CHARCOAL = mm.CHARCOAL
FEATURE_LABELS = mm.FEATURE_LABELS
Y = mm.Y

MODEL_NAMES = {
    "NPI_Cox": "NPI-Cox baseline",
    "Cox": "Cox proportional hazards",
    "Cox_ER_ayri": "Cox (ER-stratified)",
    "RSF": "Random survival forest",
}


def save_manu(fig, stem: str):
    png = OUT / f"{stem}.png"
    pdf = OUT / f"{stem}.pdf"
    mm._safe_savefig(fig, png, dpi=300, bbox_inches="tight", pad_inches=0.35)
    mm._safe_savefig(fig, pdf, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)
    print(f"  wrote {stem}.png/.pdf")


def copy_selected(src_stem: str, dst_stem: str):
    for ext in (".png", ".pdf"):
        src = SRC_FIG / f"{src_stem}{ext}"
        dst = OUT / f"{dst_stem}{ext}"
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, dst)
    print(f"  copied {src_stem} -> {dst_stem}")


# ---------------------------------------------------------------------------
# Regenerate fixed Fig06 into PMJ, then copy as M05
# ---------------------------------------------------------------------------

def regenerate_fig06_and_export_m05(D):
    # Temporarily redirect mm.OUT so Fig06 lands in PMJ/figures (existing path)
    old_out = mm.OUT
    mm.OUT = SRC_FIG
    mm.FIGURE_LOG.clear()
    try:
        mm.fig06_calib_dca(D)
    finally:
        mm.OUT = old_out
    copy_selected("Fig06_calibration_hist_dca", "M05_external_calibration_DCA")


def export_selected_copies():
    copy_selected("Fig18_features_training_vs_center", "M01_cohort_features_training_vs_center")
    copy_selected("Fig05_shap_beeswarm_mean", "M03_model_explainability_SHAP")
    copy_selected("Fig13_km_by_risk_tertile", "M06_KM_by_risk_tertile")
    copy_selected("Fig12_heatmap_receptor_site", "M07_receptor_by_first_site")


# ---------------------------------------------------------------------------
# M02 Model card
# ---------------------------------------------------------------------------

def m02_model_card(summary: dict):
    fig = plt.figure(figsize=(11.5, 7.2))
    ax = fig.add_axes([0.04, 0.04, 0.92, 0.78])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def card(x, y, w, h, title, lines, face="#F7F9FC", edge=NATURE[0]):
        box = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.2,
            edgecolor=edge,
            facecolor=face,
            transform=ax.transAxes,
            clip_on=False,
        )
        ax.add_patch(box)
        ax.text(
            x + 0.018,
            y + h - 0.04,
            title,
            fontsize=11,
            fontweight="bold",
            color=CHARCOAL,
            va="top",
            transform=ax.transAxes,
        )
        ty = y + h - 0.09
        for line in lines:
            ax.text(
                x + 0.018,
                ty,
                line,
                fontsize=8.6,
                color=CHARCOAL,
                va="top",
                transform=ax.transAxes,
            )
            ty -= 0.045

    n_main = summary.get("n_main", "—")
    n_events = summary.get("n_events", "—")
    best = summary.get("best_model_by_AUC5y", "RSF")

    card(
        0.02,
        0.52,
        0.46,
        0.42,
        "Endpoint (STEEP DRFI)",
        [
            "Distant recurrence-free interval (DRFI):",
            "time to distant metastasis or breast-cancer death",
            f"Training cohort: n={n_main}  |  events: {n_events}",
            "External hospital cohort: n=80 (23 DRFI events)",
            "Primary horizon shown: 5-year predicted risk",
        ],
        face="#EEF4FA",
        edge=NATURE[0],
    )

    feat_lines = [
        "Age at diagnosis",
        "Tumor size (log mm)",
        "Positive lymph nodes (log)",
        "Histologic grade",
        "ER / PR / HER2 receptor status",
        "Postmenopausal status",
        "Lobular histology",
        "Chemotherapy / endocrine therapy",
    ]
    card(
        0.52,
        0.52,
        0.46,
        0.42,
        "Input features (clinical only)",
        feat_lines,
        face="#F3FAF7",
        edge=NATURE[2],
    )

    card(
        0.02,
        0.08,
        0.46,
        0.38,
        "Cohort rules",
        [
            "Included: invasive breast cancer with DRFI follow-up",
            "Excluded: stage 4 at diagnosis; DCIS / benign / Phyllodes",
            "Not used as labels: binary M1 / M2A metastasis flags",
            "Missing PR/HER2 handled with indicator flags",
            "RT / surgery not used as model inputs",
        ],
        face="#FFF6F3",
        edge=NATURE[1],
    )

    card(
        0.52,
        0.08,
        0.46,
        0.38,
        "Models compared",
        [
            "NPI-Cox baseline (Nottingham prognostic index)",
            "Cox proportional hazards",
            "Cox ER-stratified",
            f"Random survival forest  ← best by 5y tdAUC ({best})",
            "Metrics: 5y time-dependent AUC, Uno C, Brier",
        ],
        face="#F8F5FC",
        edge=LANCET[4],
    )

    mm.fig_banner(
        fig,
        "Model card: what the DRFI risk score uses and predicts",
        "Oncologist summary — clinical inputs only; STEEP distant recurrence-free interval",
        top_room=0.82,
    )
    save_manu(fig, "M02_model_card")


# ---------------------------------------------------------------------------
# M04 Internal vs external
# ---------------------------------------------------------------------------

def m04_internal_vs_external(D):
    d = D["int_ext"].copy()
    boot = D["boot"].set_index("model")
    d = d.sort_values("external_AUC5y")

    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    y = np.arange(len(d))
    for i, (_, r) in enumerate(d.iterrows()):
        m = r["model"]
        ax.plot(
            [r["internal_AUC5y"], r["external_AUC5y"]],
            [i, i],
            color="#D0D4DA",
            lw=3.0,
            zorder=1,
        )
        ax.scatter(
            r["internal_AUC5y"],
            i,
            s=90,
            color=NATURE[0],
            zorder=3,
            label="Internal CV 5y tdAUC" if i == 0 else "",
        )
        ax.scatter(
            r["external_AUC5y"],
            i,
            s=90,
            color=NATURE[1],
            zorder=3,
            label="External 5y tdAUC" if i == 0 else "",
        )
        if m in boot.index and np.isfinite(boot.loc[m, "AUC_lo"]):
            ax.plot(
                [boot.loc[m, "AUC_lo"], boot.loc[m, "AUC_hi"]],
                [i, i],
                color=NATURE[1],
                lw=2.0,
                alpha=0.75,
                zorder=2,
            )

    # Highlight RSF
    if "RSF" in set(d["model"]):
        rsf_i = list(d["model"]).index("RSF")
        ax.axhspan(rsf_i - 0.35, rsf_i + 0.35, color="#FFF3CD", alpha=0.55, zorder=0)
        ax.text(
            0.47,
            rsf_i + 0.38,
            "Best model (RSF)",
            fontsize=8,
            color=NATURE[1],
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_NAMES.get(m, m) for m in d["model"]])
    ax.axvline(0.5, color=NEUTRAL, ls="--", lw=0.8)
    ax.set_xlabel("5-year time-dependent AUC (discrimination)")
    ax.set_xlim(0.45, 1.05)
    mm.light_grid(ax, "x")
    sns.despine(ax=ax)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    mm.fig_banner(
        fig,
        "Does the score travel to our hospital? Internal vs external discrimination",
        "Dumbbell: internal CV vs external 5y tdAUC; whiskers = external bootstrap 95% CI",
        top_room=0.78,
    )
    fig.subplots_adjust(left=0.28, right=0.98)
    save_manu(fig, "M04_internal_vs_external_performance")


# ---------------------------------------------------------------------------
# M08 Three clinical scenarios
# ---------------------------------------------------------------------------

def m08_three_scenarios(D):
    cv = D["cv_sum"].set_index("model")
    ext = D["ext_met"].set_index("model")
    site = pd.read_excel(ROOT / "sonuclar" / "tables" / "site_summary.xlsx")
    ext_site = pd.read_excel(ROOT / "sonuclar" / "external" / "tables" / "external_site_counts.xlsx")

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.0))

    # --- Panel A: Will DRFI occur (risk)? ---
    ax = axes[0]
    models = ["NPI_Cox", "Cox", "Cox_ER_ayri", "RSF"]
    x = np.arange(len(models))
    w = 0.36
    int_auc = [cv.loc[m, "AUC5y"] for m in models]
    ext_auc = [ext.loc[m, "AUC5y"] for m in models]
    ax.bar(x - w / 2, int_auc, width=w, color=NATURE[0], label="Internal CV")
    ax.bar(x + w / 2, ext_auc, width=w, color=NATURE[1], label="External")
    for i, m in enumerate(models):
        ax.text(i - w / 2, int_auc[i] + 0.01, f"{int_auc[i]:.2f}", ha="center", fontsize=7, color=NATURE[0])
        ax.text(i + w / 2, ext_auc[i] + 0.01, f"{ext_auc[i]:.2f}", ha="center", fontsize=7, color=NATURE[1])
    ax.set_xticks(x)
    ax.set_xticklabels(["NPI-Cox", "Cox", "Cox ER", "RSF"], fontsize=8.5)
    ax.set_ylim(0.55, 0.95)
    ax.set_ylabel("5-year time-dependent AUC")
    ax.set_title(
        "(A) Will distant relapse / DRFI occur?",
        loc="left",
        fontsize=10.5,
        fontweight="bold",
        color=CHARCOAL,
        pad=8,
    )
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    mm.light_grid(ax)
    sns.despine(ax=ax)
    # Uno C footnote on panel
    uno = cv.loc["RSF", "UnoC_10y"]
    ax.text(
        0.02,
        0.02,
        f"RSF Uno C (internal): {uno:.3f}",
        transform=ax.transAxes,
        fontsize=7.5,
        color=NEUTRAL,
    )

    # --- Panel B: Timing — risk by year ---
    ax = axes[1]
    horizons = [("AUC2y", "2y"), ("AUC5y", "5y"), ("AUC10y", "10y")]
    for i, m in enumerate(models):
        ys = [ext.loc[m, h] for h, _ in horizons]
        ax.plot(
            [1, 2, 3],
            ys,
            "o-",
            color=LANCET[i],
            lw=2,
            ms=6,
            label=MODEL_NAMES[m].split("(")[0].strip() if m != "Cox_ER_ayri" else "Cox ER-strat.",
        )
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["2-year", "5-year", "10-year"])
    ax.set_ylabel("External time-dependent AUC")
    ax.set_ylim(0.55, 1.0)
    ax.set_title(
        "(B) Timing — risk by year",
        loc="left",
        fontsize=10.5,
        fontweight="bold",
        color=CHARCOAL,
        pad=8,
    )
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    mm.light_grid(ax)
    sns.despine(ax=ax)

    # --- Panel C: Where first? ---
    ax = axes[2]
    site = site.copy()
    site["label"] = site["cause"].map(
        {
            "bone_first": "Bone",
            "visceral_first": "Visceral",
            "cns_first": "CNS",
        }
    )
    site = site.dropna(subset=["label"]).sort_values("label")
    colors_c = [TEAL_CORAL[0], NATURE[1], NATURE[3]]
    bars = ax.barh(
        site["label"],
        site["AUC5y_mean"],
        color=colors_c[: len(site)],
        xerr=site["AUC5y_std"],
        error_kw={"ecolor": NEUTRAL, "capsize": 3, "lw": 1},
        height=0.55,
    )
    ax.set_xlabel("Internal cause-specific 5y AUC (CV mean ± SD)")
    ax.set_xlim(0.4, 1.0)
    ax.set_title(
        "(C) Where first (bone / visceral / CNS)?",
        loc="left",
        fontsize=10.5,
        fontweight="bold",
        color=CHARCOAL,
        pad=8,
    )
    mm.light_grid(ax, "x")
    sns.despine(ax=ax)

    # External site counts annotation
    ext_map = {
        "bone_first": "Bone",
        "visceral_first": "Visceral",
        "cns_first": "CNS",
        "other_first": "Other",
    }
    bits = []
    for _, r in ext_site.iterrows():
        lab = ext_map.get(str(r["site_cause"]), str(r["site_cause"]))
        bits.append(f"{lab} n={int(r['n'])}")
    ax.text(
        0.02,
        -0.18,
        "External first-site counts (descriptive; small n): " + "  ·  ".join(bits),
        transform=ax.transAxes,
        fontsize=7.2,
        color=NEUTRAL,
        ha="left",
        va="top",
        wrap=True,
    )
    ax.text(
        0.02,
        0.02,
        "External site AUC not claimed (unstable)",
        transform=ax.transAxes,
        fontsize=7.2,
        color=NATURE[1],
        fontweight="bold",
    )

    mm.fig_banner(
        fig,
        "Three clinical scenarios: risk, timing, and first distant site",
        "Scenario A/B = DRFI survival metrics; Scenario C = competing-risk / site (external counts descriptive)",
        top_room=0.78,
    )
    fig.subplots_adjust(wspace=0.32, bottom=0.18)
    save_manu(fig, "M08_three_scenarios_AI_performance")


# ---------------------------------------------------------------------------
# M09 Risk separation raincloud / histogram
# ---------------------------------------------------------------------------

def m09_risk_separation(D):
    scores = D["scores"].copy()
    # Prefer RSF 5y risk; event by 5y if possible, else any DRFI event with time<=5y
    risk = scores["RSF_risk5y"].astype(float)
    t = scores["DRFI_sure_gun"].astype(float)
    e = scores["DRFI_olay"].astype(int)
    event_5y = ((e == 1) & (t <= 5 * Y)).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0), gridspec_kw={"width_ratios": [1.15, 1]})

    # Left: raincloud-style strip + box
    ax = axes[0]
    rng = np.random.default_rng(42)
    for i, (lab, mask, col) in enumerate(
        [
            ("No DRFI by 5y", event_5y == 0, TEAL_CORAL[0]),
            ("DRFI by 5y", event_5y == 1, TEAL_CORAL[1]),
        ]
    ):
        vals = risk[mask].dropna().values
        if len(vals) == 0:
            continue
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
            label=lab,
        )
        # box overlay
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        ax.plot([i - 0.18, i + 0.18], [med, med], color=CHARCOAL, lw=2, zorder=3)
        ax.plot([i, i], [q1, q3], color=CHARCOAL, lw=1.2, zorder=3)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No DRFI by 5y", "DRFI by 5y"])
    ax.set_ylabel("RSF predicted 5-year DRFI risk")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(-0.55, 1.55)
    mm.light_grid(ax)
    sns.despine(ax=ax)
    ax.set_title("Risk by outcome", loc="left", fontsize=11, fontweight="bold", color=CHARCOAL)

    # Right: overlapping histograms
    ax = axes[1]
    v0 = risk[event_5y == 0].dropna()
    v1 = risk[event_5y == 1].dropna()
    ax.hist(v0, bins=10, density=True, alpha=0.55, color=TEAL_CORAL[0])
    ax.hist(v1, bins=8, density=True, alpha=0.65, color=TEAL_CORAL[1])
    if len(v0):
        ax.axvline(v0.mean(), color=TEAL_CORAL[0], ls="--", lw=1.4)
    if len(v1):
        ax.axvline(v1.mean(), color=TEAL_CORAL[1], ls="--", lw=1.4)
    ax.set_xlabel("RSF predicted 5-year DRFI risk")
    ax.set_ylabel("Density")
    mm.light_grid(ax)
    sns.despine(ax=ax)
    ax.set_title("Risk distribution", loc="left", fontsize=11, fontweight="bold", color=CHARCOAL, pad=10)
    # Color key outside plot, upper-right of histogram panel
    ax.text(
        1.05,
        0.98,
        f"No event (n={len(v0)})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color=TEAL_CORAL[0],
        fontweight="bold",
        clip_on=False,
    )
    ax.text(
        1.05,
        0.88,
        f"DRFI by 5y (n={len(v1)})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color=TEAL_CORAL[1],
        fontweight="bold",
        clip_on=False,
    )

    # Separation callout
    if len(v0) and len(v1):
        delta = float(v1.mean() - v0.mean())
        fig.text(
            0.42,
            0.02,
            f"Mean risk: no event {v0.mean():.2f} vs DRFI {v1.mean():.2f}  (Δ = {delta:+.2f})  — external cohort",
            ha="center",
            fontsize=9,
            color=NEUTRAL,
        )

    mm.fig_banner(
        fig,
        "Scenario A: do higher scores mark patients who relapse?",
        "External cohort — RSF 5-year predicted risk by observed distant recurrence or breast-cancer death (DRFI)",
        top_room=0.78,
    )
    fig.subplots_adjust(bottom=0.14, wspace=0.30, right=0.82)
    save_manu(fig, "M09_scenario_A_risk_separation")


# ---------------------------------------------------------------------------
# M10 Scenario B — Timing (risk by year)
# ---------------------------------------------------------------------------

def m10_scenario_b_timing(D):
    ext = D["ext_met"].set_index("model")
    cv = D["cv_sum"].set_index("model")
    models = ["NPI_Cox", "Cox", "Cox_ER_ayri", "RSF"]
    short = {
        "NPI_Cox": "NPI-Cox",
        "Cox": "Cox",
        "Cox_ER_ayri": "Cox ER-strat.",
        "RSF": "RSF",
    }
    horizons = [("AUC2y", 2), ("AUC5y", 5), ("AUC10y", 10)]
    briers = [("Brier2y", 2), ("Brier5y", 5), ("Brier10y", 10)]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))

    # Left: external tdAUC trajectory
    ax = axes[0]
    for i, m in enumerate(models):
        xs = [h for _, h in horizons]
        ys = [ext.loc[m, col] for col, _ in horizons]
        ax.plot(
            xs,
            ys,
            "o-",
            color=LANCET[i],
            lw=2.2,
            ms=7,
            label=MODEL_NAMES[m],
        )
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.012, f"{y:.2f}", ha="center", fontsize=7.5, color=LANCET[i])
    ax.set_xticks([2, 5, 10])
    ax.set_xticklabels(["2-year", "5-year", "10-year"])
    ax.set_xlabel("Prediction horizon")
    ax.set_ylabel("External time-dependent AUC")
    ax.set_ylim(0.55, 1.02)
    ax.set_title(
        "Discrimination over time",
        loc="left",
        fontsize=11,
        fontweight="bold",
        color=CHARCOAL,
        pad=8,
    )
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    mm.light_grid(ax)
    sns.despine(ax=ax)

    # Right: external Brier (lower = better) + RSF callout table
    ax = axes[1]
    for i, m in enumerate(models):
        xs = [h for _, h in briers]
        ys = [ext.loc[m, col] for col, _ in briers]
        ax.plot(xs, ys, "s-", color=LANCET[i], lw=2.0, ms=6, label=short[m])
    ax.set_xticks([2, 5, 10])
    ax.set_xticklabels(["2-year", "5-year", "10-year"])
    ax.set_xlabel("Prediction horizon")
    ax.set_ylabel("External Brier score (lower is better)")
    ax.set_ylim(0.0, 0.22)
    ax.set_title(
        "Calibration error over time",
        loc="left",
        fontsize=11,
        fontweight="bold",
        color=CHARCOAL,
        pad=8,
    )
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    mm.light_grid(ax)
    sns.despine(ax=ax)

    # RSF highlight strip under figure
    rsf_line = "  ·  ".join(
        [f"{h}y AUC {ext.loc['RSF', c]:.3f}" for c, h in horizons]
        + [f"Uno C {ext.loc['RSF', 'UnoC_10y']:.3f}"]
    )
    fig.text(
        0.5,
        0.02,
        f"RSF (best overall): {rsf_line}   |   Internal 5y tdAUC {cv.loc['RSF', 'AUC5y']:.3f}",
        ha="center",
        fontsize=8.5,
        color=NEUTRAL,
    )

    mm.fig_banner(
        fig,
        "Scenario B: when is risk highest — discrimination by year (2 / 5 / 10)",
        "External cohort time-dependent AUC and Brier across horizons; distant recurrence or breast-cancer death (DRFI)",
        top_room=0.78,
    )
    fig.subplots_adjust(bottom=0.14, wspace=0.30)
    save_manu(fig, "M10_scenario_B_timing")


# ---------------------------------------------------------------------------
# M11 Scenario C — First distant site
# ---------------------------------------------------------------------------

def m11_scenario_c_site(D):
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
    # stable display order
    order = ["Bone", "Visceral", "CNS"]
    site["label"] = pd.Categorical(site["label"], categories=order, ordered=True)
    site = site.sort_values("label")

    ext_site["label"] = ext_site["site_cause"].map(label_map)
    ext_site = ext_site.dropna(subset=["label"])
    # keep Other for counts
    count_order = ["Bone", "Visceral", "CNS", "Other"]
    ext_site["label"] = pd.Categorical(ext_site["label"], categories=count_order, ordered=True)
    ext_site = ext_site.sort_values("label")

    colors = {
        "Bone": TEAL_CORAL[0],
        "Visceral": NATURE[1],
        "CNS": NATURE[3],
        "Other": NEUTRAL,
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))

    # Left: internal cause-specific AUC
    ax = axes[0]
    y = np.arange(len(site))
    ax.barh(
        y,
        site["AUC5y_mean"],
        xerr=site["AUC5y_std"],
        color=[colors[l] for l in site["label"]],
        height=0.55,
        error_kw={"ecolor": "#333333", "capsize": 3, "lw": 1.0},
    )
    ax.set_yticks(y)
    ax.set_yticklabels(list(site["label"]))
    # Values inside bars (white) — clear of whiskers
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
    ax.set_xlim(0.4, 1.05)
    ax.set_xlabel("Internal cause-specific 5y AUC (CV mean ± SD)")
    ax.set_title(
        "Can the model tell which site is first?",
        loc="left",
        fontsize=11,
        fontweight="bold",
        color=CHARCOAL,
        pad=8,
    )
    mm.light_grid(ax, "x")
    sns.despine(ax=ax)

    # Right: external first-site counts (descriptive)
    ax = axes[1]
    y2 = np.arange(len(ext_site))
    ax.barh(
        y2,
        ext_site["n"],
        color=[colors.get(l, NEUTRAL) for l in ext_site["label"]],
        height=0.55,
    )
    for i, (_, r) in enumerate(ext_site.iterrows()):
        ax.text(r["n"] + 0.15, i, f"n = {int(r['n'])}", va="center", fontsize=9, color=CHARCOAL)
    ax.set_yticks(y2)
    ax.set_yticklabels(list(ext_site["label"]))
    ax.set_xlabel("External patients with this first distant site")
    ax.set_xlim(0, max(ext_site["n"].max() + 3, 14))
    ax.set_title(
        "External first-site counts (descriptive)",
        loc="left",
        fontsize=11,
        fontweight="bold",
        color=CHARCOAL,
        pad=8,
    )
    mm.light_grid(ax, "x")
    sns.despine(ax=ax)

    fig.text(
        0.5,
        0.02,
        "Scenario C = competing-risk / first distant site  ·  Internal AUC only  ·  "
        "External counts descriptive — site AUC not claimed (small n, unstable)",
        ha="center",
        fontsize=8.5,
        color=NEUTRAL,
    )

    mm.fig_banner(
        fig,
        "Scenario C: where does distant disease appear first?",
        "Internal cause-specific discrimination for bone / visceral / CNS; external hospital site counts are descriptive only",
        top_room=0.78,
    )
    fig.subplots_adjust(bottom=0.14, wspace=0.32)
    save_manu(fig, "M11_scenario_C_first_site")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    mm.apply_style()
    print("Loading data...")
    D = mm.load_all()
    summary = json.loads((ROOT / "sonuclar" / "summary.json").read_text(encoding="utf-8"))

    print("\n[1/4] Fix Fig06 DCA legend and export M05...")
    regenerate_fig06_and_export_m05(D)

    print("\n[2/4] Copy selected figures (M01, M03, M06, M07)...")
    export_selected_copies()

    print("\n[3/4] Build oncologist panels M02, M04, M08–M11...")
    m02_model_card(summary)
    m04_internal_vs_external(D)
    m08_three_scenarios(D)
    m09_risk_separation(D)
    m10_scenario_b_timing(D)
    m11_scenario_c_site(D)

    expected = [
        "M01_cohort_features_training_vs_center",
        "M02_model_card",
        "M03_model_explainability_SHAP",
        "M04_internal_vs_external_performance",
        "M05_external_calibration_DCA",
        "M06_KM_by_risk_tertile",
        "M07_receptor_by_first_site",
        "M08_three_scenarios_AI_performance",
        "M09_scenario_A_risk_separation",
        "M10_scenario_B_timing",
        "M11_scenario_C_first_site",
    ]
    print("\n[4/4] Verify exports...")
    missing = []
    for stem in expected:
        for ext in (".png", ".pdf"):
            p = OUT / f"{stem}{ext}"
            if not p.exists():
                missing.append(p.name)
            else:
                print(f"  OK {p.name} ({p.stat().st_size // 1024} KB)")
    if missing:
        raise SystemExit(f"Missing files: {missing}")
    print(f"\nDone. {len(expected)} figures × PNG+PDF in {OUT}")


if __name__ == "__main__":
    main()
