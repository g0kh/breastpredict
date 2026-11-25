"""
PMJ manuscript figure army — English, Nature-styled, oncologist-facing.
Run: python PMJ/make_manuscript_figures.py
Exports PNG (300 dpi) + PDF to PMJ/figures/ and writes FIGURE_INDEX.md
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PolyCollection
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)
INDEX_PATH = Path(__file__).resolve().parent / "FIGURE_INDEX.md"

# Palettes
NATURE = ["#3C5488", "#E64B35", "#00A087", "#F39B7F", "#4DBBD5"]
LANCET = ["#00468B", "#ED0000", "#42B540", "#0099B4", "#925E9F"]
OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
TEAL_CORAL = ["#2A9D8F", "#E76F51"]
NEUTRAL = "#6C757D"
CHARCOAL = "#2B2D42"
SEQ = ["#F1FAEE", "#A8DADC", "#457B9D", "#1D3557"]

Y = 365.25
FIGURE_LOG: list[dict] = []

# ---------------------------------------------------------------------------
# Style / export
# ---------------------------------------------------------------------------

def apply_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "axes.edgecolor": CHARCOAL,
            "axes.labelcolor": CHARCOAL,
            "xtick.color": CHARCOAL,
            "ytick.color": CHARCOAL,
            "text.color": CHARCOAL,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


FEATURE_LABELS = {
    "age": "Age at diagnosis",
    "log_size": "Tumor size (log mm)",
    "ln": "Positive lymph nodes (log)",
    "grade": "Histologic grade",
    "er": "Estrogen receptor (ER)",
    "pr": "Progesterone receptor (PR)",
    "her2": "HER2 receptor",
    "post": "Postmenopausal status",
    "lobular": "Lobular histology",
    "ct": "Chemotherapy received",
    "ht": "Endocrine therapy received",
    "pr_miss": "Progesterone receptor missing",
    "her2_miss": "HER2 receptor missing",
}


def titled(ax, title: str, subtitle: str = ""):
    """Route single-axes titles through the same non-overlapping banner."""
    fig_banner(ax.figure, title, subtitle, top_room=0.68)


def _release_layout(fig):
    """Turn off constrained/tight layout so manual title margins work."""
    try:
        fig.set_layout_engine(None)
    except Exception:
        pass


def fig_banner(fig, title: str, subtitle: str = "", top_room: float = 0.72):
    """Dedicated top title band so title and subtitle never collide with plots."""
    _release_layout(fig)
    fig.subplots_adjust(top=top_room, hspace=0.35)
    # Absolute axes strip above the plotting area (taller band = clearer separation)
    tax = fig.add_axes([0.01, top_room + 0.015, 0.98, 1.0 - top_room - 0.02])
    tax.set_axis_off()
    tax.set_xlim(0, 1)
    tax.set_ylim(0, 1)
    tax.text(
        0.0,
        0.78,
        title,
        fontsize=13,
        fontweight="bold",
        color=CHARCOAL,
        ha="left",
        va="top",
        transform=tax.transAxes,
    )
    if subtitle:
        tax.text(
            0.0,
            0.28,
            subtitle,
            fontsize=9.5,
            color=NEUTRAL,
            ha="left",
            va="top",
            transform=tax.transAxes,
        )


def prepare_titled_axes(fig, top_room: float = 0.78):
    _release_layout(fig)
    fig.subplots_adjust(top=top_room)


def light_grid(ax, axis="y"):
    ax.grid(True, axis=axis, linestyle="--", linewidth=0.7, alpha=0.25, color="#CCCCCC")
    ax.set_axisbelow(True)


def _safe_savefig(fig, path: Path, **kwargs):
    """Write via temp file to avoid OneDrive/viewer locks (Errno 22)."""
    import os
    import tempfile
    import time

    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix
    fd, tmp_name = tempfile.mkstemp(suffix=suffix, dir=str(path.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        fig.savefig(tmp, **kwargs)
        for _ in range(5):
            try:
                if path.exists():
                    path.unlink()
                tmp.replace(path)
                return
            except OSError:
                time.sleep(0.4)
        # last resort: alternate name then copy
        alt = path.with_name(path.stem + "_new" + suffix)
        if alt.exists():
            alt.unlink()
        tmp.replace(alt)
        try:
            if path.exists():
                path.unlink()
            alt.replace(path)
        except OSError:
            print(f"  warn: left as {alt.name}")
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def save_fig(fig, stem: str, fig_type: str, clinical_q: str, use: str):
    png = OUT / f"{stem}.png"
    pdf = OUT / f"{stem}.pdf"
    _safe_savefig(fig, png, dpi=300, bbox_inches="tight", pad_inches=0.35)
    _safe_savefig(fig, pdf, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)
    FIGURE_LOG.append(
        {
            "file": stem,
            "type": fig_type,
            "clinical_question": clinical_q,
            "recommended_use": use,
            "png": str(png.name),
            "pdf": str(pdf.name),
        }
    )
    print(f"  saved {stem}")


def safe(name, fn):
    try:
        print(f"\n[{name}]")
        fn()
    except Exception as e:
        print(f"  SKIP {name}: {e}")
        import traceback

        traceback.print_exc()


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_all():
    db = pd.read_excel(ROOT / "database.xlsx", sheet_name="hastalar")
    features = pd.read_excel(ROOT / "sonuclar" / "tables" / "features_main.xlsx")
    analysis = pd.read_excel(ROOT / "sonuclar" / "tables" / "analysis_ids.xlsx")
    sites = pd.read_excel(ROOT / "sonuclar" / "tables" / "site_labels.xlsx")
    cv_sum = pd.read_excel(ROOT / "sonuclar" / "tables" / "cv_summary_main.xlsx")
    loo = pd.read_excel(ROOT / "sonuclar" / "tables" / "cohort_loo.xlsx")
    flow = pd.read_excel(ROOT / "sonuclar" / "tables" / "cohort_flow.xlsx")
    scores = pd.read_excel(ROOT / "sonuclar" / "external" / "tables" / "patient_scores.xlsx")
    int_ext = pd.read_excel(ROOT / "sonuclar" / "external" / "tables" / "internal_vs_external.xlsx")
    ext_met = pd.read_excel(ROOT / "sonuclar" / "external" / "tables" / "external_metrics.xlsx")
    boot = pd.read_excel(ROOT / "sonuclar" / "external" / "tables" / "external_bootstrap_5y.xlsx")
    # external raw for swimmer
    raw = pd.read_excel(
        ROOT / "sonuclar" / "External_Validasyon_Veri_Sablonu.xlsx",
        sheet_name="2_Veri_Girisi",
        header=None,
    )
    header = list(raw.iloc[1].values)
    ext = raw.iloc[3:].copy()
    ext.columns = header
    ext = ext.dropna(subset=["hasta_kodu"])
    ext = ext[ext["hasta_kodu"].astype(str).str.match(r"^EXT-", na=False)].reset_index(drop=True)
    # Table 7 for UpSet multi-site
    t7 = pd.read_csv(ROOT / "brca_metabric" / "NIHMS1520488-supplement-Supp_Table_7.txt", sep="\t")
    return {
        "db": db,
        "features": features,
        "analysis": analysis,
        "sites": sites,
        "cv_sum": cv_sum,
        "loo": loo,
        "flow": flow,
        "scores": scores,
        "int_ext": int_ext,
        "ext_met": ext_met,
        "boot": boot,
        "ext": ext,
        "t7": t7,
    }


def yes_tx(s):
    s = str(s).strip().upper()
    if s in ("", "NAN", "NA", "NO/NA", "NO", "HAYIR"):
        return "No"
    return "Yes"


def er_label(v):
    s = str(v).strip().lower()
    if s in ("pos", "positive", "pozitif"):
        return "ER+"
    if s in ("neg", "negative", "negatif"):
        return "ER-"
    return "ER unknown"


def site_bucket(cause):
    m = {
        "bone_first": "Bone",
        "visceral_first": "Visceral",
        "cns_first": "CNS",
        "other_first": "Other",
        "censor": "No distant relapse",
    }
    return m.get(str(cause), "Other")


# ---------------------------------------------------------------------------
# Fig 01 Alluvial
# ---------------------------------------------------------------------------

def fig01_alluvial(D):
    db = D["db"].copy()
    sites = D["sites"][["METABRIC.ID", "cause"]].rename(columns={"METABRIC.ID": "hasta_id"})
    m = db.merge(sites, on="hasta_id", how="left")
    m["ER"] = m["er"].map(er_label)
    m["CT"] = m["kemoterapi"].map(yes_tx)
    m["HT"] = m["hormon_tedavisi"].map(yes_tx)
    m["Tx"] = m["CT"] + " CT / " + m["HT"] + " HT"
    m["Outcome"] = m["cause"].map(site_bucket).fillna("Unknown")
    # keep complete ER
    m = m[m["ER"].isin(["ER+", "ER-"])]
    # aggregate flows ER -> Tx -> Outcome
    g1 = m.groupby(["ER", "Tx"], as_index=False).size().rename(columns={"size": "n"})
    g2 = m.groupby(["Tx", "Outcome"], as_index=False).size().rename(columns={"size": "n"})

    layers = [
        sorted(m["ER"].unique()),
        sorted(m["Tx"].unique()),
        ["Bone", "Visceral", "CNS", "Other", "No distant relapse"],
    ]
    layers[2] = [x for x in layers[2] if x in set(m["Outcome"])]

    fig, ax = plt.subplots(figsize=(11.5, 6.5), constrained_layout=True)
    x_pos = [0.0, 0.5, 1.0]
    colors = {
        "ER+": NATURE[0],
        "ER-": NATURE[1],
        "Bone": NATURE[0],
        "Visceral": NATURE[2],
        "CNS": NATURE[1],
        "Other": NATURE[3],
        "No distant relapse": NATURE[4],
    }
    # node vertical layout
    def layout(nodes, total_scale=None):
        counts = m.groupby(nodes.name if hasattr(nodes, "name") else None)
        return None

    # Manual node positions by count
    def node_positions(cats, col):
        counts = m[col].value_counts().reindex(cats).fillna(0)
        total = counts.sum()
        gap = 0.02
        usable = 1.0 - gap * (len(cats) - 1)
        heights = (counts / total * usable).values
        y0 = 0.0
        pos = {}
        for c, h in zip(cats, heights):
            pos[c] = (y0, y0 + h, h, counts[c])
            y0 += h + gap
        return pos

    p0 = node_positions(layers[0], "ER")
    p1 = node_positions(layers[1], "Tx")
    p2 = node_positions(layers[2], "Outcome")

    def draw_nodes(pos, x, label_right=False):
        for lab, (y0, y1, h, n) in pos.items():
            ax.add_patch(
                mpatches.FancyBboxPatch(
                    (x - 0.02, y0),
                    0.04,
                    max(h, 0.001),
                    boxstyle="round,pad=0.002,rounding_size=0.01",
                    facecolor=colors.get(lab, NEUTRAL),
                    edgecolor="white",
                    linewidth=0.5,
                    alpha=0.95,
                )
            )
            txt = f"{lab}\n(n={int(n)})"
            if label_right:
                ax.text(x + 0.035, (y0 + y1) / 2, txt, va="center", ha="left", fontsize=8, color=CHARCOAL)
            else:
                ax.text(x - 0.035, (y0 + y1) / 2, txt, va="center", ha="right", fontsize=8, color=CHARCOAL)

    def flows(g, left_col, right_col, pL, pR, xL, xR):
        # track remaining top within each node
        remL = {k: pL[k][1] for k in pL}
        remR = {k: pR[k][1] for k in pR}
        total = g["n"].sum()
        for _, r in g.sort_values("n", ascending=False).iterrows():
            a, b, n = r[left_col], r[right_col], r["n"]
            if a not in pL or b not in pR:
                continue
            h = n / total * (1.0 - 0.02 * (len(pL) - 1 if left_col == "ER" else len(pR) - 1))
            # better: scale to node usable height sum
            scale = sum(v[2] for v in pL.values())
            h = n / m.shape[0] * scale
            y1 = remL[a]
            y0 = y1 - h
            remL[a] = y0
            y3 = remR[b]
            y2 = y3 - h
            remR[b] = y2
            xs = np.linspace(xL + 0.02, xR - 0.02, 30)
            # bezier-like top/bottom
            top = y1 + (y3 - y1) * (0.5 - 0.5 * np.cos(np.pi * (xs - xs[0]) / (xs[-1] - xs[0])))
            bot = y0 + (y2 - y0) * (0.5 - 0.5 * np.cos(np.pi * (xs - xs[0]) / (xs[-1] - xs[0])))
            # simpler linear
            t = (xs - xs[0]) / (xs[-1] - xs[0])
            top = y1 * (1 - t) + y3 * t
            bot = y0 * (1 - t) + y2 * t
            ax.fill_between(xs, bot, top, color=colors.get(a, NATURE[0]), alpha=0.28, linewidth=0)

    draw_nodes(p0, x_pos[0], label_right=False)
    # middle: labels to the right of nodes to avoid flow occlusion
    draw_nodes(p1, x_pos[1], label_right=True)
    draw_nodes(p2, x_pos[2], label_right=True)
    flows(g1, "ER", "Tx", p0, p1, x_pos[0], x_pos[1])
    flows(g2, "Tx", "Outcome", p1, p2, x_pos[1], x_pos[2])

    ax.set_xlim(-0.25, 1.35)
    ax.set_ylim(-0.05, 1.12)
    ax.axis("off")
    ax.text(x_pos[0], 1.06, "Receptor", ha="center", fontsize=11, color=CHARCOAL)
    ax.text(x_pos[1], 1.06, "Systemic therapy", ha="center", fontsize=11, color=CHARCOAL)
    ax.text(x_pos[2], 1.06, "First distant site / status", ha="center", fontsize=11, color=CHARCOAL)
    fig_banner(
        fig,
        "Alluvial map of receptor status, therapy, and distant-relapse pattern",
        "METABRIC analysis cohort linked to first distant site (STEEP DRFI framework)",
        top_room=0.68,
    )
    save_fig(
        fig,
        "Fig01_alluvial_receptor_treatment_outcome",
        "Alluvial / Sankey",
        "How do receptor status and systemic therapy flow into first distant-relapse patterns?",
        "Main figure candidate for cohort phenotype → treatment → outcome narrative",
    )


# ---------------------------------------------------------------------------
# Fig 02 UpSet
# ---------------------------------------------------------------------------

def fig02_upset(D):
    t7 = D["t7"].copy()
    dist = t7[t7["TYPE.RELAPSE"].astype(str).str.upper() == "DISTANT"].copy()
    # flags already in columns
    cols = {
        "Bone": "BONE",
        "Liver": "LIVER",
        "Lung": "LUNG",
        "CNS": "BRAIN",
        "Pleura": "PLEURA",
        "Nodes": "LNS",
    }
    # patient-level any
    for name, c in cols.items():
        dist[c] = pd.to_numeric(dist[c], errors="coerce").fillna(0).astype(int)
    g = dist.groupby("METABRIC.ID")[list(cols.values())].max()
    g.columns = list(cols.keys())
    # only patients with at least one of these
    g = g[(g.sum(axis=1) > 0)]

    try:
        from upsetplot import from_indicators, UpSet

        # sample if huge for readability — keep all unique patients
        data = from_indicators(g.columns.tolist(), data=g.astype(bool))
        fig = plt.figure(figsize=(10, 6.2), constrained_layout=True)
        us = UpSet(
            data,
            subset_size="count",
            show_counts=True,
            sort_by="cardinality",
            facecolor=NATURE[0],
            element_size=32,
        )
        us.plot(fig=fig)
        fig_banner(
            fig,
            "Co-occurrence of distant metastatic sites among METABRIC relapses",
            "UpSet of anatomical site flags (Table 7); intersections show multi-site disease",
            top_room=0.68,
        )
        save_fig(
            fig,
            "Fig02_upset_metastasis_sites",
            "UpSet plot",
            "Which distant sites co-occur at relapse?",
            "Supports competing-risk / multi-site biology discussion",
        )
        return
    except Exception as e:
        print("  upsetplot fallback:", e)

    # Manual simple intersection bar of top combinations
    combos = g.astype(int).astype(str).agg("".join, axis=1)
    # better label
    def lab(row):
        return "+".join([c for c in g.columns if row[c] == 1]) or "None"

    labels = g.apply(lab, axis=1)
    top = labels.value_counts().head(12)
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    ax.barh(top.index[::-1], top.values[::-1], color=NATURE[0])
    light_grid(ax, "x")
    sns.despine(ax=ax)
    ax.set_xlabel("Number of patients")
    titled(ax, "Most frequent distant-site combinations", "Fallback intersection ranking (UpSet alternative)")
    save_fig(
        fig,
        "Fig02_upset_metastasis_sites",
        "UpSet / intersection bars",
        "Which distant sites co-occur at relapse?",
        "Supports multi-site biology discussion",
    )


# ---------------------------------------------------------------------------
# Fig 03 Raincloud
# ---------------------------------------------------------------------------

def fig03_raincloud(D):
    sc = D["scores"].copy()
    models = [
        ("NPI_Cox_risk5y", "NPI-Cox"),
        ("Cox_risk5y", "Cox"),
        ("Cox_ER_ayri_risk5y", "Cox ER-stratified"),
        ("RSF_risk5y", "RSF"),
    ]
    rows = []
    for col, name in models:
        for _, r in sc.iterrows():
            rows.append(
                {
                    "Model": name,
                    "Risk": float(r[col]),
                    "Status": "DRFI event" if int(r["DRFI_olay"]) == 1 else "Censored",
                }
            )
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 4.8), sharey=True, constrained_layout=True)
    for ax, model in zip(axes, [m[1] for m in models]):
        sub = df[df["Model"] == model]
        order = ["Censored", "DRFI event"]
        palette = {"Censored": TEAL_CORAL[0], "DRFI event": TEAL_CORAL[1]}
        sns.violinplot(
            data=sub,
            x="Status",
            y="Risk",
            order=order,
            palette=palette,
            ax=ax,
            inner=None,
            cut=0,
            linewidth=0.8,
            saturation=0.85,
        )
        sns.stripplot(
            data=sub,
            x="Status",
            y="Risk",
            order=order,
            palette=palette,
            ax=ax,
            size=3.2,
            alpha=0.55,
            jitter=0.18,
            edgecolor="white",
            linewidth=0.3,
        )
        # means
        for i, st in enumerate(order):
            mu = sub.loc[sub["Status"] == st, "Risk"].mean()
            ax.hlines(mu, i - 0.25, i + 0.25, colors=CHARCOAL, lw=1.6)
        ax.set_title(model, fontsize=11, fontweight="bold", color=CHARCOAL)
        ax.set_xlabel("")
        if ax is axes[0]:
            ax.set_ylabel("Predicted 5-year DRFI risk")
        else:
            ax.set_ylabel("")
        light_grid(ax)
        sns.despine(ax=ax)
        ax.set_ylim(0, 1)
    fig_banner(fig, "Raincloud of predicted 5-year DRFI risk by outcome (external cohort)", "External validation patients (n=80); horizontal bars mark group means")
    save_fig(
        fig,
        "Fig03_raincloud_risk_by_event",
        "Raincloud",
        "Do predicted risks separate patients who later experience DRFI?",
        "External discrimination illustration for Results",
    )


# ---------------------------------------------------------------------------
# Fig 04 Ridgeline
# ---------------------------------------------------------------------------

def fig04_ridgeline(D):
    sc = D["scores"].copy()
    models = [
        ("RSF_risk5y", "RSF 5y"),
        ("Cox_ER_ayri_risk5y", "Cox ER-strat. 5y"),
        ("Cox_risk5y", "Cox 5y"),
        ("NPI_Cox_risk5y", "NPI-Cox 5y"),
        ("RSF_risk10y", "RSF 10y"),
        ("Cox_risk10y", "Cox 10y"),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 6.2), constrained_layout=True)
    y_positions = np.arange(len(models))
    xmax = 1.0
    for i, (col, lab) in enumerate(models):
        x = sc[col].astype(float).dropna().values
        if len(x) < 5:
            continue
        kde = stats.gaussian_kde(x)
        xs = np.linspace(0, xmax, 200)
        dens = kde(xs)
        dens = dens / dens.max() * 0.85
        ax.fill_between(xs, i, i + dens, color=NATURE[i % len(NATURE)], alpha=0.75, linewidth=0)
        ax.plot(xs, i + dens, color=CHARCOAL, lw=0.7, alpha=0.7)
        ax.plot([np.median(x)], [i + 0.08], "o", color=CHARCOAL, ms=4)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([m[1] for m in models])
    ax.set_xlabel("Predicted DRFI risk")
    ax.set_xlim(0, 1)
    light_grid(ax, "x")
    sns.despine(ax=ax)
    titled(
        ax,
        "Ridgeline densities of predicted DRFI risk by model and horizon",
        "External cohort; dots mark medians",
    )
    save_fig(
        fig,
        "Fig04_ridgeline_risk_by_model",
        "Ridgeline (Joy) plot",
        "How do risk distributions differ across models and time horizons?",
        "Supplement or methods/results visual for model behavior",
    )


# ---------------------------------------------------------------------------
# Fig 05 SHAP hybrid
# ---------------------------------------------------------------------------

def fig05_shap(D):
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.util import Surv
    import shap

    F = D["features"].copy()
    y = Surv.from_arrays(D["analysis"]["event_drfi"].astype(bool).values, D["analysis"]["time_drfi"].astype(float).values)
    feat_names = list(F.columns)
    im = IterativeImputer(max_iter=10, random_state=42).fit(F)
    X = im.transform(F)
    # clip binaries
    for j, c in enumerate(feat_names):
        if c in ("pr", "her2", "post", "lobular", "ct", "ht", "pr_miss", "her2_miss", "er"):
            X[:, j] = np.clip(np.round(X[:, j]), 0, 1)
        if c == "grade":
            X[:, j] = np.clip(np.round(X[:, j]), 1, 3)
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    cox = CoxPHSurvivalAnalysis(alpha=1.0).fit(Xs, y)
    # LinearSHAP-like: contribution = coef * (x - mean); mean already 0 after scaling
    coef = cox.coef_
    # sample for beeswarm
    rng = np.random.RandomState(42)
    idx = rng.choice(len(Xs), size=min(800, len(Xs)), replace=False)
    Xs_s = Xs[idx]
    shap_vals = Xs_s * coef.reshape(1, -1)
    mean_abs = np.abs(shap_vals).mean(axis=0)
    order = np.argsort(mean_abs)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.2), gridspec_kw={"width_ratios": [1.45, 1]})
    # beeswarm
    ax = axes[0]
    for yi, j in enumerate(order):
        vals = shap_vals[:, j]
        feat = Xs_s[:, j]
        # color by feature value
        fmin, fmax = np.percentile(feat, [5, 95])
        f = np.clip((feat - fmin) / (fmax - fmin + 1e-9), 0, 1)
        # jitter
        jitter = rng.normal(0, 0.08, size=len(vals))
        cmap = plt.cm.colors.LinearSegmentedColormap.from_list("seq", SEQ)
        ax.scatter(vals, np.full_like(vals, yi) + jitter, c=f, cmap=cmap, s=8, alpha=0.55, linewidths=0)
    nice = [FEATURE_LABELS.get(feat_names[j], feat_names[j]) for j in order]
    ax.axvline(0, color=NEUTRAL, lw=0.8, ls="--")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(nice)
    ax.set_xlabel("SHAP value (impact on log partial hazard)")
    light_grid(ax, "x")
    sns.despine(ax=ax)
    ax.set_title("Beeswarm", loc="left", fontsize=11, fontweight="bold", color=CHARCOAL, pad=6)
    # mean abs bars
    ax2 = axes[1]
    ax2.barh(range(len(order)), mean_abs[order], color=NATURE[0])
    ax2.set_yticks(range(len(order)))
    ax2.set_yticklabels([])  # avoid duplicate overlapping labels between panels
    ax2.set_xlabel("Mean |SHAP|")
    light_grid(ax2, "x")
    sns.despine(ax=ax2)
    ax2.set_title("Mean absolute impact", loc="left", fontsize=11, fontweight="bold", color=CHARCOAL, pad=6)
    fig_banner(
        fig,
        "Cox DRFI model explanation: SHAP beeswarm and mean absolute impact",
        "Development cohort; contributions to the Cox linear predictor on standardized features",
        top_room=0.68,
    )
    # colorbar on the right of beeswarm only
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[0], fraction=0.035, pad=0.02)
    cbar.set_label("Feature value (low → high)", fontsize=9)
    save_fig(
        fig,
        "Fig05_shap_beeswarm_mean",
        "SHAP beeswarm + mean impact hybrid",
        "Which clinical features drive predicted DRFI hazard?",
        "Key explainability figure for oncologist interpretation",
    )


# ---------------------------------------------------------------------------
# Fig 06 Calibration + hist + DCA
# ---------------------------------------------------------------------------

def event_by_horizon(time, event, t):
    lab = np.full(len(time), np.nan)
    ev = event.astype(bool)
    lab[(time <= t) & ev] = 1
    lab[time > t] = 0
    return lab


def fig06_calib_dca(D):
    sc = D["scores"].copy()
    time = sc["DRFI_sure_gun"].astype(float).values
    event = sc["DRFI_olay"].astype(int).values
    t5 = 5 * Y
    lab = event_by_horizon(time, event, t5)
    models = {
        "RSF": sc["RSF_risk5y"].astype(float).values,
        "Cox": sc["Cox_risk5y"].astype(float).values,
        "NPI-Cox": sc["NPI_Cox_risk5y"].astype(float).values,
    }
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.4))

    # calibration
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "--", color="#AAAAAA", lw=1)
    for i, (name, p) in enumerate(models.items()):
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
        ax.plot(g["pred"], g["obs"], "o-", color=LANCET[i], label=name, ms=6, lw=1.8)
    ax.set_xlabel("Predicted 5y DRFI risk")
    ax.set_ylabel("Observed event rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    light_grid(ax)
    sns.despine(ax=ax)
    ax.set_title("Calibration", loc="left", fontsize=12, fontweight="bold", color=CHARCOAL)

    # marginal hist of RSF risk by outcome
    ax = axes[1]
    p = models["RSF"]
    mask = np.isfinite(lab) & np.isfinite(p)
    ax.hist(p[mask & (lab == 0)], bins=12, alpha=0.65, color=TEAL_CORAL[0], label="Censored", density=True)
    ax.hist(p[mask & (lab == 1)], bins=10, alpha=0.7, color=TEAL_CORAL[1], label="DRFI by 5y", density=True)
    ax.set_xlabel("RSF predicted 5y risk")
    ax.set_ylabel("Density")
    ax.legend(frameon=False, fontsize=8)
    light_grid(ax)
    sns.despine(ax=ax)
    ax.set_title("Risk distribution", loc="left", fontsize=12, fontweight="bold", color=CHARCOAL)

    # DCA
    ax = axes[2]
    thresholds = np.arange(0.05, 0.55, 0.01)
    mask0 = np.isfinite(lab)
    y = lab[mask0].astype(int)
    prev = y.mean() if len(y) else 0
    nb_all = prev - (1 - prev) * (thresholds / (1 - thresholds))
    ax.plot(thresholds, nb_all, "--", color=NEUTRAL, label="Treat all", lw=1.2)
    ax.plot(thresholds, np.zeros_like(thresholds), ":", color=NEUTRAL, label="Treat none", lw=1.2)
    for i, (name, p) in enumerate(models.items()):
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
        ax.plot(thresholds, nbs, color=LANCET[i], lw=2, label=name)
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    light_grid(ax)
    sns.despine(ax=ax)
    ax.set_title("Decision curve", loc="left", fontsize=12, fontweight="bold", color=CHARCOAL, pad=8)
    # Capture handles before placing legend on a dedicated axes (outside plot)
    dca_handles, dca_labels = ax.get_legend_handles_labels()

    fig_banner(
        fig,
        "External performance triad: calibration, risk separation, and clinical net benefit",
        "External cohort (n=80); 5-year DRFI horizon; Lancet clinical palette",
        top_room=0.68,
    )
    # Shrink panels; place DCA legend in a dedicated upper-right axes (never on curves)
    fig.subplots_adjust(right=0.78, wspace=0.32)
    leg_ax = fig.add_axes([0.80, 0.42, 0.18, 0.28])
    leg_ax.set_axis_off()
    leg_ax.legend(
        dca_handles,
        dca_labels,
        frameon=False,
        fontsize=7.5,
        loc="upper left",
        handlelength=1.6,
        borderaxespad=0.0,
    )
    save_fig(
        fig,
        "Fig06_calibration_hist_dca",
        "Calibration + marginal histogram + DCA",
        "Are predicted probabilities reliable and clinically useful at decision thresholds?",
        "Primary performance figure for external validation section",
    )


# ---------------------------------------------------------------------------
# Fig 07 Dumbbell
# ---------------------------------------------------------------------------

def fig07_dumbbell(D):
    d = D["int_ext"].copy()
    boot = D["boot"].set_index("model")
    # align names
    d = d.sort_values("external_AUC5y")
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    y = np.arange(len(d))
    for i, (_, r) in enumerate(d.iterrows()):
        ax.plot([r["internal_AUC5y"], r["external_AUC5y"]], [i, i], color="#CCCCCC", lw=2.5, zorder=1)
        ax.scatter(r["internal_AUC5y"], i, s=70, color=NATURE[0], zorder=3, label="Internal CV" if i == 0 else "")
        ax.scatter(r["external_AUC5y"], i, s=70, color=NATURE[1], zorder=3, label="External" if i == 0 else "")
        m = r["model"]
        if m in boot.index and np.isfinite(boot.loc[m, "AUC_lo"]):
            ax.plot([boot.loc[m, "AUC_lo"], boot.loc[m, "AUC_hi"]], [i, i], color=NATURE[1], lw=1.5, alpha=0.7, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(d["model"])
    ax.axvline(0.5, color=NEUTRAL, ls="--", lw=0.8)
    ax.set_xlabel("5-year time-dependent AUC")
    ax.set_xlim(0.45, 0.95)
    light_grid(ax, "x")
    sns.despine(ax=ax)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    # readable model names
    name_map = {
        "RSF": "Random survival forest",
        "Cox_ER_ayri": "Cox (ER-stratified)",
        "NPI_Cox": "NPI-Cox baseline",
        "Cox": "Cox proportional hazards",
    }
    ax.set_yticklabels([name_map.get(m, m) for m in d["model"]])
    titled(
        ax,
        "Internal versus external discrimination (dumbbell)",
        "Points: point estimates; whiskers: external bootstrap 95% CI",
    )
    save_fig(
        fig,
        "Fig07_dumbbell_internal_external_auc",
        "Dumbbell plot",
        "Does discrimination transport from METABRIC to the hospital cohort?",
        "Headline external-validation comparison figure",
    )


# ---------------------------------------------------------------------------
# Fig 08 Swimmer
# ---------------------------------------------------------------------------

def fig08_swimmer(D):
    ext = D["ext"].copy()
    ext["time"] = pd.to_numeric(ext["DRFI_sure_gun"], errors="coerce")
    ext["event"] = pd.to_numeric(ext["DRFI_olay"], errors="coerce").fillna(0).astype(int)
    ext["met_time"] = pd.to_numeric(ext["ilk_uzak_metastaz_gun"], errors="coerce")
    ext["site"] = ext["ilk_uzak_bolge"].astype(str).replace({"nan": ""})
    ext = ext.dropna(subset=["time"]).sort_values("time", ascending=True).reset_index(drop=True)
    # all 80 is ok if spaced
    n = len(ext)
    fig, ax = plt.subplots(figsize=(10.5, max(6.5, n * 0.11)), constrained_layout=True)
    site_color = {
        "kemik": OKABE[0],
        "viseral": OKABE[1],
        "beyin_sss": OKABE[2],
        "kemik+viseral": OKABE[3],
        "diger": OKABE[4],
    }
    for i, r in ext.iterrows():
        col = TEAL_CORAL[1] if r["event"] == 1 else TEAL_CORAL[0]
        ax.barh(i, r["time"] / Y, left=0, height=0.65, color=col, alpha=0.85, edgecolor="white", linewidth=0.3)
        if pd.notna(r["met_time"]) and r["met_time"] > 0:
            sc = site_color.get(str(r["site"]).lower(), CHARCOAL)
            ax.scatter(r["met_time"] / Y, i, s=28, color=sc, zorder=3, edgecolors="white", linewidths=0.4)
    ax.set_yticks(range(n))
    ax.set_yticklabels(ext["hasta_kodu"].astype(str).tolist(), fontsize=7)
    ax.set_xlabel("Years from diagnosis")
    light_grid(ax, "x")
    sns.despine(ax=ax)
    # legends
    handles = [
        mpatches.Patch(color=TEAL_CORAL[0], label="Censored follow-up"),
        mpatches.Patch(color=TEAL_CORAL[1], label="DRFI event follow-up"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=OKABE[0], markersize=7, label="First distant site"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=8)
    titled(
        ax,
        "Swimmer plot of follow-up and first distant metastasis (external cohort)",
        "Bars show observation time; markers indicate recorded first distant site",
    )
    save_fig(
        fig,
        "Fig08_swimmer_external",
        "Swimmer plot",
        "What does individual follow-up and timing of first metastasis look like externally?",
        "Patient-level narrative figure for external cohort",
    )


# ---------------------------------------------------------------------------
# Fig 09 CIF competing risks
# ---------------------------------------------------------------------------

def fig09_cif(D):
    sites = D["sites"].copy()
    # Aalen-Johansen-like via lifelines or manual CIF for cause-specific
    try:
        from lifelines import AalenJohansenFitter
    except Exception:
        AalenJohansenFitter = None

    fig, ax = plt.subplots(figsize=(9.2, 5.4), constrained_layout=True)
    causes = [("bone_first", "Bone-first", NATURE[0]), ("visceral_first", "Visceral-first", NATURE[2]), ("cns_first", "CNS-first", NATURE[1])]
    t_grid = np.linspace(0, 15 * Y, 200)

    if AalenJohansenFitter is not None:
        # map causes to integer event codes; censor=0
        df = sites[["etime", "cause"]].copy()
        code = {"censor": 0, "bone_first": 1, "visceral_first": 2, "cns_first": 3, "other_first": 4}
        df["ev"] = df["cause"].map(code).fillna(4).astype(int)
        for cause_key, lab, col in causes:
            aj = AalenJohansenFitter(calculate_variance=False)
            try:
                aj.fit(df["etime"], df["ev"], event_of_interest=code[cause_key])
                ax.plot(aj.cumulative_density_.index / Y, aj.cumulative_density_.iloc[:, 0], color=col, lw=2.2, label=lab)
            except Exception as e:
                print("  AJ fail", cause_key, e)
    else:
        # crude cumulative incidence treating others as censor
        for cause_key, lab, col in causes:
            ev = (sites["cause"] == cause_key).astype(int).values
            tm = sites["etime"].astype(float).values
            order = np.argsort(tm)
            tm, ev = tm[order], ev[order]
            n = len(tm)
            at_risk = n
            cif = 0.0
            xs, ys = [0.0], [0.0]
            for t, e in zip(tm, ev):
                if at_risk <= 0:
                    break
                if e == 1:
                    cif += (1 / at_risk) * (1)  # not proper AJ but illustrative
                    # better: use 1/at_risk for event hazard approx ignoring survival
                at_risk -= 1
                if e == 1:
                    xs.append(t / Y)
                    ys.append(cif)
            ax.step(xs, ys, where="post", color=col, lw=2.2, label=lab)

    ax.set_xlabel("Years from diagnosis")
    ax.set_ylabel("Cumulative incidence")
    ax.set_xlim(0, 15)
    ax.legend(frameon=False, loc="upper left")
    light_grid(ax)
    sns.despine(ax=ax)
    titled(
        ax,
        "Competing-risk cumulative incidence of first distant site",
        "METABRIC analysis cohort; bone-, visceral-, and CNS-first distant relapse",
    )
    save_fig(
        fig,
        "Fig09_cif_competing_sites",
        "Competing-risk CIF",
        "When do bone, visceral, and CNS first metastases accrue over time?",
        "Core oncology time-to-site figure",
    )


# ---------------------------------------------------------------------------
# Fig 10 Forest LOO
# ---------------------------------------------------------------------------

def fig10_forest(D):
    loo = D["loo"].copy()
    fig, ax = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
    models = sorted(loo["model"].unique())
    cohorts = sorted(loo["holdout_cohort"].unique())
    y = 0
    yticks, ylabels = [], []
    for c in cohorts:
        for i, m in enumerate(models):
            sub = loo[(loo["holdout_cohort"] == c) & (loo["model"] == m)]
            if sub.empty:
                continue
            auc = float(sub["AUC5y"].iloc[0])
            ax.plot(auc, y, "o", color=NATURE[i % len(NATURE)], ms=8)
            ax.hlines(y, 0.5, auc, color="#DDDDDD", lw=1)
            yticks.append(y)
            ylabels.append(f"Cohort {c} · {m}")
            y += 1
        y += 0.4
    ax.axvline(0.5, color=NEUTRAL, ls="--", lw=0.8)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8.5)
    ax.set_xlabel("5-year time-dependent AUC (holdout cohort)")
    ax.set_xlim(0.45, 0.85)
    light_grid(ax, "x")
    sns.despine(ax=ax)
    titled(
        ax,
        "Cohort leave-one-out transportability (forest-style)",
        "Each point is discrimination when one METABRIC cohort is held out",
    )
    save_fig(
        fig,
        "Fig10_forest_cohort_loo",
        "Forest / Cleveland-style LOO",
        "Is performance stable across METABRIC source cohorts?",
        "Internal-external robustness figure",
    )


# ---------------------------------------------------------------------------
# Fig 11 tdAUC trajectories
# ---------------------------------------------------------------------------

def fig11_tdauc(D):
    m = D["ext_met"].copy()
    years = [2, 3, 4, 5, 10]
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for i, (_, r) in enumerate(m.iterrows()):
        xs, ys = [], []
        for yr in years:
            col = f"AUC{yr}y"
            if col in r and pd.notna(r[col]):
                xs.append(yr)
                ys.append(r[col])
        ax.plot(xs, ys, "o-", color=NATURE[i % len(NATURE)], lw=2, ms=6, label=r["model"])
    ax.axhline(0.5, color=NEUTRAL, ls="--", lw=0.8)
    ax.set_xlabel("Horizon (years)")
    ax.set_ylabel("Time-dependent AUC")
    ax.set_ylim(0.45, 0.95)
    ax.legend(frameon=False, loc="best")
    light_grid(ax)
    sns.despine(ax=ax)
    titled(
        ax,
        "External time-dependent discrimination across horizons",
        "Hospital cohort; missing 1-year AUC reflects no events before year 1",
    )
    save_fig(
        fig,
        "Fig11_tdauc_trajectories",
        "Time-dependent AUC trajectories",
        "How does discrimination change from early to late follow-up?",
        "External validation time-dynamics figure",
    )


# ---------------------------------------------------------------------------
# Fig 12 Heatmap receptor x site
# ---------------------------------------------------------------------------

def fig12_heatmap(D):
    db = D["db"][["hasta_id", "er", "pr", "her2"]].copy()
    sites = D["sites"][["METABRIC.ID", "cause"]].rename(columns={"METABRIC.ID": "hasta_id"})
    m = db.merge(sites, on="hasta_id", how="inner")
    m = m[m["cause"].isin(["bone_first", "visceral_first", "cns_first", "other_first"])]
    m["ER"] = m["er"].map(lambda x: "ER+" if str(x).lower() in ("pos", "positive") else ("ER-" if str(x).lower() in ("neg", "negative") else np.nan))
    m["PR"] = m["pr"].map(lambda x: "PR+" if str(x) == "Positive" else ("PR-" if str(x) == "Negative" else np.nan))
    m["HER2"] = m["her2"].map(lambda x: "HER2+" if str(x) == "Positive" else ("HER2-" if str(x) == "Negative" else np.nan))
    m["Site"] = m["cause"].map(site_bucket)
    m["Receptor phenotype"] = m["ER"].fillna("ER?") + " / " + m["PR"].fillna("PR?") + " / " + m["HER2"].fillna("HER2?")
    # simplify phenotypes
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
    ct = pd.crosstab(m["Phenotype"], m["Site"], normalize="index") * 100
    order_p = ["ER+/HER2-", "HER2+", "TNBC-like", "Other"]
    order_s = ["Bone", "Visceral", "CNS", "Other"]
    ct = ct.reindex([p for p in order_p if p in ct.index])
    ct = ct[[c for c in order_s if c in ct.columns]]
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    sns.heatmap(ct, annot=True, fmt=".1f", cmap="mako", ax=ax, cbar_kws={"label": "% within phenotype"}, linewidths=0.5, linecolor="white")
    ax.set_xlabel("First distant site")
    ax.set_ylabel("Receptor phenotype")
    titled(
        ax,
        "Receptor phenotype versus first distant metastatic site",
        "Row-normalized percentages among patients with a classified first distant site",
    )
    save_fig(
        fig,
        "Fig12_heatmap_receptor_site",
        "Heatmap",
        "Do receptor phenotypes prefer different first metastatic niches?",
        "Biology/clinical pattern figure",
    )


# ---------------------------------------------------------------------------
# Fig 13 KM by risk tertile (external RSF)
# ---------------------------------------------------------------------------

def fig13_km_tertile(D):
    from lifelines import KaplanMeierFitter

    sc = D["scores"].copy()
    sc["risk"] = sc["RSF_risk5y"].astype(float)
    sc["time"] = sc["DRFI_sure_gun"].astype(float) / Y
    sc["event"] = sc["DRFI_olay"].astype(int)
    sc = sc.dropna(subset=["risk", "time"])
    sc["tertile"] = pd.qcut(sc["risk"], 3, labels=["Low risk", "Intermediate", "High risk"])
    fig, ax = plt.subplots(figsize=(8.8, 5.4), constrained_layout=True)
    kmf = KaplanMeierFitter()
    colors = [NATURE[2], NATURE[4], NATURE[1]]
    for col, lab in zip(colors, ["Low risk", "Intermediate", "High risk"]):
        sub = sc[sc["tertile"] == lab]
        if len(sub) < 5:
            continue
        kmf.fit(sub["time"], sub["event"], label=f"{lab} (n={len(sub)})")
        kmf.plot_survival_function(ax=ax, color=col, ci_show=True, linewidth=2.2)
    ax.set_xlabel("Years from diagnosis")
    ax.set_ylabel("DRFI-free probability")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="lower left")
    light_grid(ax)
    sns.despine(ax=ax)
    titled(
        ax,
        "DRFI-free survival by external RSF predicted-risk tertile",
        "Kaplan–Meier curves on the hospital cohort (illustrative; small n)",
    )
    save_fig(
        fig,
        "Fig13_km_by_risk_tertile",
        "Kaplan–Meier by risk tertile",
        "Do model-defined risk groups separate event-free survival externally?",
        "Clinical risk-stratification figure",
    )


# ---------------------------------------------------------------------------
# Fig 14 Cohort flow
# ---------------------------------------------------------------------------

def fig14_flow(D):
    flow = D["flow"].copy()
    fig, ax = plt.subplots(figsize=(10.5, 3.8), constrained_layout=True)
    xs = np.arange(len(flow))
    ns = flow["n"].astype(float).values
    ax.plot(xs, ns, "-o", color=NATURE[0], lw=2.2, ms=8)
    for i, r in flow.iterrows():
        ax.text(i, r["n"] + 25, f"{int(r['n'])}", ha="center", fontsize=9, color=CHARCOAL)
        if pd.notna(r.get("dropped")) and r["dropped"] > 0:
            ax.text(i, r["n"] - 90, f"−{int(r['dropped'])}", ha="center", fontsize=8, color=NATURE[1])
    labels = [
        "Raw Table 5",
        "Follow-up > 0",
        "Death known",
        "Exclude stage 4",
        "Exclude BENIGN/\nPHYL/DCIS",
    ]
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Patients")
    light_grid(ax)
    sns.despine(ax=ax)
    titled(
        ax,
        "Analysis cohort construction",
        "CONSORT-style attrition from METABRIC Supplement Table 5 to the DRFI modeling set",
    )
    save_fig(
        fig,
        "Fig14_cohort_flow",
        "Cohort flow",
        "How was the analytic cohort derived from METABRIC?",
        "Methods / Figure 1 candidate",
    )


# ---------------------------------------------------------------------------
# Fig 15 Features by site
# ---------------------------------------------------------------------------

def fig15_features_by_site(D):
    F = D["features"].copy()
    sites = D["sites"].copy()
    df = F.copy()
    df["cause"] = sites["cause"].values
    df = df[df["cause"].isin(["bone_first", "visceral_first", "cns_first"])]
    df["Site"] = df["cause"].map(site_bucket)
    df["Tumor size (mm)"] = np.exp(df["log_size"])
    df["Positive LN"] = np.expm1(df["ln"])
    df["Age (years)"] = df["age"]
    df["Grade"] = df["grade"]
    feats = ["Age (years)", "Tumor size (mm)", "Positive LN", "Grade"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
    order = ["Bone", "Visceral", "CNS"]
    pal = {"Bone": NATURE[0], "Visceral": NATURE[2], "CNS": NATURE[1]}
    for ax, feat in zip(axes.ravel(), feats):
        sns.violinplot(
            data=df,
            x="Site",
            y=feat,
            order=order,
            palette=pal,
            ax=ax,
            cut=0,
            inner="quartile",
            linewidth=0.8,
        )
        sns.stripplot(
            data=df,
            x="Site",
            y=feat,
            order=order,
            palette=pal,
            ax=ax,
            size=2.2,
            alpha=0.25,
            jitter=0.2,
            dodge=False,
        )
        ax.set_xlabel("")
        light_grid(ax)
        sns.despine(ax=ax)
        ax.set_title(feat, loc="left", fontsize=11, fontweight="bold", color=CHARCOAL)
    fig_banner(fig, "Clinical features at diagnosis by first distant metastatic site", "METABRIC patients with bone-, visceral-, or CNS-first distant relapse")
    save_fig(
        fig,
        "Fig15_features_by_site",
        "Violin/raincloud grid by site",
        "Do baseline clinicopathologic features differ by first metastatic niche?",
        "Exploratory phenotype–site association panel",
    )


# ---------------------------------------------------------------------------
# Fig 16 Lollipop metrics
# ---------------------------------------------------------------------------

def fig16_lollipop(D):
    cv = D["cv_sum"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    # Uno C
    ax = axes[0]
    d = cv.sort_values("UnoC_10y")
    y = np.arange(len(d))
    ax.hlines(y, 0.5, d["UnoC_10y"], color="#DDDDDD", lw=2)
    ax.plot(d["UnoC_10y"], y, "o", color=NATURE[0], ms=9)
    if "UnoC_10y_lo" in d.columns:
        for i, (_, r) in enumerate(d.iterrows()):
            ax.plot([r["UnoC_10y_lo"], r["UnoC_10y_hi"]], [i, i], color=NATURE[0], lw=1.5, alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(d["model"])
    ax.set_xlabel("Uno C-index (10y)")
    ax.set_xlim(0.55, 0.75)
    light_grid(ax, "x")
    sns.despine(ax=ax)
    ax.set_title("Discrimination", loc="left", fontsize=12, fontweight="bold", color=CHARCOAL)

    ax = axes[1]
    d2 = cv.sort_values("Brier5y", ascending=False)
    y = np.arange(len(d2))
    ax.hlines(y, d2["Brier5y"], 0.22, color="#DDDDDD", lw=2)
    ax.plot(d2["Brier5y"], y, "o", color=NATURE[1], ms=9)
    ax.set_yticks(y)
    ax.set_yticklabels(d2["model"])
    ax.set_xlabel("IPCW Brier score (5y; lower better)")
    light_grid(ax, "x")
    sns.despine(ax=ax)
    ax.set_title("Calibration / accuracy of probs", loc="left", fontsize=12, fontweight="bold", color=CHARCOAL)

    fig_banner(fig, "Internal cross-validated model comparison (Cleveland lollipop)", "METABRIC 5-fold CV summary with bootstrap intervals where available")
    save_fig(
        fig,
        "Fig16_lollipop_metrics",
        "Cleveland lollipop",
        "Which model balances discrimination and probability accuracy internally?",
        "Model-selection summary figure",
    )


# ---------------------------------------------------------------------------
# Fig 17 Treatment patterns
# ---------------------------------------------------------------------------

def fig17_treatment(D):
    db = D["db"].copy()
    db["ER"] = db["er"].map(er_label)
    db = db[db["ER"].isin(["ER+", "ER-"])]
    db["CT"] = db["kemoterapi"].map(yes_tx)
    db["HT"] = db["hormon_tedavisi"].map(yes_tx)
    ct = pd.crosstab([db["ER"], db["CT"]], db["HT"])
    # grouped bars
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True, constrained_layout=True)
    for ax, er in zip(axes, ["ER+", "ER-"]):
        sub = db[db["ER"] == er]
        tab = pd.crosstab(sub["CT"], sub["HT"])
        tab = tab.reindex(index=["No", "Yes"], columns=["No", "Yes"]).fillna(0)
        x = np.arange(2)
        w = 0.35
        ax.bar(x - w / 2, tab.loc[:, "No"].values if "No" in tab.columns else [0, 0], w, label="HT No", color=NATURE[0])
        ax.bar(x + w / 2, tab.loc[:, "Yes"].values if "Yes" in tab.columns else [0, 0], w, label="HT Yes", color=NATURE[2])
        # actually columns are HT
        vals_no = [tab.loc[r, "No"] if "No" in tab.columns and r in tab.index else 0 for r in ["No", "Yes"]]
        vals_yes = [tab.loc[r, "Yes"] if "Yes" in tab.columns and r in tab.index else 0 for r in ["No", "Yes"]]
        ax.cla()
        ax.bar(x - w / 2, vals_no, w, label="No endocrine therapy", color=NATURE[0])
        ax.bar(x + w / 2, vals_yes, w, label="Endocrine therapy", color=NATURE[2])
        ax.set_xticks(x)
        ax.set_xticklabels(["No CT", "CT"])
        ax.set_title(er, loc="left", fontsize=12, fontweight="bold", color=CHARCOAL)
        light_grid(ax)
        sns.despine(ax=ax)
        if ax is axes[0]:
            ax.set_ylabel("Patients")
            ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig_banner(fig, "Systemic therapy patterns by ER status in METABRIC", "Counts of chemotherapy × endocrine therapy within ER-positive and ER-negative disease")
    save_fig(
        fig,
        "Fig17_treatment_patterns",
        "Grouped treatment pattern bars",
        "How were chemotherapy and endocrine therapy used across ER groups?",
        "Clinical context / confounding-by-indication panel",
    )


# ---------------------------------------------------------------------------
# Fig 18 Training vs center feature comparison
# ---------------------------------------------------------------------------

def fig18_feature_compare(D):
    """Demographic + pathologic features: METABRIC development vs local center."""
    F = D["features"].copy()
    ext = D["ext"].copy()

    # Development continuous / binary
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

    def yn(series, yes=("evet", "yes", "1", "true")):
        s = series.astype(str).str.strip().str.lower()
        out = np.where(s.isin(yes), 1.0, np.where(s.isin(("hayir", "hayır", "no", "0", "false")), 0.0, np.nan))
        return out

    def pn(series):
        s = series.astype(str).str.strip().str.lower()
        return np.where(
            s.isin(("pozitif", "positive", "pos")),
            1.0,
            np.where(s.isin(("negatif", "negative", "neg")), 0.0, np.nan),
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

    fig = plt.figure(figsize=(12.5, 9.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1.0], hspace=0.50, wspace=0.35)

    # Top: continuous demographic/pathologic distributions
    for i, col in enumerate(continuous):
        ax = fig.add_subplot(gs[0, i])
        a = dev[col].dropna().astype(float).values
        b = loc[col].dropna().astype(float).values
        sns.kdeplot(a, ax=ax, color=NATURE[0], fill=True, alpha=0.35, linewidth=1.8, label="METABRIC (training)")
        sns.kdeplot(b, ax=ax, color=TEAL_CORAL[1], fill=True, alpha=0.35, linewidth=1.8, label="Local center")
        ax.axvline(np.nanmedian(a), color=NATURE[0], ls="--", lw=1.2)
        ax.axvline(np.nanmedian(b), color=TEAL_CORAL[1], ls="--", lw=1.2)
        ax.set_title(col, loc="left", fontsize=10, fontweight="bold", color=CHARCOAL, pad=4)
        ax.set_ylabel("Density" if i == 0 else "")
        ax.set_xlabel(col)
        light_grid(ax)
        sns.despine(ax=ax)
        if i == 0:
            ax.legend(frameon=False, fontsize=7.5, loc="upper right")

    # Bottom: binary pathologic / clinical rates
    axb = fig.add_subplot(gs[1, :])
    x = np.arange(len(binary))
    w = 0.36
    dev_rate = [np.nanmean(dev[c]) for c in binary]
    loc_rate = [np.nanmean(loc[c]) for c in binary]
    axb.bar(x - w / 2, dev_rate, w, color=NATURE[0], label="METABRIC (training)", edgecolor="white")
    axb.bar(x + w / 2, loc_rate, w, color=TEAL_CORAL[1], label="Local center", edgecolor="white")
    for i, (d, l) in enumerate(zip(dev_rate, loc_rate)):
        axb.text(i - w / 2, d + 0.02, f"{d:.0%}", ha="center", fontsize=8, color=CHARCOAL)
        axb.text(i + w / 2, l + 0.02, f"{l:.0%}", ha="center", fontsize=8, color=CHARCOAL)
    axb.set_xticks(x)
    axb.set_xticklabels(binary, fontsize=9)
    axb.set_ylim(0, 1.15)
    axb.set_ylabel("Proportion")
    axb.legend(frameon=False, fontsize=9, loc="upper right")
    light_grid(axb)
    sns.despine(ax=axb)
    axb.set_title(
        "Pathologic and clinical categorical features",
        loc="left",
        fontsize=11,
        fontweight="bold",
        color=CHARCOAL,
        pad=6,
    )

    fig_banner(
        fig,
        "Feature set comparison: METABRIC training cohort versus local center",
        f"Top: demographic and continuous pathologic features. Bottom: pathologic/clinical proportions. Training n={len(dev)}; local n={len(loc)}; dashed lines = medians",
        top_room=0.70,
    )
    save_fig(
        fig,
        "Fig18_features_training_vs_center",
        "Training vs center feature comparison",
        "How do demographic and pathologic features differ between METABRIC and the local center?",
        "Methods/Results cohort comparability figure for oncologists",
    )


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

def write_index():
    lines = [
        "# PMJ figure index",
        "",
        "All figures are English-labeled, exported as 300 dpi PNG and vector PDF in `PMJ/figures/`.",
        "",
        "| File | Type | Clinical question | Suggested use |",
        "|---|---|---|---|",
    ]
    for r in FIGURE_LOG:
        lines.append(
            f"| `{r['file']}` | {r['type']} | {r['clinical_question']} | {r['recommended_use']} |"
        )
    lines += [
        "",
        "## Notes for selection",
        "- Prefer **Fig14** as methods flow, **Fig18** for cohort comparability, **Fig07/Fig06** for external validation, **Fig05** for explainability, **Fig09** for site biology.",
        "- External n=80 / 23 events: interpret Fig03/08/13 as supportive, not definitive.",
        "- Style: Nature/Lancet/Okabe/Teal–Coral palettes; charcoal axes; no top/right spines.",
        "- Titles use separated title + subtitle blocks (no overlapping text).",
    ]
    INDEX_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {INDEX_PATH} ({len(FIGURE_LOG)} figures)")


def main():
    FIGURE_LOG.clear()
    apply_style()
    print("Loading data...")
    D = load_all()
    safe("Fig01", lambda: fig01_alluvial(D))
    safe("Fig02", lambda: fig02_upset(D))
    safe("Fig03", lambda: fig03_raincloud(D))
    safe("Fig04", lambda: fig04_ridgeline(D))
    safe("Fig05", lambda: fig05_shap(D))
    safe("Fig06", lambda: fig06_calib_dca(D))
    safe("Fig07", lambda: fig07_dumbbell(D))
    safe("Fig08", lambda: fig08_swimmer(D))
    safe("Fig09", lambda: fig09_cif(D))
    safe("Fig10", lambda: fig10_forest(D))
    safe("Fig11", lambda: fig11_tdauc(D))
    safe("Fig12", lambda: fig12_heatmap(D))
    safe("Fig13", lambda: fig13_km_tertile(D))
    safe("Fig14", lambda: fig14_flow(D))
    safe("Fig15", lambda: fig15_features_by_site(D))
    safe("Fig16", lambda: fig16_lollipop(D))
    safe("Fig17", lambda: fig17_treatment(D))
    safe("Fig18", lambda: fig18_feature_compare(D))
    write_index()


if __name__ == "__main__":
    main()
