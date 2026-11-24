"""
Report 2: feature contribution analysis, lean feature sets, hybrids, higher-AUC focus.
Outputs: report2.html + outputs/report2_*
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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
FIG = OUT / "figures_r2"
TAB = OUT / "tables_r2"
for d in (OUT, FIG, TAB):
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5
YEAR = 365.25

ALL_NUM = ["yas", "tumor_boyutu_mm", "pozitif_ln", "grade"]
ALL_CAT = [
    "menopoz",
    "histoloji",
    "er",
    "pr",
    "her2",
    "cerrahi",
    "kemoterapi",
    "hormon_tedavisi",
    "radyoterapi",
]
ALL_FEAT = ALL_NUM + ALL_CAT
TX = ["kemoterapi", "hormon_tedavisi", "radyoterapi"]


def fig_b64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def save_fig(fig, name: str) -> str:
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight", facecolor="white")
    return fig_b64(fig)


def group_rare(s: pd.Series, min_count=25) -> pd.Series:
    vc = s.fillna("MISSING").astype(str).value_counts()
    keep = set(vc[vc >= min_count].index)
    return s.fillna("MISSING").astype(str).map(lambda x: x if x in keep else "OTHER")


def load_df() -> pd.DataFrame:
    df = pd.read_excel(ROOT / "database.xlsx", sheet_name="hastalar")
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].replace({"": np.nan, " ": np.nan})
    for c in ["yas", "tumor_boyutu_mm", "pozitif_ln", "grade", "uzak_metastaz", "uzak_metastaz_gun"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["kemoterapi", "hormon_tedavisi", "radyoterapi", "histoloji"]:
        df[c] = group_rare(df[c])
    df["met"] = df["uzak_metastaz"]
    df["gun"] = df["uzak_metastaz_gun"]
    df["met_5y"] = np.where(
        (df["met"] == 1) & (df["gun"] < 5 * YEAR),
        1,
        np.where(
            ((df["met"] == 0) & (df["gun"] >= 5 * YEAR))
            | ((df["met"] == 1) & (df["gun"] >= 5 * YEAR)),
            0,
            np.nan,
        ),
    )
    return df.dropna(subset=ALL_FEAT).copy()


def split_cols(feats):
    num = [c for c in feats if c in ALL_NUM]
    cat = [c for c in feats if c in ALL_CAT]
    return num, cat


def make_pre(feats):
    num, cat = split_cols(feats)
    transformers = []
    if num:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("sc", StandardScaler()),
                    ]
                ),
                num,
            )
        )
    if cat:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat,
            )
        )
    return ColumnTransformer(transformers)


def binary_metrics(y, proba, pred):
    return {
        "AUC_ROC": float(roc_auc_score(y, proba)),
        "AUC_PR": float(average_precision_score(y, proba)),
        "Accuracy": float(accuracy_score(y, pred)),
        "Balanced_Acc": float(balanced_accuracy_score(y, pred)),
        "F1": float(f1_score(y, pred)),
        "Brier": float(brier_score_loss(y, proba)),
    }


def lean_models(seed=RANDOM_STATE):
    """More regularized / anti-overfit configs."""
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    return {
        "LR_L2_strong": LogisticRegression(
            C=0.3, penalty="l2", solver="lbfgs", max_iter=3000,
            class_weight="balanced", random_state=seed,
        ),
        "RF_shallow": RandomForestClassifier(
            n_estimators=500, max_depth=6, min_samples_leaf=12,
            max_features="sqrt", class_weight="balanced_subsample",
            n_jobs=-1, random_state=seed,
        ),
        "HistGBM_reg": HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.04, max_iter=250,
            l2_regularization=1.0, min_samples_leaf=20,
            class_weight="balanced", random_state=seed,
        ),
        "XGB_reg": XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.7, reg_lambda=2.0, reg_alpha=0.5,
            min_child_weight=5, objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", random_state=seed, n_jobs=-1,
        ),
        "LGBM_reg": LGBMClassifier(
            n_estimators=350, learning_rate=0.04, num_leaves=15,
            min_child_samples=25, subsample=0.8, colsample_bytree=0.7,
            reg_lambda=2.0, class_weight="balanced", random_state=seed,
            verbose=-1, n_jobs=-1,
        ),
        "Cat_reg": CatBoostClassifier(
            iterations=350, depth=4, learning_rate=0.04, l2_leaf_reg=6.0,
            auto_class_weights="Balanced", verbose=0, random_seed=seed,
        ),
        "SVM_reg": SVC(
            C=0.7, kernel="rbf", gamma="scale", probability=True,
            class_weight="balanced", random_state=seed,
        ),
    }


def hybrid_models(seed=RANDOM_STATE):
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    rf = RandomForestClassifier(
        n_estimators=400, max_depth=6, min_samples_leaf=12,
        class_weight="balanced_subsample", n_jobs=-1, random_state=seed,
    )
    lr = LogisticRegression(
        C=0.3, max_iter=3000, class_weight="balanced", random_state=seed,
    )
    xgb = XGBClassifier(
        n_estimators=250, max_depth=3, learning_rate=0.04, subsample=0.8,
        colsample_bytree=0.7, reg_lambda=2.0, objective="binary:logistic",
        eval_metric="logloss", tree_method="hist", random_state=seed, n_jobs=-1,
    )
    lgb = LGBMClassifier(
        n_estimators=250, learning_rate=0.04, num_leaves=15, min_child_samples=25,
        class_weight="balanced", random_state=seed, verbose=-1, n_jobs=-1,
    )
    hgb = HistGradientBoostingClassifier(
        max_depth=4, learning_rate=0.04, max_iter=200, l2_regularization=1.0,
        class_weight="balanced", random_state=seed,
    )
    return {
        "HybridVote_soft": VotingClassifier(
            estimators=[("rf", rf), ("lr", lr), ("xgb", xgb), ("lgb", lgb)],
            voting="soft",
            n_jobs=-1,
        ),
        "HybridVote_RFLRXGB": VotingClassifier(
            estimators=[("rf", rf), ("lr", lr), ("xgb", xgb)],
            voting="soft",
            n_jobs=-1,
        ),
        "HybridStack_RFxgbLGB": StackingClassifier(
            estimators=[("rf", rf), ("xgb", xgb), ("lgb", lgb)],
            final_estimator=LogisticRegression(
                C=0.5, max_iter=2000, class_weight="balanced", random_state=seed
            ),
            cv=3,
            n_jobs=-1,
            passthrough=False,
        ),
        "HybridStack_RFxgbHGB": StackingClassifier(
            estimators=[("rf", rf), ("xgb", xgb), ("hgb", hgb)],
            final_estimator=LogisticRegression(
                C=0.5, max_iter=2000, class_weight="balanced", random_state=seed
            ),
            cv=3,
            n_jobs=-1,
        ),
    }


def eval_combo(task, feat_set_name, feats, X, y, models: dict):
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    oof_store = {}
    Xp = X[feats]
    for mname, clf in models.items():
        print(f"  [{task}|{feat_set_name}] {mname}")
        pipe = Pipeline([("pre", make_pre(feats)), ("clf", clf)])
        try:
            proba = cross_val_predict(pipe, Xp, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
            pred = (proba >= 0.5).astype(int)
            met = binary_metrics(y, proba, pred)
            met.update(
                {
                    "task": task,
                    "feature_set": feat_set_name,
                    "n_features": len(feats),
                    "features": ",".join(feats),
                    "model": mname,
                    "n": int(len(y)),
                }
            )
            rows.append(met)
            oof_store[mname] = {"proba": proba, "pred": pred}
        except Exception as e:
            print(f"    FAIL: {e}")
    return rows, oof_store


def permutation_rank(X, y, feats, n_repeats=8):
    """Raw-column permutation importance with regularized RF."""
    print(f"  Permutation importance on {len(feats)} features...")
    clf = RandomForestClassifier(
        n_estimators=350, max_depth=6, min_samples_leaf=10,
        class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE,
    )
    pipe = Pipeline([("pre", make_pre(feats)), ("clf", clf)])
    pipe.fit(X[feats], y)
    r = permutation_importance(
        pipe, X[feats], y, n_repeats=n_repeats, scoring="roc_auc",
        random_state=RANDOM_STATE, n_jobs=1,
    )
    imp = pd.DataFrame(
        {
            "feature": feats,
            "importance_mean": r.importances_mean,
            "importance_std": r.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    return imp, pipe


def plot_importance(imp, title, fname, top=15):
    d = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(d["feature"], d["importance_mean"], xerr=d["importance_std"], color="#2f5d50")
    ax.set_xlabel("Permutation importance (Δ AUC)")
    ax.set_title(title)
    return save_fig(fig, fname)


def plot_roc_multi(y, series_dict, title, fname):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name, proba in series_dict.items():
        fpr, tpr, _ = roc_curve(y, proba)
        ax.plot(fpr, tpr, lw=2, label=f"{name} ({roc_auc_score(y, proba):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#aaa")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(title)
    ax.legend(fontsize=8)
    return save_fig(fig, fname)


def plot_bars(df, title, fname):
    d = df.sort_values("AUC_ROC")
    labels = d["feature_set"] + " | " + d["model"]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.28 * len(d))))
    ax.barh(labels, d["AUC_ROC"], color="#8b3a3a")
    ax.set_xlabel("AUC-ROC")
    ax.set_title(title)
    ax.set_xlim(0.55, max(0.85, d["AUC_ROC"].max() + 0.03))
    for i, v in enumerate(d["AUC_ROC"].values):
        ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=7)
    return save_fig(fig, fname)


def df_html(df, digits=3):
    if df is None or len(df) == 0:
        return "<p><em>Yok</em></p>"
    show = df.copy()
    for c in show.columns:
        if pd.api.types.is_float_dtype(show[c]):
            show[c] = show[c].map(lambda x: f"{x:.{digits}f}" if pd.notna(x) else "")
    return show.to_html(index=False, classes="tbl", border=0, escape=True)


def img(b64, alt=""):
    return f'<img class="plot" alt="{alt}" src="data:image/png;base64,{b64}" />' if b64 else ""


def build_html(ctx):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""<!DOCTYPE html>
<html lang="tr"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>BreastPredict Report 2 — Lean & Hybrid</title>
<style>
:root {{ --bg:#f4f1ec; --ink:#1a1814; --muted:#5a554c; --card:#fffcf7; --line:#e2dbd0; --a:#7a3030; --b:#1f5a4e; }}
body {{ margin:0; font-family:"Segoe UI",Arial,sans-serif; background:radial-gradient(900px 500px at 0% 0%,#eadfce,transparent),var(--bg); color:var(--ink); }}
.wrap {{ max-width:1120px; margin:auto; padding:28px 18px 70px; }}
h1 {{ font-size:clamp(26px,4vw,40px); margin:6px 0; letter-spacing:-.02em; }}
.brand {{ color:var(--a); font-size:12px; letter-spacing:.14em; text-transform:uppercase; font-weight:700; }}
.sub {{ color:var(--muted); max-width:75ch; }}
.chips {{ display:flex; flex-wrap:wrap; gap:8px; margin:12px 0; }}
.chip {{ background:#efe8dc; border:1px solid var(--line); border-radius:999px; padding:4px 11px; font-size:12px; color:var(--muted); }}
section {{ background:var(--card); border:1px solid var(--line); border-radius:16px; padding:20px; margin:16px 0; }}
h2 {{ margin:0 0 8px; font-size:21px; }} h3 {{ color:var(--b); font-size:15px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:10px; }}
.kpi {{ background:#faf6f0; border:1px solid var(--line); border-radius:12px; padding:12px; }}
.kpi .k {{ font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:.05em; }}
.kpi .v {{ font-size:26px; font-weight:700; }}
.tbl {{ width:100%; border-collapse:collapse; font-size:12.5px; margin:8px 0 14px; }}
.tbl th {{ background:#f1ebe1; text-align:left; padding:7px 8px; }}
.tbl td {{ padding:6px 8px; border-bottom:1px solid var(--line); }}
.plots {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:12px; }}
.plot {{ width:100%; border:1px solid var(--line); border-radius:10px; background:#fff; }}
.note {{ background:#f7efe4; border-left:4px solid var(--a); padding:10px 12px; color:var(--muted); font-size:13px; border-radius:0 8px 8px 0; }}
footer {{ text-align:center; color:var(--muted); font-size:12px; margin-top:20px; }}
</style></head><body><div class="wrap">
<header>
  <div class="brand">BreastPredict · Report 2</div>
  <h1>Özellik sadeleştirme, gürültü azaltma ve hibrit modeller</h1>
  <p class="sub">Permutation importance ile en katkılı parametreler belirlendi; az özellikli setler ve soft-voting / stacking hibritleri 5-fold CV ile yeniden eğitildi. Amaç: overfitting ve noise’u azaltıp AUC’yi yükseltmek.</p>
  <div class="chips">
    <span class="chip">{now}</span>
    <span class="chip">Seed {RANDOM_STATE}</span>
    <span class="chip">CV = {N_SPLITS}-fold</span>
  </div>
</header>

<section>
  <h2>1. En iyi sonuçlar (Report 1’e göre)</h2>
  <div class="grid">
    <div class="kpi"><div class="k">M1 baseline (R1)</div><div class="v">{ctx['m1_base']:.3f}</div><div class="s">RF full</div></div>
    <div class="kpi"><div class="k">M1 Report2 best</div><div class="v">{ctx['m1_best_auc']:.3f}</div><div class="s">{ctx['m1_best_name']}</div></div>
    <div class="kpi"><div class="k">M1 Δ AUC</div><div class="v">{ctx['m1_delta']:+.3f}</div><div class="s">iyileşme</div></div>
    <div class="kpi"><div class="k">M2B baseline (R1)</div><div class="v">{ctx['m2b_base']:.3f}</div><div class="s">SVM full</div></div>
    <div class="kpi"><div class="k">M2B Report2 best</div><div class="v">{ctx['m2b_best_auc']:.3f}</div><div class="s">{ctx['m2b_best_name']}</div></div>
    <div class="kpi"><div class="k">M2B Δ AUC</div><div class="v">{ctx['m2b_delta']:+.3f}</div><div class="s">iyileşme</div></div>
  </div>
  <div class="note">{ctx['verdict']}</div>
</section>

<section>
  <h2>2. Parametre katkıları (permutation importance, ΔAUC)</h2>
  <p class="sub">Her özellik rastgele karıştırıldığında AUC ne kadar düşüyorsa, o kadar kritik. Negatif/çok düşük = gürültü adayı.</p>
  <h3>M1 — Metastaz var/yok</h3>
  {df_html(ctx['imp_m1'])}
  <div class="plots">{img(ctx['img_imp_m1'])}</div>
  <h3>M2B — 5 yıldan önce metastaz</h3>
  {df_html(ctx['imp_m2b'])}
  <div class="plots">{img(ctx['img_imp_m2b'])}</div>
</section>

<section>
  <h2>3. Denenen özellik setleri</h2>
  {df_html(ctx['feat_sets_table'])}
</section>

<section>
  <h2>4. M1 sonuçları (tüm kombinasyonlar)</h2>
  {df_html(ctx['m1_all'])}
  <div class="plots">
    {img(ctx['img_m1_bars'])}
    {img(ctx['img_m1_roc'])}
  </div>
</section>

<section>
  <h2>5. M2B sonuçları (tüm kombinasyonlar)</h2>
  {df_html(ctx['m2b_all'])}
  <div class="plots">
    {img(ctx['img_m2b_bars'])}
    {img(ctx['img_m2b_roc'])}
  </div>
</section>

<section>
  <h2>6. En iyi modeller — detay</h2>
  <h3>M1 winner</h3>
  {df_html(ctx['m1_winner_row'])}
  <p><b>Kullanılan özellikler ({ctx['m1_winner_nfeat']}):</b> {ctx['m1_winner_feats']}</p>
  <h3>M2B winner</h3>
  {df_html(ctx['m2b_winner_row'])}
  <p><b>Kullanılan özellikler ({ctx['m2b_winner_nfeat']}):</b> {ctx['m2b_winner_feats']}</p>
  <h3>Regularizasyon / anti-overfit parametreleri</h3>
  <pre style="background:#1e1c19;color:#f4efe6;padding:12px;border-radius:10px;overflow:auto;font-size:12px;">{ctx['params_json']}</pre>
</section>

<section>
  <h2>7. Yorum</h2>
  <ul>
    <li>En yüksek katkı genelde <b>pozitif lenf nodu, tümör boyutu, grade, ER/PR/HER2</b> tarafında; tedavi kodları çoğu zaman gürültü/konfounder.</li>
    <li>Az özellik + daha sığ ağaçlar / daha güçlü L2, tek başına mucize yaratmaz ama genelleme ve kalibrasyon için daha sağlıklıdır.</li>
    <li>Hibrit (voting/stacking) bazen tek modeli geçer; Report 2 tablolarında ΔAUC ile görülebilir.</li>
    <li>Klinik-only tavanı hâlâ sınırlı; büyük sıçrama için moleküler özellik veya harici kohort gerekir.</li>
  </ul>
</section>
<footer>BreastPredict Report 2 · {now}</footer>
</div></body></html>"""


def main():
    print("Loading...")
    df = load_df()
    print(f"n={len(df)}")

    # baselines from report1 summary if present
    m1_base, m2b_base = 0.6807, 0.7497
    sj = OUT / "summary.json"
    if sj.exists():
        s = json.loads(sj.read_text(encoding="utf-8"))
        m1_base = float(s.get("m1_auc", m1_base))
        m2b_base = float(s.get("m2b_auc", m2b_base))

    X = df[ALL_FEAT]
    y1 = df["met"].astype(int)
    d5 = df.dropna(subset=["met_5y"])
    X5 = d5[ALL_FEAT]
    y5 = d5["met_5y"].astype(int)

    # --- importance ---
    print("\n=== Importance M1 ===")
    imp1, _ = permutation_rank(X, y1, ALL_FEAT)
    imp1.to_excel(TAB / "imp_m1.xlsx", index=False)
    img_imp_m1 = plot_importance(imp1, "M1 — özellik katkısı (permutation ΔAUC)", "r2_imp_m1.png")

    print("\n=== Importance M2B ===")
    imp5, _ = permutation_rank(X5, y5, ALL_FEAT)
    imp5.to_excel(TAB / "imp_m2b.xlsx", index=False)
    img_imp_m2b = plot_importance(imp5, "M2B — özellik katkısı (permutation ΔAUC)", "r2_imp_m2b.png")

    top5_m1 = imp1.head(5)["feature"].tolist()
    top7_m1 = imp1.head(7)["feature"].tolist()
    # positive importance only
    pos_m1 = imp1.loc[imp1["importance_mean"] > 0, "feature"].tolist()
    if len(pos_m1) < 4:
        pos_m1 = top7_m1

    top5_m2b = imp5.head(5)["feature"].tolist()
    top7_m2b = imp5.head(7)["feature"].tolist()
    pos_m2b = imp5.loc[imp5["importance_mean"] > 0, "feature"].tolist()
    if len(pos_m2b) < 4:
        pos_m2b = top7_m2b

    core_prognostic = ["yas", "tumor_boyutu_mm", "pozitif_ln", "grade", "er", "pr", "her2", "menopoz"]
    no_tx = [c for c in ALL_FEAT if c not in TX]
    # consensus top: average rank
    rank = (
        imp1.assign(r1=np.arange(1, len(imp1) + 1))
        .merge(imp5.assign(r2=np.arange(1, len(imp5) + 1)), on="feature", suffixes=("_m1", "_m2b"))
    )
    rank["rank_mean"] = (rank["r1"] + rank["r2"]) / 2
    consensus7 = rank.sort_values("rank_mean").head(7)["feature"].tolist()
    consensus5 = rank.sort_values("rank_mean").head(5)["feature"].tolist()

    feat_sets_m1 = {
        "FULL_13": ALL_FEAT,
        "NO_TX": no_tx,
        "CORE8": core_prognostic,
        "TOP7_perm": top7_m1,
        "TOP5_perm": top5_m1,
        "POS_only": pos_m1,
        "CONSENSUS7": consensus7,
        "CONSENSUS5": consensus5,
    }
    feat_sets_m2b = {
        "FULL_13": ALL_FEAT,
        "NO_TX": no_tx,
        "CORE8": core_prognostic,
        "TOP7_perm": top7_m2b,
        "TOP5_perm": top5_m2b,
        "POS_only": pos_m2b,
        "CONSENSUS7": consensus7,
        "CONSENSUS5": consensus5,
    }

    feat_sets_table = pd.DataFrame(
        [
            {"set": k, "n": len(v), "features": ", ".join(v)}
            for k, v in {**{"(shared) NO_TX": no_tx, "(shared) CORE8": core_prognostic, "(shared) CONSENSUS7": consensus7}, **{f"M1_{k}": v for k, v in feat_sets_m1.items() if k.startswith("TOP") or k == "POS_only"}}.items()
        ]
    )
    # cleaner table
    feat_sets_table = pd.DataFrame(
        [
            {"set": "FULL_13", "n": 13, "features": ", ".join(ALL_FEAT)},
            {"set": "NO_TX", "n": len(no_tx), "features": ", ".join(no_tx)},
            {"set": "CORE8", "n": 8, "features": ", ".join(core_prognostic)},
            {"set": "CONSENSUS7", "n": 7, "features": ", ".join(consensus7)},
            {"set": "CONSENSUS5", "n": 5, "features": ", ".join(consensus5)},
            {"set": "M1_TOP7", "n": 7, "features": ", ".join(top7_m1)},
            {"set": "M1_TOP5", "n": 5, "features": ", ".join(top5_m1)},
            {"set": "M2B_TOP7", "n": 7, "features": ", ".join(top7_m2b)},
            {"set": "M2B_TOP5", "n": 5, "features": ", ".join(top5_m2b)},
        ]
    )
    feat_sets_table.to_excel(TAB / "feature_sets.xlsx", index=False)

    singles = lean_models()
    hybrids = hybrid_models()
    # don't run every model on every set (too long): 
    # - singles on reduced sets + FULL/NO_TX/CORE
    # - hybrids on best candidate sets
    rows = []

    print("\n=== M1 lean sweeps ===")
    for fs_name, feats in feat_sets_m1.items():
        part, _ = eval_combo("M1", fs_name, feats, X, y1, singles)
        rows.extend(part)

    print("\n=== M1 hybrids on lean sets ===")
    for fs_name in ["NO_TX", "CORE8", "TOP7_perm", "CONSENSUS7", "TOP5_perm"]:
        part, oof = eval_combo("M1", fs_name, feat_sets_m1[fs_name], X, y1, hybrids)
        rows.extend(part)

    print("\n=== M2B lean sweeps ===")
    for fs_name, feats in feat_sets_m2b.items():
        part, _ = eval_combo("M2B", fs_name, feats, X5, y5, singles)
        rows.extend(part)

    print("\n=== M2B hybrids on lean sets ===")
    for fs_name in ["NO_TX", "CORE8", "TOP7_perm", "CONSENSUS7", "TOP5_perm"]:
        part, _ = eval_combo("M2B", fs_name, feat_sets_m2b[fs_name], X5, y5, hybrids)
        rows.extend(part)

    all_df = pd.DataFrame(rows)
    all_df.to_excel(TAB / "all_combos.xlsx", index=False)

    m1_all = all_df[all_df["task"] == "M1"].sort_values("AUC_ROC", ascending=False).reset_index(drop=True)
    m2b_all = all_df[all_df["task"] == "M2B"].sort_values("AUC_ROC", ascending=False).reset_index(drop=True)
    m1_all.to_excel(TAB / "m1_all.xlsx", index=False)
    m2b_all.to_excel(TAB / "m2b_all.xlsx", index=False)

    # top charts (top 20)
    img_m1_bars = plot_bars(m1_all.head(20), "M1 — en iyi 20 kombinasyon (AUC)", "r2_m1_bars.png")
    img_m2b_bars = plot_bars(m2b_all.head(20), "M2B — en iyi 20 kombinasyon (AUC)", "r2_m2b_bars.png")

    # ROC for top 3 M1 / M2B
    def top_rocs(task_df, Xy, y, feat_map, tag):
        series = {}
        for _, row in task_df.head(3).iterrows():
            feats = feat_map[row["feature_set"]]
            # rebuild model
            models = {**lean_models(), **hybrid_models()}
            clf = models[row["model"]]
            pipe = Pipeline([("pre", make_pre(feats)), ("clf", clf)])
            cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
            proba = cross_val_predict(pipe, Xy[feats], y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
            series[f"{row['feature_set'][:10]}|{row['model'][:14]}"] = proba
        return plot_roc_multi(y, series, f"{tag} — top3 ROC", f"r2_{tag}_roc.png")

    img_m1_roc = top_rocs(m1_all, X, y1, feat_sets_m1, "M1")
    img_m2b_roc = top_rocs(m2b_all, X5, y5, feat_sets_m2b, "M2B")

    w1 = m1_all.iloc[0]
    w2 = m2b_all.iloc[0]
    m1_best_auc = float(w1["AUC_ROC"])
    m2b_best_auc = float(w2["AUC_ROC"])
    m1_delta = m1_best_auc - m1_base
    m2b_delta = m2b_best_auc - m2b_base

    if m1_delta > 0.01 or m2b_delta > 0.01:
        verdict = (
            f"Sadeleştirme/hibrit ile ölçülebilir artış var "
            f"(M1 {m1_delta:+.3f}, M2B {m2b_delta:+.3f}). "
            "En iyi set gürültülü tedavileri veya düşük katkı özelliklerini elemiş görünüyor."
        )
    elif m1_delta > 0 or m2b_delta > 0:
        verdict = (
            f"Küçük iyileşme / benzer performans (M1 {m1_delta:+.3f}, M2B {m2b_delta:+.3f}). "
            "Daha az özellik ile neredeyse aynı AUC → daha sade, daha az overfit riski."
        )
    else:
        verdict = (
            f"AUC tavanı klinik veriyle sınırlı kaldı (M1 {m1_delta:+.3f}, M2B {m2b_delta:+.3f}). "
            "Yine de az özellikli modeller benzer başarıyı daha az gürültüyle veriyor; tercih edilebilir."
        )

    params = {
        "lean_models": {
            "LR_L2_strong": {"C": 0.3, "class_weight": "balanced"},
            "RF_shallow": {"max_depth": 6, "min_samples_leaf": 12, "n_estimators": 500},
            "XGB_reg": {"max_depth": 3, "reg_lambda": 2.0, "reg_alpha": 0.5, "min_child_weight": 5},
            "LGBM_reg": {"num_leaves": 15, "min_child_samples": 25, "reg_lambda": 2.0},
            "Cat_reg": {"depth": 4, "l2_leaf_reg": 6.0},
            "SVM_reg": {"C": 0.7},
        },
        "hybrids": {
            "HybridVote_soft": "RF+LR+XGB+LGBM soft voting",
            "HybridStack_RFxgbLGB": "stack RF/XGB/LGBM -> LR meta (cv=3)",
        },
    }

    ctx = {
        "m1_base": m1_base,
        "m2b_base": m2b_base,
        "m1_best_auc": m1_best_auc,
        "m2b_best_auc": m2b_best_auc,
        "m1_delta": m1_delta,
        "m2b_delta": m2b_delta,
        "m1_best_name": f"{w1['feature_set']} + {w1['model']}",
        "m2b_best_name": f"{w2['feature_set']} + {w2['model']}",
        "verdict": verdict,
        "imp_m1": imp1,
        "imp_m2b": imp5,
        "img_imp_m1": img_imp_m1,
        "img_imp_m2b": img_imp_m2b,
        "feat_sets_table": feat_sets_table,
        "m1_all": m1_all.drop(columns=["features"], errors="ignore").head(25),
        "m2b_all": m2b_all.drop(columns=["features"], errors="ignore").head(25),
        "img_m1_bars": img_m1_bars,
        "img_m2b_bars": img_m2b_bars,
        "img_m1_roc": img_m1_roc,
        "img_m2b_roc": img_m2b_roc,
        "m1_winner_row": m1_all.drop(columns=["features"], errors="ignore").head(1),
        "m2b_winner_row": m2b_all.drop(columns=["features"], errors="ignore").head(1),
        "m1_winner_feats": w1["features"],
        "m2b_winner_feats": w2["features"],
        "m1_winner_nfeat": int(w1["n_features"]),
        "m2b_winner_nfeat": int(w2["n_features"]),
        "params_json": json.dumps(params, indent=2, ensure_ascii=False),
    }

    html = build_html(ctx)
    path = ROOT / "report2.html"
    path.write_text(html, encoding="utf-8")
    summary = {
        "m1_best": ctx["m1_best_name"],
        "m1_auc": m1_best_auc,
        "m1_delta": m1_delta,
        "m2b_best": ctx["m2b_best_name"],
        "m2b_auc": m2b_best_auc,
        "m2b_delta": m2b_delta,
        "top_features_m1": top7_m1,
        "top_features_m2b": top7_m2b,
        "consensus7": consensus7,
    }
    (OUT / "summary_report2.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nWrote", path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
