"""
BreastPredict — full modeling pipeline + internal validation + XAI + HTML report.
Models:
  M1  Binary distant metastasis (yes/no)
  M2A Multi-class time bin (y1/y2/y3/y4/y5+/no_met>=5y)
  M2B Cumulative 5-year metastasis risk
  M2S Survival (Cox PH + Random Survival Forest)
  M3  Coarse first distant site (bone/visceral/other)
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
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
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
FIG = OUT / "figures"
TAB = OUT / "tables"
for d in (OUT, FIG, TAB, OUT / "models"):
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5
YEAR = 365.25

NUMERIC = ["yas", "tumor_boyutu_mm", "pozitif_ln", "grade"]
CATEGORICAL = [
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
FEATURES = NUMERIC + CATEGORICAL

# Documented hyperparameters (applied uniformly unless noted)
PARAMS = {
    "LogisticRegression": {
        "C": 1.0,
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": 2000,
        "class_weight": "balanced",
    },
    "RandomForest": {
        "n_estimators": 400,
        "max_depth": 12,
        "min_samples_leaf": 5,
        "class_weight": "balanced_subsample",
        "n_jobs": -1,
    },
    "ExtraTrees": {
        "n_estimators": 400,
        "max_depth": 14,
        "min_samples_leaf": 4,
        "class_weight": "balanced",
        "n_jobs": -1,
    },
    "HistGBM": {
        "max_depth": 8,
        "learning_rate": 0.06,
        "max_iter": 300,
        "l2_regularization": 0.1,
        "class_weight": "balanced",
    },
    "GradientBoosting": {
        "n_estimators": 250,
        "learning_rate": 0.05,
        "max_depth": 3,
        "subsample": 0.9,
    },
    "XGBoost": {
        "n_estimators": 350,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "objective": "binary:logistic / multi:softprob",
        "eval_metric": "logloss",
        "tree_method": "hist",
    },
    "LightGBM": {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "class_weight": "balanced",
    },
    "CatBoost": {
        "iterations": 400,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 3.0,
        "auto_class_weights": "Balanced",
        "verbose": 0,
    },
    "SVM_RBF": {
        "C": 1.0,
        "kernel": "rbf",
        "gamma": "scale",
        "probability": True,
        "class_weight": "balanced",
    },
    "CoxPH": {"penalizer": 0.1, "l1_ratio": 0.0},
    "RSF": {
        "n_estimators": 200,
        "min_samples_split": 10,
        "min_samples_leaf": 6,
        "max_features": "sqrt",
        "n_jobs": -1,
    },
}


def fig_to_b64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def save_fig(fig, name: str) -> str:
    path = FIG / name
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    b64 = fig_to_b64(fig)
    return b64


def group_rare(series: pd.Series, min_count: int = 25, other: str = "OTHER") -> pd.Series:
    vc = series.fillna("MISSING").astype(str).value_counts()
    keep = set(vc[vc >= min_count].index)
    return series.fillna("MISSING").astype(str).map(lambda x: x if x in keep else other)


def exclusive_label(met, gun):
    if pd.isna(met) or pd.isna(gun):
        return np.nan
    if met == 1:
        y = gun / YEAR
        if y < 1:
            return "y1"
        if y < 2:
            return "y2"
        if y < 3:
            return "y3"
        if y < 4:
            return "y4"
        return "y5_plus"
    if gun >= 5 * YEAR:
        return "no_met_ge5y"
    return "censor_lt5y"


def coarse_site(s):
    if pd.isna(s):
        return np.nan
    u = str(s).upper()
    has_bone = "BONE" in u
    has_cns = any(x in u for x in ["BRAIN", "CNS"])
    has_visc = any(
        x in u
        for x in ["LIVER", "LUNG", "PLEURA", "PERITONE", "ASCITES", "ADRENAL", "MEDIAST"]
    )
    tags = []
    if has_bone:
        tags.append("bone")
    if has_visc:
        tags.append("visceral")
    if has_cns:
        tags.append("cns")
    if not tags:
        return "other"
    if len(tags) == 1:
        return tags[0]
    return "multi"


def load_prepared() -> pd.DataFrame:
    df = pd.read_excel(ROOT / "database.xlsx", sheet_name="hastalar")
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].replace({"": np.nan, " ": np.nan})
    df["gun"] = pd.to_numeric(df["uzak_metastaz_gun"], errors="coerce")
    df["met"] = pd.to_numeric(df["uzak_metastaz"], errors="coerce")
    df["grade"] = pd.to_numeric(df["grade"], errors="coerce")
    df["yas"] = pd.to_numeric(df["yas"], errors="coerce")
    df["tumor_boyutu_mm"] = pd.to_numeric(df["tumor_boyutu_mm"], errors="coerce")
    df["pozitif_ln"] = pd.to_numeric(df["pozitif_ln"], errors="coerce")

    for c in ["kemoterapi", "hormon_tedavisi", "radyoterapi", "histoloji"]:
        df[c] = group_rare(df[c], min_count=25)

    df["ex_label"] = [
        exclusive_label(m, g) for m, g in zip(df["met"], df["gun"])
    ]
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
    df["site_coarse_raw"] = df["ilk_uzak_metastaz_bolgesi"].map(coarse_site)
    # collapse rare CNS into other for modeling stability
    df["site_label"] = df["site_coarse_raw"].replace({"cns": "other", "multi": "multi_other", "other": "multi_other"})

    need = FEATURES + ["met", "gun", "ex_label", "met_5y", "site_label", "hasta_id"]
    full = df.dropna(subset=FEATURES).copy()
    return full


def make_preprocess(numeric, categorical) -> ColumnTransformer:
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric),
            ("cat", cat_pipe, categorical),
        ]
    )


def get_binary_models():
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    return {
        "LogisticRegression": LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="lbfgs",
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=400,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=400,
            max_depth=14,
            min_samples_leaf=4,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "HistGBM": HistGradientBoostingClassifier(
            max_depth=8,
            learning_rate=0.06,
            max_iter=300,
            l2_regularization=0.1,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=350,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            verbose=-1,
            n_jobs=-1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=400,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=3.0,
            auto_class_weights="Balanced",
            verbose=0,
            random_seed=RANDOM_STATE,
        ),
        "SVM_RBF": SVC(
            C=1.0,
            kernel="rbf",
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }


def get_multiclass_models(n_classes: int):
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    return {
        "LogisticRegression": LogisticRegression(
            C=1.0,
            max_iter=2500,
            class_weight="balanced",
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=400,
            max_depth=14,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "HistGBM": HistGradientBoostingClassifier(
            max_depth=8,
            learning_rate=0.06,
            max_iter=300,
            l2_regularization=0.1,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=350,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            verbose=-1,
            n_jobs=-1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=400,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=3.0,
            auto_class_weights="Balanced",
            verbose=0,
            random_seed=RANDOM_STATE,
            loss_function="MultiClass",
        ),
    }


def binary_metrics(y_true, y_prob, y_pred):
    return {
        "AUC_ROC": float(roc_auc_score(y_true, y_prob)),
        "AUC_PR": float(average_precision_score(y_true, y_prob)),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "F1": float(f1_score(y_true, y_pred)),
        "Brier": float(brier_score_loss(y_true, y_prob)),
    }


def multiclass_metrics(y_true, y_pred, y_prob=None, labels=None):
    out = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "Macro_F1": float(f1_score(y_true, y_pred, average="macro")),
        "Weighted_F1": float(f1_score(y_true, y_pred, average="weighted")),
    }
    if y_prob is not None and labels is not None and len(np.unique(y_true)) > 2:
        try:
            out["AUC_OvR_macro"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro", labels=labels)
            )
        except Exception:
            out["AUC_OvR_macro"] = np.nan
    return out


def evaluate_binary(name, X, y, models):
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    oof = {}
    pre = make_preprocess(NUMERIC, CATEGORICAL)
    for mname, clf in models.items():
        print(f"  [{name}] {mname} ...")
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        try:
            proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=1)
            pred = (proba[:, 1] >= 0.5).astype(int)
            met = binary_metrics(y, proba[:, 1], pred)
            met["model"] = mname
            rows.append(met)
            oof[mname] = {"proba": proba[:, 1], "pred": pred}
        except Exception as e:
            print(f"    FAIL {mname}: {e}")
    res = pd.DataFrame(rows).sort_values("AUC_ROC", ascending=False).reset_index(drop=True)
    return res, oof, y


def evaluate_multiclass(name, X, y, models, class_order):
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    # map labels to 0..K-1 for XGBoost
    label_to_i = {c: i for i, c in enumerate(class_order)}
    y_enc = np.array([label_to_i[v] for v in y])
    rows = []
    oof = {}
    pre = make_preprocess(NUMERIC, CATEGORICAL)
    for mname, clf in models.items():
        print(f"  [{name}] {mname} ...")
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        try:
            proba = cross_val_predict(pipe, X, y_enc, cv=cv, method="predict_proba", n_jobs=1)
            pred_i = proba.argmax(axis=1)
            pred = np.array(class_order)[pred_i]
            met = multiclass_metrics(y, pred, proba, labels=list(range(len(class_order))))
            # remap AUC labels issue: use y_enc
            try:
                met["AUC_OvR_macro"] = float(
                    roc_auc_score(y_enc, proba, multi_class="ovr", average="macro")
                )
            except Exception:
                met["AUC_OvR_macro"] = np.nan
            met["model"] = mname
            rows.append(met)
            oof[mname] = {"proba": proba, "pred": pred, "y_enc": y_enc}
        except Exception as e:
            print(f"    FAIL {mname}: {e}")
    res = pd.DataFrame(rows).sort_values("Macro_F1", ascending=False).reset_index(drop=True)
    return res, oof, y, class_order


def fit_best_and_explain(X, y, best_name, models, task="binary", class_order=None):
    """Fit on full data for SHAP / coefficients / permutation importance."""
    import shap

    pre = make_preprocess(NUMERIC, CATEGORICAL)
    clf = models[best_name]
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    if task == "multiclass" and class_order is not None:
        label_to_i = {c: i for i, c in enumerate(class_order)}
        y_fit = np.array([label_to_i[v] for v in y])
    else:
        y_fit = np.asarray(y)

    pipe.fit(X, y_fit)
    Xt = pipe.named_steps["pre"].transform(X)
    feature_names = pipe.named_steps["pre"].get_feature_names_out()
    feature_names = [f.replace("num__", "").replace("cat__", "") for f in feature_names]

    expl = {"model": best_name, "feature_names": feature_names, "images": {}, "tables": {}}

    # Coefficients for linear
    if best_name == "LogisticRegression":
        coef = pipe.named_steps["clf"].coef_
        if coef.ndim == 1 or coef.shape[0] == 1:
            s = pd.Series(coef.ravel(), index=feature_names).sort_values(key=np.abs, ascending=False)
            top = s.head(25)
            fig, ax = plt.subplots(figsize=(8, 7))
            colors = ["#c0392b" if v > 0 else "#2980b9" for v in top.values]
            ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1])
            ax.set_title(f"{best_name} — katsayılar (top 25)")
            ax.set_xlabel("Katsayı (ölçeklenmiş özellikler)")
            expl["images"]["coef"] = save_fig(fig, f"xai_coef_{task}.png")
            expl["tables"]["coef"] = (
                top.reset_index().rename(columns={"index": "feature", 0: "coefficient"})
            )
        else:
            # multiclass: mean abs
            s = pd.Series(np.mean(np.abs(coef), axis=0), index=feature_names).sort_values(ascending=False)
            top = s.head(25)
            fig, ax = plt.subplots(figsize=(8, 7))
            ax.barh(top.index[::-1], top.values[::-1], color="#8e44ad")
            ax.set_title(f"{best_name} — |katsayı| ortalaması (top 25)")
            expl["images"]["coef"] = save_fig(fig, f"xai_coef_{task}.png")
            expl["tables"]["coef"] = top.reset_index().rename(columns={"index": "feature", 0: "mean_abs_coef"})

    # Tree / SHAP
    try:
        model = pipe.named_steps["clf"]
        # sample for speed
        rng = np.random.RandomState(RANDOM_STATE)
        idx = rng.choice(len(Xt), size=min(400, len(Xt)), replace=False)
        Xs = Xt[idx]

        if best_name in {"RandomForest", "ExtraTrees", "XGBoost", "LightGBM", "CatBoost", "HistGBM", "GradientBoosting"}:
            if best_name == "CatBoost":
                explainer = shap.TreeExplainer(model)
            elif best_name == "HistGBM":
                explainer = shap.Explainer(model.predict, Xs)  # fallback
                sv = explainer(Xs)
                shap_vals = sv.values if hasattr(sv, "values") else sv
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[1] if task == "binary" else np.mean([np.abs(v) for v in shap_vals], axis=0)
                mean_abs = np.mean(np.abs(shap_vals), axis=0)
                if mean_abs.ndim > 1:
                    mean_abs = mean_abs.mean(axis=-1)
                s = pd.Series(mean_abs, index=feature_names).sort_values(ascending=False).head(25)
                fig, ax = plt.subplots(figsize=(8, 7))
                ax.barh(s.index[::-1], s.values[::-1], color="#16a085")
                ax.set_title(f"{best_name} — |SHAP| ortalama (top 25)")
                expl["images"]["shap"] = save_fig(fig, f"xai_shap_{task}.png")
                expl["tables"]["shap"] = s.reset_index().rename(columns={"index": "feature", 0: "mean_abs_shap"})
                return pipe, expl
            else:
                explainer = shap.TreeExplainer(model)

            shap_values = explainer.shap_values(Xs)
            if isinstance(shap_values, list):
                # binary: often [class0, class1]; multiclass: list per class
                if task == "binary" and len(shap_values) == 2:
                    sv = shap_values[1]
                else:
                    sv = np.mean([np.abs(v) for v in shap_values], axis=0)
                    mean_abs = np.mean(sv, axis=0)
                    s = pd.Series(mean_abs, index=feature_names).sort_values(ascending=False).head(25)
                    fig, ax = plt.subplots(figsize=(8, 7))
                    ax.barh(s.index[::-1], s.values[::-1], color="#16a085")
                    ax.set_title(f"{best_name} — |SHAP| ortalama (top 25)")
                    expl["images"]["shap"] = save_fig(fig, f"xai_shap_{task}.png")
                    expl["tables"]["shap"] = s.reset_index().rename(
                        columns={"index": "feature", 0: "mean_abs_shap"}
                    )
                    # also native importance if available
                    if hasattr(model, "feature_importances_"):
                        fi = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(25)
                        fig, ax = plt.subplots(figsize=(8, 7))
                        ax.barh(fi.index[::-1], fi.values[::-1], color="#d35400")
                        ax.set_title(f"{best_name} — native feature importance")
                        expl["images"]["fi"] = save_fig(fig, f"xai_fi_{task}.png")
                        expl["tables"]["fi"] = fi.reset_index().rename(columns={"index": "feature", 0: "importance"})
                    return pipe, expl
            else:
                sv = shap_values
                if sv.ndim == 3:  # (n, features, classes)
                    mean_abs = np.mean(np.abs(sv), axis=(0, 2))
                else:
                    mean_abs = np.mean(np.abs(sv), axis=0)
                s = pd.Series(mean_abs, index=feature_names).sort_values(ascending=False).head(25)
                fig, ax = plt.subplots(figsize=(8, 7))
                ax.barh(s.index[::-1], s.values[::-1], color="#16a085")
                ax.set_title(f"{best_name} — mean |SHAP| (top 25)")
                expl["images"]["shap"] = save_fig(fig, f"xai_shap_{task}.png")
                expl["tables"]["shap"] = s.reset_index().rename(columns={"index": "feature", 0: "mean_abs_shap"})
        else:
            # permutation importance fallback (SVM etc.)
            r = permutation_importance(pipe, X, y_fit, n_repeats=5, random_state=RANDOM_STATE, n_jobs=1)
            # map back hard; use transformed names via approximate: permute on raw via pipeline scores
            s = pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False).head(20)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(s.index[::-1], s.values[::-1], color="#7f8c8d")
            ax.set_title(f"{best_name} — permutation importance")
            expl["images"]["perm"] = save_fig(fig, f"xai_perm_{task}.png")
            expl["tables"]["perm"] = s.reset_index().rename(columns={"index": "feature", 0: "perm_importance"})
    except Exception as e:
        print(f"  XAI partial fail ({best_name}): {e}")
        # native importance fallback
        model = pipe.named_steps["clf"]
        if hasattr(model, "feature_importances_"):
            fi = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(25)
            fig, ax = plt.subplots(figsize=(8, 7))
            ax.barh(fi.index[::-1], fi.values[::-1], color="#d35400")
            ax.set_title(f"{best_name} — native feature importance")
            expl["images"]["fi"] = save_fig(fig, f"xai_fi_{task}.png")
            expl["tables"]["fi"] = fi.reset_index().rename(columns={"index": "feature", 0: "importance"})

    if hasattr(pipe.named_steps["clf"], "feature_importances_") and "fi" not in expl["images"]:
        fi = pd.Series(pipe.named_steps["clf"].feature_importances_, index=feature_names).sort_values(ascending=False).head(25)
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.barh(fi.index[::-1], fi.values[::-1], color="#d35400")
        ax.set_title(f"{best_name} — native feature importance")
        expl["images"]["fi"] = save_fig(fig, f"xai_fi_{task}.png")
        expl["tables"]["fi"] = fi.reset_index().rename(columns={"index": "feature", 0: "importance"})

    return pipe, expl


def plot_binary_diagnostics(y, oof, best, task_tag):
    imgs = {}
    proba = oof[best]["proba"]
    pred = oof[best]["pred"]

    fig, ax = plt.subplots(figsize=(6, 5))
    fpr, tpr, _ = roc_curve(y, proba)
    ax.plot(fpr, tpr, color="#c0392b", lw=2, label=f"AUC={roc_auc_score(y, proba):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="#95a5a6")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{task_tag} — ROC ({best})")
    ax.legend()
    imgs["roc"] = save_fig(fig, f"{task_tag}_roc.png")

    fig, ax = plt.subplots(figsize=(6, 5))
    p, r, _ = precision_recall_curve(y, proba)
    ax.plot(r, p, color="#2980b9", lw=2, label=f"AP={average_precision_score(y, proba):.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{task_tag} — Precision-Recall ({best})")
    ax.legend()
    imgs["pr"] = save_fig(fig, f"{task_tag}_pr.png")

    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y, pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{task_tag} — Confusion ({best})")
    imgs["cm"] = save_fig(fig, f"{task_tag}_cm.png")

    fig, ax = plt.subplots(figsize=(6, 5))
    pt, pp = calibration_curve(y, proba, n_bins=8, strategy="quantile")
    ax.plot(pp, pt, "o-", color="#27ae60")
    ax.plot([0, 1], [0, 1], "--", color="#95a5a6")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"{task_tag} — Calibration ({best})")
    imgs["cal"] = save_fig(fig, f"{task_tag}_cal.png")

    # model comparison bar
    return imgs


def plot_model_bars(df_metrics, metric, title, fname):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    d = df_metrics.sort_values(metric)
    ax.barh(d["model"], d[metric], color="#34495e")
    ax.set_xlabel(metric)
    ax.set_title(title)
    for i, v in enumerate(d[metric].values):
        ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=8)
    return save_fig(fig, fname)


def plot_multiclass_cm(y, pred, labels, title, fname):
    fig, ax = plt.subplots(figsize=(7, 6))
    cm = confusion_matrix(y, pred, labels=labels)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.xticks(rotation=30, ha="right")
    return save_fig(fig, fname)


def run_survival(X, y_event, y_time):
    """Cox PH + RSF with 5-fold CV C-index."""
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index
    from sklearn.model_selection import KFold

    try:
        from sksurv.ensemble import RandomSurvivalForest
        from sksurv.util import Surv
        has_rsf = True
    except Exception:
        has_rsf = False

    # prepare design matrix once
    pre = make_preprocess(NUMERIC, CATEGORICAL)
    Xt = pre.fit_transform(X)
    feat = [f.replace("num__", "").replace("cat__", "") for f in pre.get_feature_names_out()]
    # drop constant / near-collinear columns lightly
    Xt_df = pd.DataFrame(Xt, columns=feat)
    # remove perfectly constant
    Xt_df = Xt_df.loc[:, Xt_df.std() > 1e-8]
    # shorten duplicate-ish names
    Xt_df.columns = [f"f{i}_{c[:40]}" for i, c in enumerate(Xt_df.columns)]

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    cox_coefs = None

    # Cox via lifelines on each fold
    c_scores = []
    print("  [M2S] CoxPH ...")
    for tr, te in kf.split(Xt_df):
        dtr = Xt_df.iloc[tr].copy()
        dtr["T"] = y_time.iloc[tr].values
        dtr["E"] = y_event.iloc[tr].values
        dte = Xt_df.iloc[te].copy()
        cph = CoxPHFitter(penalizer=0.1)
        try:
            cph.fit(dtr, duration_col="T", event_col="E", show_progress=False)
            risk = cph.predict_partial_hazard(dte)
            c = concordance_index(y_time.iloc[te], -risk, y_event.iloc[te])
            c_scores.append(c)
        except Exception as e:
            print("    Cox fold fail:", e)
    if c_scores:
        rows.append({"model": "CoxPH", "C_index_mean": float(np.mean(c_scores)), "C_index_std": float(np.std(c_scores))})
        # full fit for coefs
        dfull = Xt_df.copy()
        dfull["T"] = y_time.values
        dfull["E"] = y_event.values
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(dfull, duration_col="T", event_col="E", show_progress=False)
        cox_coefs = cph.summary[["coef", "exp(coef)", "p"]].sort_values("coef", key=np.abs, ascending=False).head(25)
        cox_coefs = cox_coefs.reset_index().rename(columns={"index": "feature", "covariate": "feature"})
        if "covariate" in cox_coefs.columns:
            cox_coefs = cox_coefs.rename(columns={"covariate": "feature"})

    if has_rsf:
        print("  [M2S] RSF ...")
        y_struct = Surv.from_arrays(y_event.astype(bool).values, y_time.values)
        rsf_scores = []
        for tr, te in kf.split(Xt_df):
            rsf = RandomSurvivalForest(
                n_estimators=200,
                min_samples_split=10,
                min_samples_leaf=6,
                max_features="sqrt",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
            rsf.fit(Xt_df.iloc[tr].values, y_struct[tr])
            rsf_scores.append(rsf.score(Xt_df.iloc[te].values, y_struct[te]))
        rows.append(
            {
                "model": "RandomSurvivalForest",
                "C_index_mean": float(np.mean(rsf_scores)),
                "C_index_std": float(np.std(rsf_scores)),
            }
        )

    return pd.DataFrame(rows), cox_coefs


def df_to_html_table(df: pd.DataFrame, digits=3) -> str:
    if df is None or len(df) == 0:
        return "<p><em>Yok</em></p>"
    show = df.copy()
    for c in show.columns:
        if pd.api.types.is_float_dtype(show[c]):
            show[c] = show[c].map(lambda x: f"{x:.{digits}f}" if pd.notna(x) else "")
    return show.to_html(index=False, classes="tbl", border=0, escape=True)


def img_tag(b64, alt=""):
    if not b64:
        return ""
    return f'<img class="plot" alt="{alt}" src="data:image/png;base64,{b64}" />'


def build_html(ctx: dict) -> str:
    params_json = json.dumps(PARAMS, indent=2, ensure_ascii=False)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>BreastPredict — Modelleme Raporu</title>
<style>
:root {{
  --bg:#f6f3ee; --ink:#1c1a17; --muted:#5c574f; --card:#fffdf9;
  --line:#e4ddd2; --accent:#8b3a3a; --accent2:#2f5d50; --chip:#efe8dc;
}}
* {{ box-sizing:border-box; }}
body {{
  margin:0; font-family:"Segoe UI", "Helvetica Neue", Arial, sans-serif;
  background:
    radial-gradient(1200px 600px at 10% -10%, #efe2d3 0%, transparent 60%),
    radial-gradient(900px 500px at 100% 0%, #d9e6e1 0%, transparent 55%),
    var(--bg);
  color:var(--ink); line-height:1.55;
}}
.wrap {{ max-width:1100px; margin:0 auto; padding:32px 20px 80px; }}
header {{
  padding:28px 0 10px; border-bottom:1px solid var(--line); margin-bottom:28px;
}}
.brand {{ font-size:13px; letter-spacing:.14em; text-transform:uppercase; color:var(--accent); font-weight:700; }}
h1 {{ font-size:clamp(28px,4vw,42px); margin:8px 0 6px; letter-spacing:-.02em; }}
.sub {{ color:var(--muted); max-width:70ch; }}
.meta {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:14px; }}
.chip {{ background:var(--chip); border:1px solid var(--line); border-radius:999px; padding:4px 12px; font-size:12px; color:var(--muted); }}
section {{
  background:var(--card); border:1px solid var(--line); border-radius:18px;
  padding:22px 22px 10px; margin:18px 0; box-shadow:0 10px 30px rgba(40,30,20,.04);
}}
h2 {{ margin:0 0 8px; font-size:22px; letter-spacing:-.01em; }}
h3 {{ margin:18px 0 8px; font-size:16px; color:var(--accent2); }}
p.lead {{ color:var(--muted); margin-top:0; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:12px; margin:14px 0 8px; }}
.kpi {{ background:#faf7f2; border:1px solid var(--line); border-radius:14px; padding:14px; }}
.kpi .k {{ font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:.06em; }}
.kpi .v {{ font-size:28px; font-weight:700; margin-top:4px; }}
.kpi .s {{ font-size:12px; color:var(--muted); }}
.tbl {{ width:100%; border-collapse:collapse; font-size:13px; margin:8px 0 16px; }}
.tbl th {{ text-align:left; background:#f3eee6; padding:8px 10px; border-bottom:1px solid var(--line); }}
.tbl td {{ padding:7px 10px; border-bottom:1px solid var(--line); vertical-align:top; }}
.tbl tr:hover td {{ background:#fbf8f3; }}
.plots {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:14px; margin:10px 0 18px; }}
.plot {{ width:100%; border:1px solid var(--line); border-radius:12px; background:#fff; }}
pre {{
  background:#1e1c19; color:#f4efe6; padding:14px; border-radius:12px; overflow:auto;
  font-size:12px; line-height:1.4;
}}
.note {{ background:#f8f1e8; border-left:4px solid var(--accent); padding:10px 12px; border-radius:0 10px 10px 0; color:var(--muted); font-size:13px; }}
footer {{ margin-top:28px; color:var(--muted); font-size:12px; text-align:center; }}
</style>
</head>
<body>
<div class="wrap">
<header>
  <div class="brand">BreastPredict · METABRIC klinik+patoloji</div>
  <h1>Metastaz risk, zaman ve bölge modelleme raporu</h1>
  <p class="sub">Internal validasyon: {N_SPLITS}-fold stratified CV. Özellik seti: klinik + PR + HER2 complete-case. Explainable AI: katsayılar, native importance ve SHAP.</p>
  <div class="meta">
    <span class="chip">Tarih: {now}</span>
    <span class="chip">n complete-case: {ctx['n_full']}</span>
    <span class="chip">Seed: {RANDOM_STATE}</span>
    <span class="chip">Kaynak: database.xlsx</span>
  </div>
</header>

<section>
  <h2>1. Çalışma özeti</h2>
  <p class="lead">Üç ana görev: (M1) uzak metastaz var/yok, (M2) zaman/risk (çok sınıflı + 5-yıl kümülatif + survival), (M3) kaba ilk uzak metastaz bölgesi.</p>
  <div class="grid">
    <div class="kpi"><div class="k">M1 en iyi AUC</div><div class="v">{ctx['m1_best_auc']:.3f}</div><div class="s">{ctx['m1_best']}</div></div>
    <div class="kpi"><div class="k">M2A en iyi Macro-F1</div><div class="v">{ctx['m2a_best_f1']:.3f}</div><div class="s">{ctx['m2a_best']}</div></div>
    <div class="kpi"><div class="k">M2B 5y AUC</div><div class="v">{ctx['m2b_best_auc']:.3f}</div><div class="s">{ctx['m2b_best']}</div></div>
    <div class="kpi"><div class="k">M2S en iyi C-index</div><div class="v">{ctx['m2s_best_c']:.3f}</div><div class="s">{ctx['m2s_best']}</div></div>
    <div class="kpi"><div class="k">M3 en iyi Macro-F1</div><div class="v">{ctx['m3_best_f1']:.3f}</div><div class="s">{ctx['m3_best']}</div></div>
  </div>
  <div class="note">Tedavi değişkenleri endikasyon karışıklığı taşıyabilir; sonuçlar prognostik/asosiyatif yorumlanmalı. External validasyon bu raporda yok — ayrı kohort gerekir.</div>
</section>

<section>
  <h2>2. Kohort ve etiketler</h2>
  {df_to_html_table(ctx['cohort_table'])}
  <h3>Sınıf dağılımları</h3>
  {df_to_html_table(ctx['label_table'])}
</section>

<section>
  <h2>3. Model hiperparametreleri</h2>
  <p class="lead">Tüm algoritmalarda kullanılan sabit parametreler (iç CV ile karşılaştırma adil olsun diye Optuna ile fold içi sızıntılı arama yapılmadı).</p>
  <pre>{params_json}</pre>
</section>

<section>
  <h2>4. M1 — Metastaz var / yok</h2>
  <p class="lead">n={ctx['m1_n']} · met=1 / met=0. Metrikler OOF (cross_val_predict) tahminlerinden.</p>
  {df_to_html_table(ctx['m1_metrics'])}
  <div class="plots">
    {img_tag(ctx['imgs'].get('m1_bars'), 'M1 bars')}
    {img_tag(ctx['imgs'].get('m1_roc'), 'ROC')}
    {img_tag(ctx['imgs'].get('m1_pr'), 'PR')}
    {img_tag(ctx['imgs'].get('m1_cm'), 'CM')}
    {img_tag(ctx['imgs'].get('m1_cal'), 'Calibration')}
  </div>
  <h3>Explainable AI — {ctx['m1_best']}</h3>
  <div class="plots">
    {img_tag(ctx['imgs'].get('m1_shap'), 'SHAP')}
    {img_tag(ctx['imgs'].get('m1_fi'), 'FI')}
    {img_tag(ctx['imgs'].get('m1_coef'), 'Coef')}
    {img_tag(ctx['imgs'].get('m1_perm'), 'Perm')}
  </div>
  {df_to_html_table(ctx.get('m1_xai_table'))}
</section>

<section>
  <h2>5. M2A — Zaman sınıfları (6 sınıf)</h2>
  <p class="lead">y1 / y2 / y3 / y4 / y5_plus / no_met_ge5y · n={ctx['m2a_n']}. Erken sansür çıkarıldı. Dengesiz sınıflar → Macro-F1 birincil metrik.</p>
  {df_to_html_table(ctx['m2a_metrics'])}
  <div class="plots">
    {img_tag(ctx['imgs'].get('m2a_bars'), 'bars')}
    {img_tag(ctx['imgs'].get('m2a_cm'), 'cm')}
    {img_tag(ctx['imgs'].get('m2a_shap'), 'shap')}
    {img_tag(ctx['imgs'].get('m2a_fi'), 'fi')}
  </div>
  {df_to_html_table(ctx.get('m2a_xai_table'))}
</section>

<section>
  <h2>6. M2B — 5 yıldan önce metastaz (kümülatif)</h2>
  <p class="lead">n={ctx['m2b_n']}. “5. yıldan önce olacak mı?” binary ufuk modeli.</p>
  {df_to_html_table(ctx['m2b_metrics'])}
  <div class="plots">
    {img_tag(ctx['imgs'].get('m2b_bars'), 'bars')}
    {img_tag(ctx['imgs'].get('m2b_roc'), 'roc')}
    {img_tag(ctx['imgs'].get('m2b_pr'), 'pr')}
    {img_tag(ctx['imgs'].get('m2b_cm'), 'cm')}
    {img_tag(ctx['imgs'].get('m2b_shap'), 'shap')}
    {img_tag(ctx['imgs'].get('m2b_fi'), 'fi')}
  </div>
  {df_to_html_table(ctx.get('m2b_xai_table'))}
</section>

<section>
  <h2>7. M2S — Survival (zaman-to-metastaz)</h2>
  <p class="lead">Olay=uzak metastaz, süre=TDR (gün). CoxPH penalizer=0.1; RSF n_estimators=200. Metrik: C-index (5-fold).</p>
  {df_to_html_table(ctx['m2s_metrics'])}
  <h3>CoxPH — |coef| en yüksek özellikler (hazard)</h3>
  {df_to_html_table(ctx.get('m2s_cox'))}
  <div class="plots">{img_tag(ctx['imgs'].get('m2s_cox'), 'cox')}</div>
</section>

<section>
  <h2>8. M3 — İlk uzak metastaz bölgesi (kaba)</h2>
  <p class="lead">Sınıflar: bone / visceral / multi_other (cns+other+multi birleşik). n={ctx['m3_n']}. İnce anatomik etiketler yetersiz olduğu için kaba grup kullanıldı.</p>
  {df_to_html_table(ctx['m3_metrics'])}
  <div class="plots">
    {img_tag(ctx['imgs'].get('m3_bars'), 'bars')}
    {img_tag(ctx['imgs'].get('m3_cm'), 'cm')}
    {img_tag(ctx['imgs'].get('m3_shap'), 'shap')}
    {img_tag(ctx['imgs'].get('m3_fi'), 'fi')}
  </div>
  {df_to_html_table(ctx.get('m3_xai_table'))}
</section>

<section>
  <h2>9. Yorum ve sınırlılıklar</h2>
  <ul>
    <li>Internal validasyon OOF/CV ile yapıldı; aynı kohortta external sayılmaz.</li>
    <li>PR/HER2 eksikliği complete-case’i ~1838’e düşürdü; bias riski vardır.</li>
    <li>Bölge modeli n=416 ve sınıf dengesizliği nedeniyle daha düşük performans beklenir.</li>
    <li>SHAP değerleri örneklem üzerinde (≤400) hesaplandı; göreli ağırlık / katkı sırası için uygundur.</li>
    <li>Sonraki adım: harici kohort ile external validation + kalibrasyon güncelleme.</li>
  </ul>
</section>

<footer>BreastPredict otomatik rapor · {now}</footer>
</div>
</body>
</html>
"""


def pick_xai_table(expl: dict):
    for k in ("shap", "coef", "fi", "perm"):
        if k in expl.get("tables", {}):
            return expl["tables"][k]
    return pd.DataFrame()


def main():
    print("Loading data...")
    df = load_prepared()
    print(f"Complete-case feature rows: {len(df)}")

    cohort_rows = [
        {"tanim": "Toplam database", "n": 3240},
        {"tanim": "Klinik+PR+HER2 complete-case", "n": len(df)},
        {"tanim": "M1 binary (hepsi)", "n": len(df)},
        {"tanim": "M2A exclusive 6-class", "n": int(df["ex_label"].isin(["y1","y2","y3","y4","y5_plus","no_met_ge5y"]).sum())},
        {"tanim": "M2B 5y cumulative usable", "n": int(df["met_5y"].notna().sum())},
        {"tanim": "M2S survival", "n": int(df["gun"].notna().sum())},
        {"tanim": "M3 site coarse", "n": int(df["site_label"].notna().sum())},
    ]
    cohort_table = pd.DataFrame(cohort_rows)

    label_table = pd.concat(
        [
            df["met"].value_counts().rename_axis("label").reset_index(name="n").assign(task="M1_met"),
            df.loc[df["ex_label"].isin(["y1","y2","y3","y4","y5_plus","no_met_ge5y"]), "ex_label"]
            .value_counts()
            .rename_axis("label")
            .reset_index(name="n")
            .assign(task="M2A"),
            df["met_5y"].value_counts().rename_axis("label").reset_index(name="n").assign(task="M2B_5y"),
            df["site_label"].value_counts().rename_axis("label").reset_index(name="n").assign(task="M3"),
        ],
        ignore_index=True,
    )

    imgs = {}
    ctx = {"imgs": imgs, "n_full": len(df), "cohort_table": cohort_table, "label_table": label_table}

    # -------- M1 --------
    print("\n=== M1 Binary metastasis ===")
    X1 = df[FEATURES]
    y1 = df["met"].astype(int)
    m1_models = get_binary_models()
    m1_metrics, m1_oof, _ = evaluate_binary("M1", X1, y1, m1_models)
    m1_metrics.to_excel(TAB / "m1_metrics.xlsx", index=False)
    best1 = m1_metrics.iloc[0]["model"]
    ctx["m1_metrics"] = m1_metrics
    ctx["m1_best"] = best1
    ctx["m1_best_auc"] = float(m1_metrics.iloc[0]["AUC_ROC"])
    ctx["m1_n"] = len(y1)
    imgs["m1_bars"] = plot_model_bars(m1_metrics, "AUC_ROC", "M1 — AUC-ROC karşılaştırması", "m1_bars.png")
    d1 = plot_binary_diagnostics(y1, m1_oof, best1, "m1")
    imgs.update({f"m1_{k}": v for k, v in d1.items()})
    _, expl1 = fit_best_and_explain(X1, y1, best1, m1_models, task="m1")
    for k, v in expl1["images"].items():
        imgs[f"m1_{k}"] = v
    ctx["m1_xai_table"] = pick_xai_table(expl1)

    # -------- M2A --------
    print("\n=== M2A Exclusive time classes ===")
    mask2a = df["ex_label"].isin(["y1", "y2", "y3", "y4", "y5_plus", "no_met_ge5y"])
    d2a = df.loc[mask2a].copy()
    class_order = ["y1", "y2", "y3", "y4", "y5_plus", "no_met_ge5y"]
    # drop classes with too few if any after complete-case — keep all present
    present = [c for c in class_order if (d2a["ex_label"] == c).sum() >= 5]
    d2a = d2a[d2a["ex_label"].isin(present)]
    X2a = d2a[FEATURES]
    y2a = d2a["ex_label"].astype(str)
    m2a_models = get_multiclass_models(n_classes=len(present))
    m2a_metrics, m2a_oof, _, _ = evaluate_multiclass("M2A", X2a, y2a, m2a_models, present)
    m2a_metrics.to_excel(TAB / "m2a_metrics.xlsx", index=False)
    best2a = m2a_metrics.iloc[0]["model"]
    ctx["m2a_metrics"] = m2a_metrics
    ctx["m2a_best"] = best2a
    ctx["m2a_best_f1"] = float(m2a_metrics.iloc[0]["Macro_F1"])
    ctx["m2a_n"] = len(y2a)
    imgs["m2a_bars"] = plot_model_bars(m2a_metrics, "Macro_F1", "M2A — Macro-F1", "m2a_bars.png")
    imgs["m2a_cm"] = plot_multiclass_cm(y2a, m2a_oof[best2a]["pred"], present, f"M2A CM ({best2a})", "m2a_cm.png")
    _, expl2a = fit_best_and_explain(X2a, y2a, best2a, m2a_models, task="m2a", class_order=present)
    for k, v in expl2a["images"].items():
        imgs[f"m2a_{k}"] = v
    ctx["m2a_xai_table"] = pick_xai_table(expl2a)

    # -------- M2B --------
    print("\n=== M2B 5-year cumulative ===")
    d2b = df.dropna(subset=["met_5y"]).copy()
    X2b = d2b[FEATURES]
    y2b = d2b["met_5y"].astype(int)
    m2b_models = get_binary_models()
    m2b_metrics, m2b_oof, _ = evaluate_binary("M2B", X2b, y2b, m2b_models)
    m2b_metrics.to_excel(TAB / "m2b_metrics.xlsx", index=False)
    best2b = m2b_metrics.iloc[0]["model"]
    ctx["m2b_metrics"] = m2b_metrics
    ctx["m2b_best"] = best2b
    ctx["m2b_best_auc"] = float(m2b_metrics.iloc[0]["AUC_ROC"])
    ctx["m2b_n"] = len(y2b)
    imgs["m2b_bars"] = plot_model_bars(m2b_metrics, "AUC_ROC", "M2B — AUC-ROC", "m2b_bars.png")
    d2 = plot_binary_diagnostics(y2b, m2b_oof, best2b, "m2b")
    imgs.update({f"m2b_{k}": v for k, v in d2.items()})
    _, expl2b = fit_best_and_explain(X2b, y2b, best2b, m2b_models, task="m2b")
    for k, v in expl2b["images"].items():
        imgs[f"m2b_{k}"] = v
    ctx["m2b_xai_table"] = pick_xai_table(expl2b)

    # -------- M2S --------
    print("\n=== M2S Survival ===")
    d2s = df.dropna(subset=["gun", "met"]).copy()
    X2s = d2s[FEATURES]
    m2s_metrics, cox_coefs = run_survival(X2s, d2s["met"].astype(int), d2s["gun"].astype(float))
    m2s_metrics.to_excel(TAB / "m2s_metrics.xlsx", index=False)
    if cox_coefs is not None:
        cox_coefs.to_excel(TAB / "m2s_cox_coefs.xlsx", index=False)
        # plot
        feat_col = "feature" if "feature" in cox_coefs.columns else cox_coefs.columns[0]
        fig, ax = plt.subplots(figsize=(8, 7))
        plot_df = cox_coefs.head(20).iloc[::-1]
        ax.barh(plot_df[feat_col].astype(str), plot_df["coef"], color="#8b3a3a")
        ax.set_xlabel("log-hazard coefficient")
        ax.set_title("CoxPH — top |coef| features")
        imgs["m2s_cox"] = save_fig(fig, "m2s_cox.png")
    ctx["m2s_metrics"] = m2s_metrics
    ctx["m2s_cox"] = cox_coefs if cox_coefs is not None else pd.DataFrame()
    if len(m2s_metrics):
        best_idx = m2s_metrics["C_index_mean"].idxmax()
        ctx["m2s_best"] = m2s_metrics.loc[best_idx, "model"]
        ctx["m2s_best_c"] = float(m2s_metrics.loc[best_idx, "C_index_mean"])
    else:
        ctx["m2s_best"] = "NA"
        ctx["m2s_best_c"] = 0.0

    # -------- M3 --------
    print("\n=== M3 Site ===")
    d3 = df.dropna(subset=["site_label"]).copy()
    site_order = [c for c in ["bone", "visceral", "multi_other"] if (d3["site_label"] == c).sum() >= 10]
    d3 = d3[d3["site_label"].isin(site_order)]
    X3 = d3[FEATURES]
    y3 = d3["site_label"].astype(str)
    m3_models = get_multiclass_models(n_classes=len(site_order))
    m3_metrics, m3_oof, _, _ = evaluate_multiclass("M3", X3, y3, m3_models, site_order)
    m3_metrics.to_excel(TAB / "m3_metrics.xlsx", index=False)
    best3 = m3_metrics.iloc[0]["model"]
    ctx["m3_metrics"] = m3_metrics
    ctx["m3_best"] = best3
    ctx["m3_best_f1"] = float(m3_metrics.iloc[0]["Macro_F1"])
    ctx["m3_n"] = len(y3)
    imgs["m3_bars"] = plot_model_bars(m3_metrics, "Macro_F1", "M3 — Macro-F1", "m3_bars.png")
    imgs["m3_cm"] = plot_multiclass_cm(y3, m3_oof[best3]["pred"], site_order, f"M3 CM ({best3})", "m3_cm.png")
    _, expl3 = fit_best_and_explain(X3, y3, best3, m3_models, task="m3", class_order=site_order)
    for k, v in expl3["images"].items():
        imgs[f"m3_{k}"] = v
    ctx["m3_xai_table"] = pick_xai_table(expl3)

    # save summary json
    summary = {
        "m1_best": ctx["m1_best"],
        "m1_auc": ctx["m1_best_auc"],
        "m2a_best": ctx["m2a_best"],
        "m2a_macro_f1": ctx["m2a_best_f1"],
        "m2b_best": ctx["m2b_best"],
        "m2b_auc": ctx["m2b_best_auc"],
        "m2s_best": ctx["m2s_best"],
        "m2s_cindex": ctx["m2s_best_c"],
        "m3_best": ctx["m3_best"],
        "m3_macro_f1": ctx["m3_best_f1"],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    html = build_html(ctx)
    report_path = ROOT / "BreastPredict_Rapor.html"
    report_path.write_text(html, encoding="utf-8")
    print(f"\nReport written: {report_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
