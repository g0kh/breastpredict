"""Hızlı kanıt: (A) class_weight kalibrasyon etkisi, (B) doğru sağkalım çerçevesi."""
import sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.metrics import roc_auc_score, brier_score_loss
from sksurv.util import Surv
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_ipcw, cumulative_dynamic_auc, brier_score

Y = 365.25
s5 = pd.read_csv(sys.argv[1], sep="\t")
db = pd.read_excel(sys.argv[2], sheet_name="hastalar")

# ---------------- A: kullanıcının M1 kurgusu, LR balanced vs ağırlıksız ----------------
NUM = ["yas", "tumor_boyutu_mm", "pozitif_ln", "grade"]
CAT = ["menopoz", "histoloji", "er", "pr", "her2", "cerrahi", "kemoterapi", "hormon_tedavisi", "radyoterapi"]
d = db.copy()
for c in ["kemoterapi", "hormon_tedavisi", "radyoterapi", "histoloji"]:
    vc = d[c].fillna("MISSING").astype(str).value_counts(); keep = set(vc[vc >= 25].index)
    d[c] = d[c].fillna("MISSING").astype(str).map(lambda x: x if x in keep else "OTHER")
d = d.dropna(subset=NUM + CAT)
X, y = d[NUM + CAT], d["uzak_metastaz"].astype(int)
pre = ColumnTransformer([("n", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), NUM),
                         ("c", OneHotEncoder(handle_unknown="ignore"), CAT)])
cv = StratifiedKFold(5, shuffle=True, random_state=42)
print(f"A) M1 kurgusu n={len(y)}  olay oranı={y.mean():.3f}  null Brier={y.mean()*(1-y.mean()):.4f}")
for cw in ["balanced", None]:
    p = cross_val_predict(Pipeline([("p", pre), ("m", LogisticRegression(C=1.0, max_iter=3000, class_weight=cw))]),
                          X, y, cv=cv, method="predict_proba")[:, 1]
    lp = np.log(p / (1 - p))
    slope = LogisticRegression(C=1e6).fit(lp.reshape(-1, 1), y).coef_[0, 0]
    print(f"   LR class_weight={str(cw):9s} AUC={roc_auc_score(y, p):.4f}  Brier={brier_score_loss(y, p):.4f}  "
          f"ort.tahmin={p.mean():.3f}")

# ---------------- B: temiz kohort + DRFI + sağkalım metrikleri ----------------
s = s5.copy()
s = s[(s["T"] > 0) & s["Death"].notna() & (s["Stage"] != 4) & ~s["Histological.Type"].isin(["BENIGN", "PHYL", "DCIS"])]
s = s.merge(db[["hasta_id", "pr", "her2"]], left_on="METABRIC.ID", right_on="hasta_id", how="left")
ev = ((s.DR == 1) | (s.DeathBreast == 1)).values            # STEEP DRFI: uzak nüks veya meme kanseri ölümü
tm = np.where(s.DR == 1, s.TDR, s["T"]).astype(float)
tm[tm <= 0] = 1.0
yS = Surv.from_arrays(ev, tm)
h = s["Histological.Type"].fillna("")
F = pd.DataFrame({
    "age": s["Age.At.Diagnosis"],
    "log_size": np.log(s["Size"].clip(lower=1)),
    "ln": np.log1p(s["Lymph.Nodes.Positive"].clip(upper=20)),
    "grade": s["Grade"],
    "er": (s["ER.Status"] == "pos").astype(float),
    "pr": s["pr"].map({"Positive": 1, "Negative": 0}),
    "her2": s["her2"].map({"Positive": 1, "Negative": 0}),
    "post": s["Inferred.Menopausal.State"].map({"post": 1, "pre": 0}),
    "lobular": h.str.contains("ILC").astype(float),
    "ct": (s["CT"] != "NO/NA").astype(float),
    "ht": (s["HT"] != "NO/NA").astype(float),
})
F["pr_miss"] = F["pr"].isna().astype(float); F["her2_miss"] = F["her2"].isna().astype(float)
F["er_x_logt_proxy"] = 0.0  # yer tutucu (kullanılmıyor)
F = F.drop(columns="er_x_logt_proxy")
cohort = s["Cohort"].values
print(f"\nB) Temiz kohort n={len(s)}  DRFI olayı={ev.sum()}  (evre4/benign/DCIS/T=0 dışlandı, PR/HER2 eksikleri impute)")

def imp_scale(Xtr, Xte):
    im = IterativeImputer(max_iter=10, random_state=42).fit(Xtr)
    a, b = im.transform(Xtr), im.transform(Xte)
    for j, c in enumerate(Xtr.columns):
        if c in ("pr", "her2", "grade", "post"):
            a[:, j] = np.clip(np.round(a[:, j]), 0 if c != "grade" else 1, 1 if c != "grade" else 3)
            b[:, j] = np.clip(np.round(b[:, j]), 0 if c != "grade" else 1, 1 if c != "grade" else 3)
    sc = StandardScaler().fit(a)
    return sc.transform(a), sc.transform(b)

class ERStratCox:
    """ER+ ve ER- için ayrı Cox (orantısız tehlike sorununa basit çözüm)."""
    def fit(self, X, y, er):
        self.m = {g: CoxPHSurvivalAnalysis(alpha=1.0).fit(X[er == g], y[er == g]) for g in (0, 1)}
        return self
    def risk_at(self, X, er, t):
        out = np.zeros(len(X))
        for g in (0, 1):
            idx = er == g
            if idx.any():
                fns = self.m[g].predict_survival_function(X[idx])
                out[idx] = [1 - f(t) for f in fns]
        return out

def evaluate(split_iter, label):
    res = {k: [] for k in ["Cox", "Cox_ER_ayrı", "RSF"]}
    for tr, te in split_iter:
        Xtr, Xte = imp_scale(F.iloc[tr], F.iloc[te])
        ytr, yte = yS[tr], yS[te].copy()
        cap = ytr["time"][ytr["event"] == 0].max() - 1      # test takibini eğitimin sansür ufkunda kes (IPCW için)
        over = yte["time"] > cap
        yte["event"][over] = False; yte["time"][over] = cap
        er_tr, er_te = F["er"].values[tr].astype(int), F["er"].values[te].astype(int)
        tmax = min(ytr["time"].max(), yte["time"].max())
        times = np.array([5 * Y, 10 * Y]); times = times[times < tmax]
        preds = {}
        cox = CoxPHSurvivalAnalysis(alpha=1.0).fit(Xtr, ytr)
        preds["Cox"] = {t: np.array([1 - f(t) for f in cox.predict_survival_function(Xte)]) for t in times}
        ec = ERStratCox().fit(Xtr, ytr, er_tr)
        preds["Cox_ER_ayrı"] = {t: ec.risk_at(Xte, er_te, t) for t in times}
        rsf = RandomSurvivalForest(n_estimators=300, min_samples_leaf=15, max_features="sqrt", n_jobs=-1, random_state=42).fit(Xtr, ytr)
        sf = rsf.predict_survival_function(Xte)
        preds["RSF"] = {t: np.array([1 - f(t) for f in sf]) for t in times}
        for k, pt in preds.items():
            r = {}
            for t, p in pt.items():
                r[f"AUC{int(round(t / Y))}y"] = cumulative_dynamic_auc(ytr, yte, p, [t])[0][0]
                r[f"Brier{int(round(t / Y))}y"] = brier_score(ytr, yte, 1 - p, [t])[1][0]
            r["UnoC_10y"] = concordance_index_ipcw(ytr, yte, pt[times[-1]], tau=times[-1])[0]
            res[k].append(r)
    print(f"\n   {label}")
    for k, rows in res.items():
        df = pd.DataFrame(rows)
        print("   " + f"{k:12s} " + "  ".join(f"{c}={df[c].mean():.3f}±{df[c].std():.3f}" for c in df.columns))

if "--only-loco" not in sys.argv:
    evaluate(KFold(5, shuffle=True, random_state=42).split(F), "5-katlı CV (ortalama±SD)")
evaluate(((np.where(cohort != c)[0], np.where(cohort == c)[0]) for c in sorted(set(cohort))),
         "Kohort-dışarıda-bırak (internal-external, 5 merkez)")
