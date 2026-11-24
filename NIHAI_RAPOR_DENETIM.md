# BreastPredict — Nihai Denetim Raporu (LLM / Peer Review için)

**Proje:** breastpredict (METABRIC klinik + patoloji reseptörleri ile uzak metastaz modelleme)  
**Depo:** https://github.com/g0kh/breastpredict  
**Yerel yol:** `C:\Users\egkah\Desktop\breastpredict`  
**Rapor tarihi:** 2026-11-24  
**Amaç:** Yapılan tüm işlemlerin, veri setinin, etiketlerin, algoritmaların, hiperparametrelerin, dizilimlerin ve sınırlılıkların başka bir LLM / hakem tarafından doğrulanabilir biçimde belgelenmesi.

---

## 0. Yönetici özeti

| Madde | İçerik |
|---|---|
| Kaynak veri | METABRIC (Rueda 2019 Supp Table 5/7 + cBioPortal klinik sample) |
| Ana tablo | `database.xlsx` → 3240 hasta, 19 sütun |
| Modelleme kohortu | Klinik + PR + HER2 **complete-case** ≈ **1838** hasta |
| Görevler | M1 binary met; M2A 6-sınıf zaman; M2B 5y kümülatif; M2S survival; M3 kaba bölge |
| Validasyon | **Internal only:** Stratified 5-fold CV, OOF (`cross_val_predict`); seed=42 |
| External validation | **Yapılmadı** (ayrı kohort yok) |
| Report 1 en iyi | M1 AUC 0.681 (RF); M2B AUC 0.750 (SVM); M2S C-index 0.686 (RSF) |
| Report 2 en iyi | M1 AUC 0.686 (TOP7 + HybridVote); M2B AUC 0.758 (FULL + Cat_reg) |
| Ana sonuç | Klinik-only ile AUC tavanı ~0.68–0.76; moleküler veri olmadan büyük sıçrama beklenmez |

**HTML çıktılar:**
- `BreastPredict_Rapor.html` (Report 1 — tüm modeller + XAI)
- `report2.html` (Report 2 — özellik sadeleştirme + hibrit)

**Tekrarlanabilir scriptler:**
- `run_full_modeling.py`
- `run_report2.py`
- `scan_cohorts.py`

---

## 1. Veri seti

### 1.1 Kaynaklar

| Kaynak | Rol |
|---|---|
| `brca_metabric/NIHMS1520488-supplement-Supp_Table_5.txt` | Omurga klinik + `DR` (uzak met olayı) + `TDR` (gün) |
| `brca_metabric/NIHMS1520488-supplement-Supp_Table_7.txt` | Nüks / ilk uzak metastaz bölgesi |
| `brca_metabric/brca_metabric/data_clinical_patient.txt` | cBioPortal hasta klinik |
| `brca_metabric/brca_metabric/data_clinical_sample.txt` | **PR_STATUS**, **HER2_STATUS** |
| Repoda olmayan | mRNA, metilasyon, CNA, `brca_metabric.tar.gz` (GitHub boyutu) |

### 1.2 `database.xlsx` şeması (19 sütun)

| Sütun | Anlam | Kaynak | Not |
|---|---|---|---|
| `hasta_id` | METABRIC ID | Table 5 | MB-… |
| `yas` | Tanı yaşı | Table 5 | yıl |
| `menopoz` | İnferred menopausal state | Table 5 | pre/post |
| `tumor_boyutu_mm` | Primer tümör boyutu | Table 5 | mm |
| `pozitif_ln` | Pozitif LN sayısı | Table 5 | adet |
| `grade` | Histolojik grade | Table 5 | 1/2/3 |
| `histoloji` | Histolojik tip | Table 5 | IDC, ILC, … |
| `er` | Östrojen reseptörü | Table 5 | pos/neg |
| `pr` | Progesteron reseptörü | cBioPortal sample | Positive/Negative; örtüşmezse boş |
| `her2` | HER2 klinik status | cBioPortal sample | SNP6 kopya sayısı **kullanılmadı** |
| `cerrahi` | Meme cerrahisi | Table 5 | MASTECTOMY / BREAST CONSERVING |
| `kemoterapi` | KT rejim kodu | Table 5 | NO/NA, CMF, AC, … |
| `hormon_tedavisi` | HT | Table 5 | TAM, AI, NO/NA, … |
| `radyoterapi` | RT | Table 5 | CW, CW-NODAL, NO/NA, … |
| `uzak_metastaz` | Olay | Table 5 `DR` | 1/0 |
| `uzak_metastaz_gun` | Süre | Table 5 `TDR` | met=0 iken **sansür süresi** |
| `uzak_metastaz_ay` | Türetilmiş | TDR/30 | |
| `ilk_uzak_metastaz_bolgesi` | İlk uzak bölge | Table 7 | 618 hastada dolu |
| `nuks_oykusu_tam` | Complete recurrence history | Table 5 | YES/NO |

**Açılmayan sütunlar:** Ki-67, LVI (pakette yok).

### 1.3 Ham kohort sayıları (tüm database)

| Tanım | n |
|---|---:|
| Toplam hasta | 3240 |
| Uzak met var (DR=1) | 1080 |
| Uzak met yok (DR=0) | 2160 |
| İlk uzak bölge dolu | 618 |
| PR veya HER2 eksik | ~1260 |
| Klinik çekirdek (PR/HER2 hariç) dolu | 2962 |
| Klinik + PR + HER2 complete-case | **1838** (modellemede kullanılan ana set) |

### 1.4 Ön işleme (modelleme pipeline)

1. Boş string → NaN  
2. Sayısal alanlar `to_numeric(..., errors='coerce')`  
3. Nadir kategoriler (`kemoterapi`, `hormon_tedavisi`, `radyoterapi`, `histoloji`): frekans &lt; 25 → `OTHER`  
4. Complete-case: 13 özellikte NaN olan satırlar düşülür  
5. Kategorikler: OneHotEncoder (`handle_unknown='ignore'`)  
6. Sayısal: median impute + StandardScaler  
7. Pipeline: `ColumnTransformer` → classifier  

**Modellemede kullanılan 13 özellik (FULL):**  
`yas, tumor_boyutu_mm, pozitif_ln, grade, menopoz, histoloji, er, pr, her2, cerrahi, kemoterapi, hormon_tedavisi, radyoterapi`

---

## 2. Modelleme görevleri ve etiketler

### 2.1 M1 — Metastaz var/yok (binary)

- **Hedef:** `uzak_metastaz` ∈ {0,1}  
- **n (complete-case):** 1838 (met1≈573, met0≈1265)  
- **Birincil metrik:** AUC-ROC  
- **Ek metrikler:** AUC-PR, Accuracy, Balanced Accuracy, F1, Brier  

### 2.2 M2A — Zaman sınıfları (6 sınıf, birbirini dışlayan)

- **Etiket kuralı (1 yıl = 365.25 gün):**  
  - met=1 & TDR&lt;1y → `y1`  
  - met=1 & 1–2y → `y2`  
  - met=1 & 2–3y → `y3`  
  - met=1 & 3–4y → `y4`  
  - met=1 & ≥4y → `y5_plus`  
  - met=0 & takip ≥5y → `no_met_ge5y`  
  - met=0 & takip &lt;5y → `censor_lt5y` (**çıkarıldı**)  
- **Complete-case usable n:** ≈1662  
  - no_met_ge5y 1089; y5_plus 238; y2 112; y3 100; y4 70; y1 53  
- **Birincil metrik:** Macro-F1 (dengesizlik nedeniyle)  
- **Ek:** Weighted-F1, Balanced Acc, AUC OvR macro  

### 2.3 M2B — 5 yıldan önce metastaz (kümülatif binary)

- **pozitif:** met=1 ve TDR &lt; 5y  
- **negatif:** (met=0 ve FU≥5y) VEYA (met=1 ve TDR≥5y)  
- **hariç:** met=0 ve FU&lt;5y  
- **n:** ≈1662  
- **Birincil metrik:** AUC-ROC  

### 2.4 M2S — Survival (zaman-to-metastaz)

- **Olay:** `uzak_metastaz`  
- **Süre:** `uzak_metastaz_gun`  
- **Metrik:** Harrell C-index (5-fold)  
- **Modeller:** CoxPH (lifelines, penalizer=0.1), RandomSurvivalForest (scikit-survival)  

### 2.5 M3 — İlk uzak metastaz bölgesi (kaba)

- Ham bölge 618; complete-case + özellik ≈416  
- Kaba etiket: `bone` / `visceral` / `multi_other` (cns+other+multi birleşik)  
- **Birincil metrik:** Macro-F1  

---

## 3. Validasyon protokolü

| Öğe | Değer |
|---|---|
| Tip | Internal validation |
| Yöntem | StratifiedKFold, n_splits=**5**, shuffle=True |
| random_state | **42** |
| Tahmin | `sklearn.model_selection.cross_val_predict` (OOF) |
| Binary eşik | 0.5 (raporlarda; eşik optimizasyonu yapılmadı) |
| Nested CV / Optuna | **Yapılmadı** (Report 1 sabit HP; Report 2 sabit regularize HP) |
| External validation | **Yok** |
| Train/test tek split | Kullanılmadı (tüm metrikler CV-OOF) |

**Önemli:** Aynı METABRIC kohortunda CV = internal; genel klinik performans iddiası için external gerekir.

---

## 4. Report 1 — Algoritmalar, hiperparametreler, dizilim

### 4.1 Binary modeller (M1, M2B) — hiperparametreler

| Algoritma | Ana parametreler |
|---|---|
| LogisticRegression | C=1.0, penalty=l2, solver=lbfgs, max_iter=2000, class_weight=balanced |
| RandomForest | n_estimators=400, max_depth=12, min_samples_leaf=5, class_weight=balanced_subsample |
| ExtraTrees | n_estimators=400, max_depth=14, min_samples_leaf=4, class_weight=balanced |
| HistGBM | max_depth=8, learning_rate=0.06, max_iter=300, l2_regularization=0.1, class_weight=balanced |
| XGBoost | n_estimators=350, max_depth=5, lr=0.05, subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0, tree_method=hist |
| LightGBM | n_estimators=400, lr=0.05, num_leaves=31, subsample=0.9, colsample_bytree=0.8, class_weight=balanced |
| CatBoost | iterations=400, depth=6, lr=0.05, l2_leaf_reg=3.0, auto_class_weights=Balanced |
| SVM_RBF | C=1.0, kernel=rbf, gamma=scale, probability=True, class_weight=balanced |

### 4.2 Survival hiperparametreleri

| Model | Parametreler |
|---|---|
| CoxPH | penalizer=0.1, l1_ratio=0.0 (lifelines) |
| RSF | n_estimators=200, min_samples_split=10, min_samples_leaf=6, max_features=sqrt |

### 4.3 M1 sonuç dizilimi (AUC-ROC ↓)

| Sıra | Model | AUC_ROC | AUC_PR | Accuracy | F1 | Brier |
|---:|---|---:|---:|---:|---:|---:|
| 1 | RandomForest | **0.6807** | 0.4917 | 0.6834 | 0.5126 | 0.2132 |
| 2 | LogisticRegression | 0.6783 | 0.4852 | 0.6638 | 0.5072 | 0.2242 |
| 3 | SVM_RBF | 0.6779 | 0.4682 | 0.6986 | 0.3177 | 0.1962 |
| 4 | ExtraTrees | 0.6689 | 0.4780 | 0.6545 | 0.5066 | 0.2221 |
| 5 | XGBoost | 0.6565 | 0.4739 | 0.6877 | 0.3801 | 0.2088 |
| 6 | CatBoost | 0.6561 | 0.4670 | 0.6659 | 0.4858 | 0.2216 |
| 7 | HistGBM | 0.6531 | 0.4744 | 0.6632 | 0.4794 | 0.2274 |
| 8 | LightGBM | 0.6434 | 0.4639 | 0.6621 | 0.4651 | 0.2372 |

### 4.4 M2B sonuç dizilimi (AUC-ROC ↓)

| Sıra | Model | AUC_ROC | AUC_PR | Accuracy | F1 |
|---:|---|---:|---:|---:|---:|
| 1 | SVM_RBF | **0.7497** | 0.4562 | 0.7822 | 0.3092 |
| 2 | CatBoost | 0.7464 | 0.5030 | 0.7569 | 0.4753 |
| 3 | LogisticRegression | 0.7459 | 0.4831 | 0.7142 | 0.4952 |
| 4 | RandomForest | 0.7421 | 0.4946 | 0.7383 | 0.4803 |
| 5 | XGBoost | 0.7366 | 0.4868 | 0.7912 | 0.4089 |
| 6 | ExtraTrees | 0.7226 | 0.4652 | 0.7268 | 0.4853 |
| 7 | LightGBM | 0.7123 | 0.4533 | 0.7467 | 0.4409 |
| 8 | HistGBM | 0.7117 | 0.4630 | 0.7377 | 0.4536 |

### 4.5 M2A dizilimi (Macro-F1 ↓)

| Sıra | Model | Macro_F1 | Weighted_F1 | Acc | AUC_OvR |
|---:|---|---:|---:|---:|---:|
| 1 | RandomForest | **0.2271** | 0.5154 | 0.4886 | 0.6281 |
| 2 | HistGBM | 0.2208 | 0.5426 | 0.5578 | 0.5943 |
| 3 | CatBoost | 0.2207 | 0.4805 | 0.4362 | 0.6099 |
| 4 | LightGBM | 0.2035 | 0.5383 | 0.5728 | 0.6084 |
| 5 | XGBoost | 0.1880 | 0.5478 | 0.6318 | 0.6278 |
| 6 | LogisticRegression | 0.1825 | 0.3927 | 0.3231 | 0.6115 |

**Yorum:** 6-sınıf zaman görevi bu veri ile zayıf; y1 n≈53.

### 4.6 M2S dizilimi (C-index ↓)

| Sıra | Model | C_index_mean | C_index_std |
|---:|---|---:|---:|
| 1 | RandomSurvivalForest | **0.6859** | 0.0402 |
| 2 | CoxPH | 0.6818 | 0.0277 |

### 4.7 M3 dizilimi (Macro-F1 ↓)

| Sıra | Model | Macro_F1 | Acc | AUC_OvR |
|---:|---|---:|---:|---:|
| 1 | RandomForest | **0.4295** | 0.4623 | 0.5916 |
| 2 | LogisticRegression | 0.3977 | 0.4121 | 0.5848 |
| 3 | CatBoost | 0.3830 | 0.4095 | 0.5582 |
| 4 | LightGBM | 0.3769 | 0.3945 | 0.5416 |
| 5 | HistGBM | 0.3695 | 0.3920 | 0.5453 |
| 6 | XGBoost | 0.3692 | 0.4095 | 0.5503 |

### 4.8 Report 1 XAI

- En iyi modele SHAP (TreeExplainer / Explainer) veya katsayı / permutation importance  
- SHAP örneklem boyutu ≤400  
- Figürler HTML’e base64 gömülü  

---

## 5. Report 2 — Özellik katkısı, sadeleştirme, hibritler

### 5.1 Amaç

1. Permutation importance (ΔAUC) ile özellik katkısını ölçmek  
2. Az özellikli setler denemek (noise ↓, overfit riski ↓)  
3. Daha güçlü regularizasyon  
4. Soft voting / stacking hibritleri  
5. Report 1 baseline ile ΔAUC karşılaştırması  

### 5.2 Permutation importance yöntemi

- Model: RF (max_depth=6, min_samples_leaf=10, n_estimators=350, balanced_subsample)  
- Scoring: `roc_auc`  
- n_repeats=8  
- Ham sütun düzeyinde (one-hot aggregate değil; pipeline üzerinden kolon permute)

### 5.3 M1 özellik katkısı (ΔAUC ↓)

| Sıra | Özellik | importance_mean | std |
|---:|---|---:|---:|
| 1 | pozitif_ln | 0.0483 | 0.0064 |
| 2 | tumor_boyutu_mm | 0.0352 | 0.0022 |
| 3 | radyoterapi | 0.0246 | 0.0053 |
| 4 | kemoterapi | 0.0228 | 0.0045 |
| 5 | yas | 0.0200 | 0.0030 |
| 6 | cerrahi | 0.0153 | 0.0027 |
| 7 | grade | 0.0127 | 0.0020 |
| 8 | her2 | 0.0097 | 0.0028 |
| 9 | pr | 0.0094 | 0.0012 |
| 10 | menopoz | 0.0077 | 0.0018 |
| 11 | hormon_tedavisi | 0.0066 | 0.0006 |
| 12 | histoloji | 0.0050 | 0.0005 |
| 13 | er | 0.0042 | 0.0006 |

**Uyarı:** KT/RT yüksek katkı gösterebilir; bu genellikle **endikasyon confounded** (yüksek risklilere tedavi) anlamına gelir, nedensel koruma değildir.

### 5.4 M2B özellik katkısı (ΔAUC ↓)

| Sıra | Özellik | importance_mean |
|---:|---|---:|
| 1 | tumor_boyutu_mm | 0.0592 |
| 2 | pozitif_ln | 0.0440 |
| 3 | pr | 0.0188 |
| 4 | yas | 0.0187 |
| 5 | grade | 0.0168 |
| 6 | kemoterapi | 0.0158 |
| 7 | cerrahi | 0.0124 |
| 8 | radyoterapi | 0.0109 |
| 9 | her2 | 0.0107 |
| 10 | hormon_tedavisi | 0.0075 |
| 11 | er | 0.0070 |
| 12 | histoloji | 0.0063 |
| 13 | menopoz | 0.0039 |

### 5.5 Denenen özellik setleri

| Set | n_feat | İçerik özeti |
|---|---:|---|
| FULL_13 | 13 | Tüm özellikler |
| NO_TX | 10 | KT/HT/RT çıkarılmış |
| CORE8 | 8 | yas, boyut, LN, grade, er, pr, her2, menopoz |
| CONSENSUS7 | 7 | M1+M2B rank ortalaması top7 |
| CONSENSUS5 | 5 | consensus top5 |
| TOP7_perm / TOP5_perm | 7/5 | Göreve özel permutation top |
| POS_only | değişken | importance_mean &gt; 0 olanlar |

### 5.6 Report 2 regularize / lean hiperparametreler

| Model | Parametreler |
|---|---|
| LR_L2_strong | C=0.3, class_weight=balanced |
| RF_shallow | n_estimators=500, max_depth=6, min_samples_leaf=12, max_features=sqrt |
| HistGBM_reg | max_depth=4, lr=0.04, max_iter=250, l2=1.0, min_samples_leaf=20 |
| XGB_reg | max_depth=3, lr=0.04, reg_lambda=2, reg_alpha=0.5, min_child_weight=5 |
| LGBM_reg | num_leaves=15, min_child_samples=25, reg_lambda=2, lr=0.04 |
| Cat_reg | depth=4, iterations=350, lr=0.04, l2_leaf_reg=6 |
| SVM_reg | C=0.7, RBF, probability=True |

### 5.7 Hibrit algoritmalar

| Ad | Yapı |
|---|---|
| HybridVote_soft | Soft voting: RF + LR + XGB + LGBM |
| HybridVote_RFLRXGB | Soft voting: RF + LR + XGB |
| HybridStack_RFxgbLGB | Stacking: RF/XGB/LGBM → LR meta (cv=3) |
| HybridStack_RFxgbHGB | Stacking: RF/XGB/HistGBM → LR meta (cv=3) |

### 5.8 Report 2 — M1 en iyi 5

| Sıra | feature_set | model | n_feat | AUC |
|---:|---|---|---:|---:|
| 1 | TOP7_perm | HybridVote_RFLRXGB | 7 | **0.6855** |
| 2 | TOP7_perm | LR_L2_strong | 7 | 0.6839 |
| 3 | CONSENSUS7 | HybridVote_RFLRXGB | 7 | 0.6835 |
| 4 | CONSENSUS7 | LR_L2_strong | 7 | 0.6828 |
| 5 | FULL_13 | LR_L2_strong | 13 | 0.6815 |

**Δ vs Report1 M1 best (0.6807):** **+0.0049**

### 5.9 Report 2 — M2B en iyi 5

| Sıra | feature_set | model | n_feat | AUC |
|---:|---|---|---:|---:|
| 1 | FULL_13 | Cat_reg | 13 | **0.7578** |
| 2 | CONSENSUS7 | Cat_reg | 7 | 0.7561 |
| 3 | POS_only | Cat_reg | 13 | 0.7560 |
| 4 | TOP7_perm | Cat_reg | 7 | 0.7548 |
| 5 | CONSENSUS7 | HybridVote_RFLRXGB | 7 | 0.7514 |

**Δ vs Report1 M2B best (0.7497):** **+0.0081**

---

## 6. Ortam ve yazılım

| Bileşen | Sürüm / not |
|---|---|
| OS | Windows 10 |
| Python | 3.12.10 |
| Ana paketler | numpy, pandas, scikit-learn, xgboost, lightgbm, catboost, imbalanced-learn, scikit-survival, lifelines, shap, optuna (kurulu; Report1/2’de Optuna arama kullanılmadı), matplotlib, seaborn, plotly, scienceplots, python-docx, openpyxl, xlsxwriter, jupyter |
| requirements | `requirements.txt` |
| Seed | 42 |

---

## 7. Dosya envanteri (çıktılar)

| Dosya | İçerik |
|---|---|
| `database.xlsx` | Ana veri + sözlük |
| `BreastPredict_Rapor.html` | Report 1 |
| `report2.html` | Report 2 |
| `outputs/summary.json` | Report 1 özet |
| `outputs/summary_report2.json` | Report 2 özet |
| `outputs/tables/*.xlsx` | M1/M2A/M2B/M2S/M3 metrikleri |
| `outputs/tables_r2/*.xlsx` | Importance + tüm kombinasyonlar |
| `outputs/figures/` | Report 1 figürleri |
| `outputs/figures_r2/` | Report 2 figürleri |
| `run_full_modeling.py` | Report 1 pipeline |
| `run_report2.py` | Report 2 pipeline |
| `cohort_model_ozet.xlsx` | Kohort uygunluk özeti |

---

## 8. Bilinen sınırlılıklar (denetim için kritik)

1. **Sadece klinik + PR/HER2:** genomik yok → performans tavanı düşük.  
2. **Complete-case bias:** PR/HER2 eksik ~1260 hasta dışarıda.  
3. **Tedavi confounded:** KT/RT önem skoru yüksek olabilir; nedensel değil.  
4. **NO/NA belirsizliği:** “verilmedi” vs “bilinmiyor” ayrılmıyor.  
5. **M2A sınıf dengesizliği:** y1 çok az → Macro-F1 düşük (beklenen).  
6. **M3 örneklem küçük:** n≈416.  
7. **Eşik 0.5 sabit:** Youden / cost-sensitive eşik yok.  
8. **Nested CV / Bayesian HP search yok.**  
9. **External validation yok.**  
10. **AUC ≠ accuracy:** AUC sıralama kalitesi; Accuracy ayrı metrik.

---

## 9. Sonuçlar — tek bakışta

| Görev | Best (R1) | Metrik R1 | Best (R2) | Metrik R2 |
|---|---|---:|---|---:|
| M1 binary | RandomForest | AUC 0.681 | TOP7 + HybridVote_RFLRXGB | AUC **0.686** |
| M2B 5y | SVM_RBF | AUC 0.750 | FULL + Cat_reg | AUC **0.758** |
| M2A 6-class | RandomForest | MacroF1 0.227 | (R2’de yeniden optimize edilmedi) | — |
| M2S survival | RSF | C-index 0.686 | — | — |
| M3 site | RandomForest | MacroF1 0.430 | — | — |

**Genel hüküm:** Mevcut özellik setiyle modeller **orta düzey prognostik ayrım** sağlıyor. Report 2 sadeleştirme/hibrit ile **marjinal** iyileşme (+0.005 / +0.008) getirdi. Klinik olarak anlamlı sıçrama için moleküler özellik, impute ile n artışı, survival/time-dependent AUC ve external validation önerilir.

---

## 10. Başka LLM’e doğrulatma için kontrol listesi

Aşağıdakileri kaynak dosyalarla karşılaştırın:

- [ ] `database.xlsx` satır sayısı 3241 (1 header + 3240)?  
- [ ] Complete-case n ≈ 1838 mi?  
- [ ] M1 R1 best AUC ≈ 0.6807 RandomForest mı?  
- [ ] M2B R1 best AUC ≈ 0.7497 SVM mı?  
- [ ] M2S C-index ≈ 0.686 RSF mı?  
- [ ] R2 M1 best ≈ 0.6855 TOP7+HybridVote mı?  
- [ ] R2 M2B best ≈ 0.7578 Cat_reg mi?  
- [ ] External validation iddiası yok mu?  
- [ ] Seed 42, 5-fold stratified mı?  
- [ ] Tedavi değişkenlerinin confounded olabileceği belirtilmiş mi?

---

*Bu belge, proje klasöründeki `summary.json`, `summary_report2.json`, `outputs/tables/*` ve pipeline scriptlerinden üretilmiş denetim amaçlı nihai rapordur.*
