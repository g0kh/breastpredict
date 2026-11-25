# Manuscript figure guide (`PMJ/manu/figures`)

Oncologist-facing reading order for the DRFI risk manuscript pack.  
All figure text is English. Each figure is exported as **PNG (300 dpi)** and **PDF**.

**Endpoint language:** Distant recurrence or breast-cancer death (**DRFI**).  
**Best model (by 5y tdAUC):** Random survival forest (RSF).

---

## Submission figures (journal set)

Location: [`PMJ/manu/figures/journal/`](figures/journal/).  
Generator: `python PMJ/manu/make_journal_figures.py`  
No large title banners; panels marked **A / B / C** only.

| Figure | File | Panels | Source / merge |
|---|---|---|---|
| Fig 1 | `Fig1_cohort_case_mix` | A densities + B categorical bars | M01 / Fig18 (cleaned) |
| Fig 2 | `Fig2_SHAP_explainability` | A beeswarm + B mean \|SHAP\| | M03 / Fig05 (300 dpi) |
| Fig 3 | `Fig3_scenario_A_discrimination` | A dumbbell + B 5y risk strip | M04 + M09 left |
| Fig 4 | `Fig4_calibration_DCA_KM` | A calibration + B DCA + C KM | M05 (no hist) + M06 |
| Fig 5 | `Fig5_scenario_B_timing` | A tdAUC + B Brier | M10 (RSF labels only) |
| Fig 6 | `Fig6_scenario_C_first_site` | A heatmap + B internal AUC + C external counts | M07 + M11 |
| Supp Fig 1 | `SuppFig1_model_card` | — | M02 (optional supplement) |

**Retired from the article set (kept on disk under `manu/figures/`):**  
M02 (→ Supp only), M08 overview, M05 middle histogram, M09 right histogram.

**Locked colors:** NPI-Cox `#00468B`, Cox `#ED0000`, Cox ER-stratified `#42B540`, RSF `#0099B4`; Bone `#2A9D8F`, Visceral `#E64B35`, CNS `#F39B7F`, Other `#6C757D`.

---

## Legacy M-pack reading order

| Order | File | What an oncologist should see |
|---|---|---|
| 1 | `M01_cohort_features_training_vs_center` | Who was studied: training (METABRIC) vs your hospital — age, size, nodes, receptors, treatments side by side. |
| 2 | `M02_model_card` | What the score uses and predicts: STEEP DRFI endpoint, clinical inputs in plain English, exclusions (stage 4/DCIS), models compared. |
| 3 | `M03_model_explainability_SHAP` | Which factors push risk up or down (age, nodes, size, grade, receptors, therapy). |
| 4 | `M04_internal_vs_external_performance` | Does the score travel to our hospital? Internal CV vs external 5y tdAUC (+ bootstrap CI); RSF highlighted. |
| 5 | `M05_external_calibration_DCA` | Are probabilities reliable, and is there net benefit vs treat-all / treat-none at decision thresholds? |
| 6 | `M06_KM_by_risk_tertile` | Risk groups separate survival: low / mid / high tertiles of predicted risk on Kaplan–Meier. |
| 7 | `M08_three_scenarios_AI_performance` | Overview strip: (A) will DRFI occur? (B) timing by year? (C) where first (bone/visceral/CNS)? |
| 8 | `M09_scenario_A_risk_separation` | Scenario A detail: patients with DRFI by 5y have higher predicted risk than those without (external). |
| 9 | `M10_scenario_B_timing` | Scenario B detail: does discrimination change at 2 / 5 / 10 years? (external tdAUC + Brier). |
| 10 | `M11_scenario_C_first_site` | Scenario C detail: bone / visceral / CNS — internal cause-specific AUC + external site counts. |
| — | `M07_receptor_by_first_site` | Biology context: receptor status × first distant site heatmap (supporting, not a performance claim). |

Suggested narrative path: **M01 → M02 → M03 → M04 → M05 → M06**, then **M08 → M09 → M10 → M11** for the three-scenario story; **M07** as site-biology supplement.

---

## Per-figure notes

### M01 — Cohort features (training vs center)
- Source: Fig18.
- Compare distributions/means so readers see whether the external hospital looks like the training population.

### M02 — Model card
- Infographic: endpoint, inputs (full English names), exclusions, models (NPI-Cox, Cox, Cox ER-stratified, RSF).
- Explicitly avoids invalid binary M1/M2A labels as main claims.

### M03 — Explainability (SHAP)
- Source: Fig05.
- Feature names spelled out (not abbreviations alone).

### M04 — Internal vs external
- Dumbbell + numeric callouts (5y tdAUC, Uno C).
- External bootstrap CI shown; interpret point estimates with wide CI in mind (n=80).

### M05 — Calibration + DCA
- Source: fixed Fig06 (DCA legend outside plot, upper-right).
- Calibration, risk histogram, decision curve on one row.

### M06 — KM by risk tertile
- Source: Fig13.
- Visual answer: “Do high-score patients fail earlier?”

### M07 — Receptor × first site
- Source: Fig12.
- Descriptive biology; not used as primary performance evidence.

### M08 — Three scenarios (A / B / C)
- **(A) Risk:** internal + external 5y tdAUC (and Uno C note).
- **(B) Timing:** external tdAUC at 2 / 5 / 10 years.
- **(C) Site:** internal cause-specific AUC; external site counts only (small-n caveat — no external site AUC claim).

### M09 — Scenario A risk separation
- External raincloud + histogram of RSF 5y risk: no DRFI by 5y vs DRFI by 5y (Teal–Coral).
- Histogram legend locked to panel upper-right (not in the title band).

### M10 — Scenario B timing
- External tdAUC trajectory at 2 / 5 / 10 years (all models).
- Companion Brier panel (lower = better); RSF callout strip.

### M11 — Scenario C first site
- Left: internal cause-specific 5y AUC for bone / visceral / CNS.
- Right: external first-site counts only (descriptive; no external site AUC claim).

---

## Regenerate

```bash
# Submission figures 1–6
python PMJ/manu/make_journal_figures.py

# Legacy M01–M11 pack
python PMJ/manu/make_manu_figures.py
```

Journal set writes to `PMJ/manu/figures/journal/`. Legacy pack needs existing `PMJ/figures/` sources for M01/M03/M06/M07 and `sonuclar/` tables.

---

## File checklist

| Stem | PNG | PDF |
|---|---|---|
| M01_cohort_features_training_vs_center | ✓ | ✓ |
| M02_model_card | ✓ | ✓ |
| M03_model_explainability_SHAP | ✓ | ✓ |
| M04_internal_vs_external_performance | ✓ | ✓ |
| M05_external_calibration_DCA | ✓ | ✓ |
| M06_KM_by_risk_tertile | ✓ | ✓ |
| M07_receptor_by_first_site | ✓ | ✓ |
| M08_three_scenarios_AI_performance | ✓ | ✓ |
| M09_scenario_A_risk_separation | ✓ | ✓ |
| M10_scenario_B_timing | ✓ | ✓ |
| M11_scenario_C_first_site | ✓ | ✓ |
