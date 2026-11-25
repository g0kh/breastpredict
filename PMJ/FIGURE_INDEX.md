# PMJ figure index

All figures are English-labeled, exported as 300 dpi PNG and vector PDF in `PMJ/figures/`.

| File | Type | Clinical question | Suggested use |
|---|---|---|---|
| `Fig01_alluvial_receptor_treatment_outcome` | Alluvial / Sankey | How do receptor status and systemic therapy flow into first distant-relapse patterns? | Main figure candidate for cohort phenotype → treatment → outcome narrative |
| `Fig02_upset_metastasis_sites` | UpSet plot | Which distant sites co-occur at relapse? | Supports competing-risk / multi-site biology discussion |
| `Fig03_raincloud_risk_by_event` | Raincloud | Do predicted risks separate patients who later experience DRFI? | External discrimination illustration for Results |
| `Fig04_ridgeline_risk_by_model` | Ridgeline (Joy) plot | How do risk distributions differ across models and time horizons? | Supplement or methods/results visual for model behavior |
| `Fig05_shap_beeswarm_mean` | SHAP beeswarm + mean impact hybrid | Which clinical features drive predicted DRFI hazard? | Key explainability figure for oncologist interpretation |
| `Fig06_calibration_hist_dca` | Calibration + marginal histogram + DCA | Are predicted probabilities reliable and clinically useful at decision thresholds? | Primary performance figure for external validation section |
| `Fig07_dumbbell_internal_external_auc` | Dumbbell plot | Does discrimination transport from METABRIC to the hospital cohort? | Headline external-validation comparison figure |
| `Fig08_swimmer_external` | Swimmer plot | What does individual follow-up and timing of first metastasis look like externally? | Patient-level narrative figure for external cohort |
| `Fig09_cif_competing_sites` | Competing-risk CIF | When do bone, visceral, and CNS first metastases accrue over time? | Core oncology time-to-site figure |
| `Fig10_forest_cohort_loo` | Forest / Cleveland-style LOO | Is performance stable across METABRIC source cohorts? | Internal-external robustness figure |
| `Fig11_tdauc_trajectories` | Time-dependent AUC trajectories | How does discrimination change from early to late follow-up? | External validation time-dynamics figure |
| `Fig12_heatmap_receptor_site` | Heatmap | Do receptor phenotypes prefer different first metastatic niches? | Biology/clinical pattern figure |
| `Fig13_km_by_risk_tertile` | Kaplan–Meier by risk tertile | Do model-defined risk groups separate event-free survival externally? | Clinical risk-stratification figure |
| `Fig14_cohort_flow` | Cohort flow | How was the analytic cohort derived from METABRIC? | Methods / Figure 1 candidate |
| `Fig15_features_by_site` | Violin/raincloud grid by site | Do baseline clinicopathologic features differ by first metastatic niche? | Exploratory phenotype–site association panel |
| `Fig16_lollipop_metrics` | Cleveland lollipop | Which model balances discrimination and probability accuracy internally? | Model-selection summary figure |
| `Fig17_treatment_patterns` | Grouped treatment pattern bars | How were chemotherapy and endocrine therapy used across ER groups? | Clinical context / confounding-by-indication panel |
| `Fig18_features_training_vs_center` | Training vs center feature comparison | How do demographic and pathologic features differ between METABRIC and the local center? | Methods/Results cohort comparability figure for oncologists |

## Notes for selection
- Prefer **Fig14** as methods flow, **Fig18** for cohort comparability, **Fig07/Fig06** for external validation, **Fig05** for explainability, **Fig09** for site biology.
- External n=80 / 23 events: interpret Fig03/08/13 as supportive, not definitive.
- Style: Nature/Lancet/Okabe/Teal–Coral palettes; charcoal axes; no top/right spines.
- Titles use separated title + subtitle blocks (no overlapping text).