"""Draft manuscript (author-editable). Run: python PMJ/manu/build_manuscript_docx.py"""
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

OUT = Path(__file__).resolve().parent / "Manuscript_DRFI_three_scenarios_DRAFT.docx"


def set_run_font(run, name="Times New Roman", size=12, bold=False, italic=False, color=None):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    if color:
        run.font.color.rgb = RGBColor(*color)


def add_p(doc, text, *, size=12, bold=False, italic=False, center=False, space_after=8, space_before=0, first_line=True):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_after = Pt(space_after)
    pf.space_before = Pt(space_before)
    pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    if center:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif first_line:
        pf.first_line_indent = Cm(1.25)
    r = p.add_run(text)
    set_run_font(r, size=size, bold=bold, italic=italic)
    return p


def add_h(doc, text, level=1):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(16 if level == 1 else 12)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    r = p.add_run(text)
    set_run_font(r, size=14 if level == 1 else 12, bold=True)
    return p


def add_body(doc, text):
    return add_p(doc, text, first_line=True)


def add_nofirst(doc, text, **kw):
    kw["first_line"] = False
    return add_p(doc, text, **kw)


def shade_header(cell):
    tc = cell._tePr if hasattr(cell, "_tePr") else cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = tcPr.makeelement(
        "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}shd",
        {
            "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val": "clear",
            "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}color": "auto",
            "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}fill": "1D3557",
        },
    )
    tcPr.append(shd)
    for p in cell.paragraphs:
        for r in p.runs:
            r.font.color.rgb = RGBColor(255, 255, 255)
            r.bold = True
            r.font.name = "Times New Roman"
            r.font.size = Pt(10)


def add_table(doc, headers, rows, caption):
    cap = doc.add_paragraph()
    cap.paragraph_format.space_before = Pt(10)
    cap.paragraph_format.space_after = Pt(4)
    cap.paragraph_format.first_line_indent = Cm(0)
    r = cap.add_run(caption)
    set_run_font(r, size=11, bold=True)
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    for j, h in enumerate(headers):
        cell = table.rows[0].cells[j]
        cell.text = ""
        p = cell.paragraphs[0]
        run = p.add_run(h)
        set_run_font(run, size=10, bold=True, color=(255, 255, 255))
        shade_header(cell)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = table.rows[i + 1].cells[j]
            cell.text = ""
            p = cell.paragraphs[0]
            run = p.add_run(str(val))
            set_run_font(run, size=10)
    note = doc.add_paragraph()
    note.paragraph_format.space_after = Pt(8)
    return table


def build():
    doc = Document()
    sec = doc.sections[0]
    sec.page_width = Cm(21.0)
    sec.page_height = Cm(29.7)
    sec.left_margin = Cm(2.5)
    sec.right_margin = Cm(2.5)
    sec.top_margin = Cm(2.5)
    sec.bottom_margin = Cm(2.5)
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(12)
    style._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")

    add_nofirst(
        doc,
        "DRAFT FOR AUTHOR EDITING — not for submission. Replace bracketed placeholders (authors, institution, ethics, funding). Numbers below are taken from the analysis outputs in this project (sonuclar/) and the manuscript figures M01–M11.",
        size=11,
        italic=True,
        space_after=12,
    )

    add_p(
        doc,
        "Predicting whether, when, and where distant recurrence occurs after surgery for breast cancer: development of a clinical survival model in METABRIC and external validation in a single-center cohort",
        size=16,
        bold=True,
        center=True,
        first_line=False,
        space_after=10,
    )
    add_p(
        doc,
        "[Author 1], [Author 2], [Author 3]",
        size=12,
        center=True,
        first_line=False,
        space_after=2,
    )
    add_p(
        doc,
        "[Department of Medical Oncology], [Institution], [City], [Country]",
        size=11,
        italic=True,
        center=True,
        first_line=False,
        space_after=2,
    )
    add_p(
        doc,
        "Corresponding author: [Name, email]",
        size=11,
        center=True,
        first_line=False,
        space_after=12,
    )

    add_h(doc, "Abstract")
    add_body(
        doc,
        "Background. After surgery for early breast cancer, distant recurrence can still be approached with curative intent when it is detected early and is limited in extent. Many tools have been developed to estimate that risk. Multigene assays such as Oncotype DX, MammaPrint, and EndoPredict inform chemotherapy decisions in selected patients, but reimbursement rules still restrict access in many settings. Even when such tests are available, breast cancer remains a systemic disease and distant relapse is not abolished. We therefore asked whether routinely recorded clinical variables, learned from a large open cohort, could support three practical questions: will distant recurrence or breast-cancer death occur, when is that risk expressed, and which organ is affected first.",
    )
    add_body(
        doc,
        "Methods. The development cohort was assembled from the public METABRIC clinical tables (Pereira et al., supplementary clinical data linked to the METABRIC resource). After exclusion of stage 4 disease, ductal carcinoma in situ, benign lesions, and phyllodes tumors, 3,208 patients remained, of whom 1,251 had a distant recurrence-free interval (DRFI) event as defined in the STEEP system (distant relapse or death from breast cancer). Predictors were age, tumor size, number of positive lymph nodes, histologic grade, estrogen receptor (ER), progesterone receptor (PR), HER2, menopausal status, lobular histology, and receipt of chemotherapy or endocrine therapy, with missingness indicators for PR and HER2. Four survival models were compared under five-fold cross-validation: a Cox model using the Nottingham Prognostic Index (NPI-Cox), a multivariable Cox model, an ER-stratified Cox model, and a random survival forest. The prespecified primary metric was the 5-year time-dependent area under the curve (AUC). Models were then locked and applied, without refitting, to a retrospective single-center cohort of 80 patients with operable breast cancer (23 DRFI events). Scenario A addressed occurrence of DRFI, scenario B the change in discrimination from 2 to 10 years, and scenario C the first distant site (bone, visceral, or central nervous system). Site discrimination was estimated internally; external site results are reported only as counts.",
    )
    add_body(
        doc,
        "Results. In internal cross-validation the random survival forest had the highest 5-year time-dependent AUC (0.732; fold-based interval 0.717–0.747), close to the ER-stratified Cox model (0.729) and above the clinical Cox model (0.714) and NPI-Cox (0.690). In the external cohort the same ranking was preserved, with a 5-year AUC of 0.834 for the random survival forest (bootstrap 95% confidence interval 0.741–0.909). Discrimination was strongest at 2 years (AUC 0.914) and weaker at 10 years (AUC 0.739). Patients who had a DRFI event by 5 years had a higher predicted risk than those who did not (mean 0.44 versus 0.24). High, intermediate, and low risk thirds separated on Kaplan–Meier curves. Absolute probabilities were not exchangeable with observed rates (calibration slope about 1.65 before recalibration). Cause-specific 5-year AUCs for first site were approximately 0.70 for bone, 0.75 for visceral disease, and 0.80 for central nervous system disease, the last with a wide standard deviation. Among 23 external events the first recorded sites were bone (n=11), visceral (n=7), central nervous system (n=4), and other (n=1); an external site AUC is not claimed.",
    )
    add_body(
        doc,
        "Conclusions. A clinical survival model trained on an open breast-cancer cohort can rank the risk of distant recurrence or breast-cancer death in a contemporary single-center series, and it retains useful short- and medium-term discrimination. It does not replace multigene assays, it is not calibrated for immediate threshold decisions without local adjustment, and it cannot yet assign the first metastatic site in an external hospital. Used as a research score alongside routine clinicopathologic data, it offers a transparent starting point for the three questions clinicians actually ask after surgery.",
    )
    add_nofirst(
        doc,
        "Keywords: breast cancer; distant recurrence-free interval; survival model; random survival forest; external validation; metastatic site",
        italic=True,
        first_line=False,
        space_before=4,
    )

    add_h(doc, "Introduction")
    add_body(
        doc,
        "Surgery remains the cornerstone of curative treatment for early breast cancer, yet a substantial minority of patients later develop distant disease. When that relapse is found while it is still limited—particularly an isolated or oligometastatic deposit—local ablative treatment combined with systemic therapy can still be delivered with curative intent in selected patients. That possibility depends on recognizing who is at risk, and on recognizing it early enough that surveillance and subsequent treatment are timed to the biology of the relapse rather than to a late, widespread presentation. For this reason, estimation of recurrence risk has been a central theme of adjuvant breast oncology for more than two decades.",
    )
    add_body(
        doc,
        "The tools that entered practice first were clinicopathologic. Tumor size, nodal burden, histologic grade, and hormone-receptor status were summarized in indices such as the Nottingham Prognostic Index and, later, in online calculators. These scores are available to every clinic. They are also coarse. Two patients with the same index can follow very different courses, and the index itself says little about when a relapse is most likely or which organ will declare it.",
    )
    add_body(
        doc,
        "A second generation of tools used tumor gene expression to refine chemotherapy decisions. Oncotype DX reports a recurrence score that, in hormone-receptor-positive, HER2-negative disease, identifies patients who can safely omit adjuvant chemotherapy. MammaPrint, evaluated in MINDACT, asks a similar question with a 70-gene signature. EndoPredict combines a molecular score with tumor size and nodal status to estimate late distant recurrence and to guide extended endocrine therapy as well as chemotherapy. These assays have changed conversations in the clinic. They have not become universal. In many health systems, including settings where reimbursement is incomplete or restricted to narrow indications, patients and physicians still make adjuvant decisions without a multigene result. Access is a practical constraint, not a theoretical one.",
    )
    add_body(
        doc,
        "Even when a genomic panel is obtained and chemotherapy or endocrine therapy is given accordingly, breast cancer is a systemic disease. Micrometastases may already be present at diagnosis. Systemic treatment reduces but does not remove the risk of later distant relapse, and that residual risk extends well beyond the first five years, especially in hormone-receptor-positive disease. A negative genomic assay is not a guarantee. Clinicians therefore still need a way to discuss residual risk with the information they already hold in the record: age, stage, receptors, and the treatment that was actually delivered.",
    )
    add_body(
        doc,
        "Most published models answer only the first of three questions a patient asks after surgery. The first is whether distant recurrence, or death from breast cancer, is likely at all. The second is when that risk is concentrated—within two years, by five years, or as a late event approaching ten years—because the answer changes both surveillance and the value of extended endocrine therapy. The third is where the first distant site is likely to be. Bone, visceral organs, and the central nervous system do not carry the same symptoms, the same imaging yield, or the same chance of a still-curable presentation. A model that is silent on timing and site leaves two of the three clinical questions unanswered.",
    )
    add_body(
        doc,
        "Large open clinical datasets now make it possible to train such models without proprietary genomic inputs. The METABRIC resource, and the curated clinical tables released with subsequent genomic characterizations of that series, records long-term relapse and death alongside ordinary clinicopathologic variables. The present study used that resource to build a distant recurrence-free interval (DRFI) model and then tested it, without refitting, in a real consecutive cohort from the authors’ hospital. The work was organized explicitly around three scenarios: (A) will a DRFI event occur; (B) if the risk is time-dependent, how does discrimination change from 2 to 10 years; and (C) if distant disease occurs, is the first site bone, visceral, or central nervous system. The aim was not to replace Oncotype DX, MammaPrint, or EndoPredict, and not to produce a deployable decision threshold. The aim was to test whether an open-data clinical survival model can travel to a hospital population and remain informative for those three questions.",
    )

    add_h(doc, "Methods")
    add_h(doc, "Study design and reporting", level=2)
    add_body(
        doc,
        "This was a model-development and external-validation study. Development used a public cohort; validation used a retrospective single-center cohort of patients treated with surgery for invasive breast cancer. The report is written so that a clinical reader can follow the cohort, the endpoint, the inputs, and the performance claims, and so that a methods reader can reconstruct the modeling choices. Reporting was aligned with the spirit of TRIPOD+AI (transparent reporting of a multivariable prediction model for individual prognosis, extended to regression and machine-learning approaches), without claiming that every checklist item is complete in this draft. No genomic assay result was used as an input or as an outcome.",
    )

    add_h(doc, "Development data", level=2)
    add_body(
        doc,
        "The development data were the clinical annotations of the Molecular Taxonomy of Breast Cancer International Consortium (METABRIC), as released with the somatic-mutation and transcriptomic characterization of that series (supplementary clinical tables accompanying Pereira et al., building on the original METABRIC cohort of Curtis et al.). The starting clinical table contained 3,240 rows. Patients were excluded in sequence if follow-up time was missing or non-positive (2 patients), if vital status was missing (2 patients), if stage at diagnosis was 4 (11 patients), or if the recorded histology was benign, phyllodes, or ductal carcinoma in situ (17 patients). The analysis cohort therefore comprised 3,208 patients. A prespecified sensitivity cohort further required a complete recurrence history and contained 2,639 patients. Radiotherapy and type of surgery were recorded in the source tables but were not used as model inputs, because their coding was incomplete and because they are downstream of, rather than intrinsic to, baseline risk. Binary labels that simply flag “any metastasis” without time or site were not used as outcomes.",
    )
    add_body(
        doc,
        "The primary endpoint followed the STEEP definition of distant recurrence-free interval. An event was distant relapse or death attributed to breast cancer, whichever was recorded. Time was measured from diagnosis to distant relapse when relapse occurred, and otherwise to last follow-up or death. Deaths that were not attributed to breast cancer were treated as censored for DRFI. In the development cohort there were 1,251 DRFI events: 1,072 with distant relapse and 179 breast-cancer deaths without a recorded distant relapse. This distinction matters for scenario C, which concerns the first distant site and therefore applies to patients with a documented distant relapse rather than to every DRFI death.",
    )

    add_h(doc, "Predictors", level=2)
    add_body(
        doc,
        "Only variables that a medical oncologist would expect to find in a standard chart, and that could be reconstructed in both cohorts, were retained. Continuous inputs were age at diagnosis (years), natural log of tumor size in millimeters (values below 1 mm were floored at 1 mm before the log), and the natural log of one plus the number of positive lymph nodes (node counts above 20 were capped). Grade was entered as recorded (1, 2, or 3). Binary inputs were ER status, PR status, HER2 status, postmenopausal status, lobular histology (invasive lobular carcinoma versus other invasive histologies), receipt of chemotherapy, and receipt of endocrine therapy. PR and HER2 were frequently unrecorded in METABRIC (1,255 patients each). Rather than delete those patients, we kept a missingness indicator for each receptor so that the model could use the fact of missingness without inventing a positive or negative result. Chemotherapy and endocrine therapy were coded from the source fields as received versus not received or not recorded. The Nottingham Prognostic Index, when used, was the baseline comparator only; it was not one of the multivariable inputs.",
    )
    add_body(
        doc,
        "Ki-67 was not available in a form that would support the Clinical Treatment Score post-5 years (CTS5), so CTS5 was not calculated. No gene-expression subtype, no multigene recurrence score, and no circulating-tumor markers were used.",
    )

    add_h(doc, "External cohort", level=2)
    add_body(
        doc,
        "External validation used a retrospective cohort from the authors’ institution [ethics approval number to be inserted]. Consecutive patients who had undergone surgery for invasive breast cancer, who had the same clinical fields available, and who had follow-up for distant relapse and vital status were included. Eighty patients met these criteria. Twenty-three had a DRFI event during follow-up (event proportion 0.288). Median follow-up was 7.3 years. The same feature definitions and the same DRFI coding were applied. The development models were not refitted on these patients. First distant site was abstracted from the clinical record into bone, visceral, central nervous system, or other. Mixed bone-and-visceral presentations were grouped with bone for this descriptive count, as prespecified in the extraction map. Because only 23 events were available, no external discrimination statistic was calculated for site.",
    )

    add_h(doc, "Models", level=2)
    add_body(
        doc,
        "Four survival models were specified before external scoring. The first, NPI-Cox, was a penalized Cox proportional-hazards model whose only covariate was the Nottingham Prognostic Index, standardized within the training fold. It represents the clinicopathologic baseline a reader already knows. The second was a multivariable Cox model on the full, standardized feature set, with a ridge penalty (alpha = 1.0 in the scikit-survival implementation) to limit overfitting from correlated receptors and treatment variables. The third fit separate Cox models within ER-positive and ER-negative strata, with a pooled Cox model as a fallback if a stratum was too small; this acknowledges that the weight of size, nodes, and endocrine therapy is not the same on both sides of the ER divide. The fourth was a random survival forest: 300 trees, minimum 15 patients in a terminal node, and a random subset of features equal to the square root of the number of inputs at each split. The forest does not assume proportional hazards and can represent interactions (for example, nodal burden with receptor status) that a main-effects Cox model averages away.",
    )
    add_body(
        doc,
        "Within each training fold, missing values were imputed with iterative imputation (a Bayesian-ridge style multivariate imputer, 10 iterations) fitted on the training fold only and then applied to the held-out fold. Features were standardized with the training-fold mean and standard deviation. The same imputer and scaler, fitted on the full development cohort, were applied once to the external cohort at the time of locked validation. Predicted risk at a horizon was one minus the model’s survival function at that time.",
    )

    add_h(doc, "Internal validation, metrics, and the three scenarios", level=2)
    add_body(
        doc,
        "Internal performance used five-fold cross-validation with shuffling and a fixed seed (random state 42). Each patient received an out-of-fold prediction from a model that had not seen that patient. The primary discrimination metric was the 5-year time-dependent AUC (cumulative/dynamic), which asks whether a patient who has the event by 5 years is assigned a higher risk than a patient still event-free at 5 years, with appropriate handling of censoring. Overall ranking across follow-up was summarized by Uno’s concordance index at 10 years, which is a censoring-robust C statistic. Accuracy of the probability itself, as opposed to its rank, was summarized by the Brier score at 5 and 10 years (lower is better). Fold-level means were reported with the range implied by the fold estimates. A sensitivity analysis repeated the same cross-validation in the 2,639 patients with a complete recurrence history.",
    )
    add_body(
        doc,
        "Scenario A is the primary DRFI question: discrimination, calibration, decision-curve net benefit, separation of risk groups, and the simple contrast of predicted 5-year risk in patients who did or did not have an event by 5 years. Scenario B re-estimates time-dependent AUC and the Brier score at 2, 5, and 10 years on the external cohort, to show whether the score is an early-relapse tool, a late-relapse tool, or both. Scenario C fits cause-specific survival models for bone-first, visceral-first, and central-nervous-system-first events in the development data and reports the cross-validated 5-year AUC with its standard deviation across folds. External scenario C is descriptive only.",
    )
    add_body(
        doc,
        "External uncertainty for the 5-year AUC and Brier score used 400 patient-level bootstrap resamples. Calibration was inspected by grouping predicted 5-year risk and plotting mean predicted risk against the observed event rate, and by a logistic calibration slope on the logit of predicted risk. A slope far from 1 means the numbers should not be read as personal percentages until they are recalibrated. Decision-curve analysis compared the models with “treat all” and “treat none” across threshold probabilities from 0.05 to 0.55. This curve is a statement about net benefit in this dataset, not a recommendation to give or withhold a named drug. Kaplan–Meier curves by tertile of predicted risk were drawn to show whether the score separates observed DRFI in a way a clinician can see without an AUC. Explainability used a SHAP-style attribution from the standardized Cox model (each patient’s standardized feature value multiplied by the corresponding coefficient), displayed as a beeswarm and as mean absolute impact. This attributes the linear clinical model; it is not a tree-specific explanation of the forest.",
    )

    add_h(doc, "Software", level=2)
    add_body(
        doc,
        "Analyses were run in Python 3.10.11 (64-bit) on Windows. Survival models and time-dependent metrics used scikit-survival 0.25.0 (CoxPHSurvivalAnalysis, RandomSurvivalForest, cumulative dynamic AUC, inverse-probability-of-censoring Brier score, and Uno concordance). Imputation and scaling used scikit-learn 1.7.1. Kaplan–Meier curves and the Aalen–Johansen estimator for cumulative incidence used lifelines 0.30.0. The SHAP-style display used the shap package 0.49.1 together with the Cox coefficients. Data handling used pandas 2.3.1 and NumPy 1.26.4. Figures were drawn with Matplotlib 3.10.3. The random seed was 42 throughout.",
    )

    add_h(doc, "Ethics", level=2)
    add_body(
        doc,
        "METABRIC clinical tables are publicly released research data. The external cohort is a retrospective chart review; [insert local ethics committee name and approval number, and state whether consent was waived]. This draft does not report identifiable patient data.",
    )

    add_h(doc, "Results")
    add_h(doc, "Who was studied", level=2)
    add_body(
        doc,
        "The development cohort and the hospital cohort are summarized in Table 1 and compared visually in Figure M01. The METABRIC analysis set was an older, largely postmenopausal population (median age 61.9 years; 78.1% postmenopausal) with a median tumor size of about 25 mm and a median of zero positive nodes. ER was positive in 71.5%. Among patients with a recorded result, PR was positive in 52.8% and HER2 in 12.3%; both receptors were missing in 1,255 patients (39.1%). Grade 3 disease was recorded in 1,548 patients (48.3%; grade missing in 162). Lobular histology accounted for 12.0%. Chemotherapy had been given to 19.6% and endocrine therapy to 55.5%, figures that reflect both the era of the cohort and the coding of treatment (including “not recorded” with “not given”). The source series largely predates routine adjuvant trastuzumab, so HER2-positive disease in the training data was not managed as it is managed now.",
    )
    add_body(
        doc,
        "The external cohort was smaller, younger (median age 54.9 years), and more often treated with chemotherapy (66.3%) and endocrine therapy (73.8%). Median tumor size was 28 mm. ER positivity was similar (73.8%), PR positivity was 55.7%, and HER2 positivity was higher (20.3%). Two-thirds were postmenopausal. Median follow-up was 7.3 years, and 23 of 80 patients (28.8%) had a DRFI event, compared with 1,251 of 3,208 (39.0%) in development, where follow-up was longer and the event definition accumulated late breast-cancer deaths. Figure M01 is the right place to see these case-mix differences before any performance number is interpreted: the hospital series is not a random slice of METABRIC. It is a younger, more heavily treated group with more HER2-positive disease.",
    )
    add_body(
        doc,
        "Figure M02 states, on a single card, what the score is allowed to use and what it is not. The endpoint is DRFI, not a binary flag for “metastasis yes/no.” Stage 4 disease, in situ disease, and non-epithelial tumors were excluded. The four models sit on that card so that later performance figures can be read against a named comparator rather than against an unspecified algorithm.",
    )

    add_h(doc, "What drives the score", level=2)
    add_body(
        doc,
        "Figure M03 shows the direction and the relative magnitude of the clinical drivers in the Cox model. As expected for distant recurrence, higher grade, larger tumor size, and a greater nodal burden pushed risk upward, and ER positivity and endocrine therapy pushed it downward. Age, menopausal status, HER2, PR, lobular histology, and chemotherapy contributed with smaller average effects, and the missingness indicators for PR and HER2 were visible, which is a reminder that “unknown receptor” was itself informative in the historical data rather than a nuisance to be dropped. These attributions explain the linear model. The random survival forest, which performed slightly better, can use the same variables in combination; the beeswarm should be read as a clinical map of the inputs, not as a claim that each bar is an independent causal effect of giving or withholding a drug. Chemotherapy, in particular, is given to higher-risk patients, so its coefficient mixes treatment effect with indication.",
    )

    add_h(doc, "Does the score travel to the hospital cohort?", level=2)
    add_body(
        doc,
        "Internal and external 5-year discrimination are shown side by side in Figure M04 and in Table 2. Inside METABRIC, the random survival forest was the strongest of the four models (5-year time-dependent AUC 0.732, interval across the cross-validation estimate 0.717–0.747). The ER-stratified Cox model was close behind (0.729), the single Cox model was intermediate (0.714), and NPI-Cox was the weakest (0.690). Uno concordance at 10 years was 0.668 for the forest, 0.668 for the ER-stratified Cox model, 0.668 for the Cox model, and 0.647 for NPI-Cox: the models separate risk, but the separation is moderate, which is what clinicians should expect from clinicopathologic variables alone. Five-year Brier scores were 0.163 for the forest and the ER-stratified Cox model, 0.165 for the Cox model, and 0.174 for NPI-Cox. Restricting development to patients with a complete recurrence history (n=2,639) did not weaken the result; the forest’s 5-year AUC in that sensitivity analysis was 0.753.",
    )
    add_body(
        doc,
        "Applied unchanged to the 80 hospital patients, every model discriminated better on the point estimate than it had internally, and the same model remained best. The random survival forest reached a 5-year time-dependent AUC of 0.834. The bootstrap 95% confidence interval was 0.741 to 0.909, so the point estimate should not be quoted as if it were known to two decimal places. The ER-stratified Cox model (0.816; 0.711–0.907), the Cox model (0.811; 0.690–0.915), and NPI-Cox (0.811; 0.701–0.912) all cleared the internal NPI benchmark, and their intervals overlap. The clinically fair summary is that the score did travel: patients in this hospital who were destined to have a DRFI event were, on average, ranked above those who were not, and a forest added a modest increment over a Cox model built from the same chart. The increment is not large enough, and the external sample is not large enough, to declare the forest uniquely required for practice.",
    )

    add_h(doc, "Are the percentages trustworthy, and do risk groups separate?", level=2)
    add_body(
        doc,
        "Discrimination and calibration are different questions. Figure M05 places them on one row for the external cohort, together with a decision curve. The calibration plot does not lie on the identity line. The logistic calibration slope for the preferred model was approximately 1.65 before any correction and moved to approximately 1.00 after a simple recalibration. In plain language, the rank order was more trustworthy than the raw percentage. A predicted “30%” in this export of the model should not be counselled as a personal 30% until the score has been recalibrated on local outcomes. After that recalibration step, the slope was compatible with a usable probability, but the external sample is too small for that recalibration to be considered definitive.",
    )
    add_body(
        doc,
        "The same figure shows that, across a range of threshold probabilities, the models sat above the strategies of treating everyone alike or treating no one. That is evidence of potential decision value, not evidence for a particular drug. Net benefit in 80 patients is a hypothesis-generating result. Figure M06 shows the complementary clinical picture: when external patients were split into thirds of predicted risk, the Kaplan–Meier curves separated, with the high-risk third carrying more DRFI events earlier. Tertiles are a display, not a validated three-group staging system, and they should not be turned into surveillance schedules on this sample alone.",
    )

    add_h(doc, "Scenario A — will distant recurrence or breast-cancer death occur?", level=2)
    add_body(
        doc,
        "Scenario A is the occurrence question and is the analysis the primary endpoint supports. Figure M08 (panel A) repeats the internal-versus-external 5-year AUC comparison in a form meant for a clinical slide. Figure M09 shows the same fact without a receiver-operating curve. In the external cohort, 14 patients had a DRFI event by 5 years and 66 did not. Mean random-forest 5-year risk was 0.44 in the first group and 0.24 in the second (difference +0.20). The distributions overlap—some patients with an event had intermediate scores, and some patients without an event had scores in the same range—so the figure supports risk stratification rather than a rule that a single cut-off identifies who will relapse. Taken with the Kaplan–Meier tertiles and the external AUC, scenario A is the part of the study that is ready to be discussed as a positive validation: the score marks, in a new hospital, patients who are more likely to experience distant recurrence or breast-cancer death.",
    )

    add_h(doc, "Scenario B — if it occurs, when is the risk expressed?", level=2)
    add_body(
        doc,
        "Scenario B asks whether that ranking is stable over time (Figure M08, panel B, and Figure M10). On the external cohort the random survival forest was strongest at the early horizon and less strong later: time-dependent AUC 0.914 at 2 years, 0.834 at 5 years, and 0.739 at 10 years. The ER-stratified Cox model followed the same downward slope (0.875, 0.816, and 0.719). NPI-Cox was flatter between 2 and 5 years (0.817 and 0.811) and then fell to 0.712 at 10 years. Brier scores rose for every model as the horizon lengthened, which is expected as more events and more competing clinical histories accumulate; at 5 years the forest Brier score was 0.131, similar to the Cox models and lower than NPI-Cox (0.147). Uno concordance at 10 years on the external cohort was 0.750 for the forest.",
    )
    add_body(
        doc,
        "The clinical reading is that the score is most informative for early distant recurrence and remains informative, with attenuated discrimination, out to 10 years. It should not be described as a specific late-relapse assay. Extended endocrine therapy decisions, which turn on residual risk after year 5 in hormone-receptor-positive disease, are not directly answered here, because the model does not include Ki-67 or a genomic residual-risk score and because the 10-year external AUC, while above 0.7, sits well below the 2-year figure. Scenario B supports a conversation about timing. It does not by itself set the duration of endocrine therapy.",
    )

    add_h(doc, "Scenario C — where does distant disease appear first?", level=2)
    add_body(
        doc,
        "Scenario C is the weakest of the three claims and is presented as such (Figure M08, panel C, and Figure M11). In the development cohort, first distant site among patients with a documented distant relapse was other or unspecified in 537, visceral in 258, bone in 246, and central nervous system in 31. The large “other” category limits how finely a site model can be trained. Cross-validated cause-specific 5-year AUCs were 0.70 (standard deviation 0.10) for bone-first, 0.75 (standard deviation 0.02) for visceral-first, and 0.80 (standard deviation 0.14) for central-nervous-system-first disease. The central-nervous-system estimate is the least stable because the event is rare. These are internal figures. They suggest that the same clinical variables carry some information about site, particularly visceral versus not, and they do not justify a site prediction in clinic.",
    )
    add_body(
        doc,
        "In the hospital cohort the 23 DRFI events with a recorded first site were bone in 11, visceral in 7, central nervous system in 4, and other in 1 (Figure M11, right panel). No external AUC for site is reported. With single-digit counts, a site AUC would be noise. Figure M07, a heatmap of receptor status against first site in the development data, is biological context rather than a performance claim: receptor patterns differ by site in the way clinicians already recognize (for example, the relative contribution of HER2 and of hormone-receptor-negative disease to visceral and central nervous system events), and the heatmap is included so that scenario C is not detached from that biology.",
    )

    add_h(doc, "What the eleven figures are for", level=2)
    add_body(
        doc,
        "The manuscript figures are the results, not an appendix of exploratory plots. M01 establishes case mix. M02 fixes the endpoint and the legal inputs of the model. M03 names the clinical drivers. M04 and M05 answer whether the score travels and whether its probabilities and decision curves are usable. M06 shows separation that does not require a metric to see. M07 keeps site biology visible. M08 is the one-panel summary of the three scenarios, and M09, M10, and M11 are the scenario-specific figures a reader can be sent to from the abstract. Performance numbers in the text are the numbers drawn on those figures.",
    )

    add_h(doc, "Discussion")
    add_body(
        doc,
        "This study started from a clinical sequence that is easy to state and hard to model. After surgery, early recognition of a still-limited distant relapse can keep curative-intent treatment in reach. Many groups have tried to predict that relapse. Genomic panels changed chemotherapy decisions, but reimbursement still keeps them out of reach for a large fraction of patients. And even after a genomic result and after systemic therapy, breast cancer can recur because it is a systemic disease. We asked whether a model built only from open clinical data could address the three questions that follow from that sequence, and whether the answer would still be visible in a real hospital cohort. For the first question—will distant recurrence or breast-cancer death occur—the answer is a cautious yes. For the second—when—the answer is that the signal is strongest early and is still present, weaker, at ten years. For the third—where—the answer is that the development data contain a signal that this external series is too small to confirm.",
    )
    add_body(
        doc,
        "The development results sit where a methods-aware clinician would expect them to sit. A Nottingham-index Cox model was a serious baseline, not a straw man, and it achieved a 5-year time-dependent AUC of 0.69. Adding the full clinical vector, and allowing the coefficients to differ by ER, gained roughly four points of AUC. The random survival forest gained a little more, to 0.73, with a nearly identical Brier score to the ER-stratified Cox model. That pattern is familiar from other clinical prediction problems: a flexible model extracts a small additional signal from interactions, and most of the usable information was already in a transparent regression. We therefore do not present the forest as a black box that “discovered” recurrence. We present it as the best-ranking member of a set that also contains models an oncologist can write down. If a journal or a reviewer prefers the ER-stratified Cox model for interpretability, little discrimination is lost. Figure M03 exists so that preference can be exercised in the open.",
    )
    add_body(
        doc,
        "External performance needs the same restraint. Point estimates above the internal AUCs are not evidence that the model “works better” in the authors’ hospital. The cohort is younger, more often treated with chemotherapy, and enriched for HER2-positive disease relative to a pre-trastuzumab training set. Case mix can move an AUC in either direction. The bootstrap intervals are wide and overlap across models. What the external study does support is transport of ranking: the order of models was preserved, the forest’s interval lay entirely above 0.74, risk tertiles separated on Kaplan–Meier curves, and patients with a 5-year event carried a clearly higher mean score (0.44 versus 0.24). Those are the findings we would take into a clinical conference. We would not take the raw predicted percentages. A calibration slope of about 1.65 means the percentages need local adjustment before anyone is told a number. Recalibration brought the slope to 1 in this sample; that step should be repeated, and reported, if the score is ever used for counseling.",
    )
    add_body(
        doc,
        "Scenario B is the finding with the most direct link to how clinics already think about time. Early distant relapse is a different clinical object from a relapse in year 8 of an endocrine-therapy course. The external AUC falling from 0.91 at 2 years to 0.74 at 10 years says the chart variables we used—size, nodes, grade, receptors, and the treatments recorded at baseline—are better at marking aggressive early dissemination than at marking late endocrine resistance. That is consistent with the biology that motivated genomic late-recurrence assays in the first place. It is also a boundary on our claim. A tool with this shape can help discuss who needs closer early follow-up. It is the wrong tool, used alone, to decide five extra years of an aromatase inhibitor.",
    )
    add_body(
        doc,
        "Scenario C was included because the organ of first relapse changes symptoms, imaging, and the chance that local ablative treatment is still rational. The internal cause-specific AUCs, around 0.70 to 0.80, show that site is not pure noise given these covariates, and the very small standard deviation for visceral-first events suggests that signal is the most repeatable of the three. The central nervous system result should not be over-read: 31 development events and a standard deviation of 0.14 produce a flattering point estimate. Externally we observed the site mix clinicians recognize—bone most frequent, then visceral, then central nervous system, with a single other site—but observation of counts is not validation of a site predictor. Until a multicenter series with hundreds of site-specific events is scored with a locked model, scenario C remains a research question attached to a model whose validated job is scenario A.",
    )
    add_body(
        doc,
        "Placing the work among existing tools requires saying what it is not. It is not Oncotype DX, which was built and prospectively validated to decide chemotherapy in hormone-receptor-positive, HER2-negative disease. It is not MammaPrint, which was tested in MINDACT for the same class of decision. It is not EndoPredict, which targets late distant recurrence and extended endocrine therapy. Those assays measure RNA from the tumor. This model measures the chart. Patients who can obtain a genomic assay should not be told that a METABRIC forest replaces it. Patients who cannot obtain one—because reimbursement does not cover it, because the tumor block is insufficient, or because the clinical subtype sits outside the assay’s indication—currently rely on grade, nodes, and receptor status in any case. A model that uses exactly those variables, shows its drivers, beats the Nottingham index by a modest margin, and keeps its ranking in a new hospital is a more honest decision aid than an informal impression, provided its probabilities are recalibrated and its limits on timing and site are spoken aloud.",
    )
    add_body(
        doc,
        "The modeling choices were deliberately conservative for a clinical journal. We did not tune a large hyperparameter search against the external cohort, which would have turned validation into a second training set. We did not use deep learning; with a few thousand patients and a dozen clinical inputs, a penalized Cox model and a modest forest are the methods the sample size can support, and they are the methods readers can compare with the rest of the survival literature. We reported time-dependent AUC and a censoring-aware concordance index rather than a naive accuracy at an arbitrary cut-off, because DRFI is a time-to-event outcome. We reported calibration and a decision curve because a high AUC with a dishonest probability is not clinically usable. We reported a prespecified sensitivity cohort with complete recurrence histories so that the main result is not an artifact of incomplete relapse forms. These are the elements clinical readers now expect when a paper says that machine learning was used, and they are more informative here than a longer catalogue of secondary metrics.",
    )
    add_body(
        doc,
        "Several limitations follow directly from the design. The development cohort is historical and largely pre-trastuzumab, and anti-HER2 therapy, CDK4/6 inhibitors, and contemporary chemotherapy are not in the training treatments. The external HER2-positive fraction was higher than in METABRIC, so part of the case mix the model saw at validation did not exist, therapeutically, when the model was trained. Treatment variables are confounded by indication. Missing PR and HER2 were common and were handled with indicators; that is preferable to deletion, but an indicator learned in 1990s–2000s data may not mean the same thing in a hospital where HER2 is almost never missing. The external cohort has 80 patients and 23 events. Intervals are wide, calibration is imprecise, decision curves are unstable, and site-specific external performance cannot be estimated. Tertile Kaplan–Meier curves are descriptive. We did not have Ki-67, so we could not benchmark CTS5. We did not compare the score with a genomic assay head-to-head, because the assay results were not in the chart. Follow-up and cause-of-death assignment in both cohorts carry the usual retrospective error. Finally, a single external center cannot establish geographic transport.",
    )
    add_body(
        doc,
        "The next useful study is not a more complex algorithm. It is a larger, prospective, or at least multicenter, external series with locked predictions, a prespecified recalibration step, and enough site-specific events to accept or retire scenario C. Within hormone-receptor-positive, HER2-negative disease, a comparison against patients who did and did not have a reimbursed genomic assay would show whether the clinical score adds anything once the assay is known, and whether it is a reasonable surrogate where the assay is not. Until then, the appropriate use of this work is as a fully documented research model: open training data, a STEEP endpoint, a Nottingham baseline, an external cohort that was not used for fitting, and three separately graded claims.",
    )

    add_h(doc, "Conclusions")
    add_body(
        doc,
        "Using publicly available METABRIC clinical data, we built a distant recurrence-free interval model from variables that are already in the oncology chart, and we tested it in a single-center surgical cohort. The model ranks patients by the risk of distant recurrence or breast-cancer death, the random survival forest being slightly ahead of a transparent ER-stratified Cox model and clearly ahead of the Nottingham Prognostic Index. In 80 external patients the ranking was preserved, early discrimination was stronger than late discrimination, and higher scores marked patients who relapsed. Absolute probabilities require local recalibration. Prediction of the first metastatic organ is not externally validated. The score does not replace Oncotype DX, MammaPrint, or EndoPredict where those assays are indicated and accessible. Where they are not, a recalibrated clinical score of this kind is a coherent way to make the residual, systemic risk of breast cancer explicit.",
    )

    add_h(doc, "Tables")
    add_table(
        doc,
        ["Characteristic", "METABRIC development (n=3,208)", "External cohort (n=80)"],
        [
            ["DRFI events, n (%)", "1,251 (39.0)", "23 (28.8)"],
            ["Median follow-up", "Long-term (source cohort)", "7.3 years"],
            ["Median age, years", "61.9", "54.9"],
            ["Median tumor size, mm", "about 25", "28"],
            ["Median positive nodes", "0", "—"],
            ["ER positive, %", "71.5", "73.8"],
            ["PR positive, %", "52.8 of those tested*", "55.7"],
            ["HER2 positive, %", "12.3 of those tested*", "20.3"],
            ["PR or HER2 missing, n", "1,255 each", "rare / as recorded"],
            ["Postmenopausal, %", "78.1", "66.7"],
            ["Grade 3, n (%)", "1,548 (48.3)", "—"],
            ["Lobular histology, %", "12.0", "—"],
            ["Chemotherapy, %", "19.6", "66.3"],
            ["Endocrine therapy, %", "55.5", "73.8"],
        ],
        "Table 1. Clinicopathologic profile of the development and external cohorts.",
    )
    add_nofirst(
        doc,
        "*In METABRIC, PR and HER2 were each missing in 1,255 of 3,208 patients (39.1%). Percent positive is among patients with a recorded result (PR 1,031/1,953; HER2 241/1,953). Chemotherapy and endocrine therapy in METABRIC are coded as received versus not received or not recorded. Dashes mark items not summarized in the external descriptive table. DRFI, distant recurrence-free interval.",
        size=10,
        italic=True,
        space_after=10,
    )

    add_table(
        doc,
        ["Model", "Internal 5y tdAUC", "Internal Uno C (10y)", "External 5y tdAUC (95% CI)", "External AUC 2y / 10y"],
        [
            ["NPI-Cox", "0.690", "0.647", "0.811 (0.701–0.912)", "0.817 / 0.712"],
            ["Cox", "0.714", "0.668", "0.811 (0.690–0.915)", "0.783 / 0.727"],
            ["Cox, ER-stratified", "0.729", "0.668", "0.816 (0.711–0.907)", "0.875 / 0.719"],
            ["Random survival forest", "0.732", "0.668", "0.834 (0.741–0.909)", "0.914 / 0.739"],
        ],
        "Table 2. Discrimination for distant recurrence-free interval.",
    )
    add_nofirst(
        doc,
        "Internal values are five-fold cross-validation estimates in 3,208 patients (1,251 events). External values are from a locked model in 80 patients (23 events); the interval is a 400-draw bootstrap confidence interval around the 5-year time-dependent AUC. tdAUC, time-dependent area under the curve; NPI, Nottingham Prognostic Index.",
        size=10,
        italic=True,
        space_after=10,
    )

    add_table(
        doc,
        ["First distant site", "Development events, n", "Internal 5y cause-specific AUC, mean (SD)", "External events, n"],
        [
            ["Bone", "246", "0.70 (0.10)", "11"],
            ["Visceral", "258", "0.75 (0.02)", "7"],
            ["Central nervous system", "31", "0.80 (0.14)", "4"],
            ["Other or unspecified", "537", "not modeled as a target", "1"],
        ],
        "Table 3. Scenario C — first distant site.",
    )
    add_nofirst(
        doc,
        "Development site counts are among patients with a documented distant relapse (1,072), not among all 1,251 DRFI events; 179 breast-cancer deaths had no recorded distant site. Internal AUCs are cross-validated cause-specific estimates. External counts are descriptive. An external site AUC is not reported. SD, standard deviation across folds.",
        size=10,
        italic=True,
        space_after=10,
    )

    add_h(doc, "Figure legends")
    legends = [
        "Figure M01. Clinicopathologic features in the METABRIC development cohort and in the external hospital cohort. The panel is the case-mix check that should be read before performance figures. File: M01_cohort_features_training_vs_center.",
        "Figure M02. Model card. Endpoint (STEEP distant recurrence-free interval), clinical inputs in full words, exclusions (stage 4, in situ and non-invasive histologies), and the four models compared. File: M02_model_card.",
        "Figure M03. Explainability of the clinical Cox model. SHAP-style attributions (standardized feature value × coefficient) show which inputs push predicted DRFI risk up or down. This is an explanation of the linear model, not a tree-level explanation of the random survival forest. File: M03_model_explainability_SHAP.",
        "Figure M04. Internal versus external 5-year time-dependent AUC. Blue marks internal cross-validation; coral marks the external point estimate with bootstrap confidence interval. The random survival forest is highlighted. Numeric callouts were omitted so that intervals are not crossed by labels. File: M04_internal_vs_external_performance.",
        "Figure M05. External calibration, distribution of predicted risk by 5-year outcome, and decision curve. The decision-curve legend sits outside the plotting panel. Net benefit is compared with treat-all and treat-none strategies and is not a treatment recommendation. File: M05_external_calibration_DCA.",
        "Figure M06. Kaplan–Meier curves of DRFI by tertile of predicted risk in the external cohort. A visual check that high scores fail earlier. File: M06_KM_by_risk_tertile.",
        "Figure M07. Receptor status by first distant site in the development data. Biological context for scenario C; not a performance claim. File: M07_receptor_by_first_site.",
        "Figure M08. The three scenarios on one page. (A) Will DRFI occur — internal and external 5-year AUC. (B) Timing — external time-dependent AUC at 2, 5, and 10 years. (C) First site — internal cause-specific AUC, with external counts noted as descriptive and external site AUC explicitly not claimed. File: M08_three_scenarios_AI_performance.",
        "Figure M09. Scenario A in the external cohort. Predicted 5-year risk from the random survival forest in patients with and without a DRFI event by 5 years (n=14 versus n=66). Mean risk 0.44 versus 0.24. File: M09_scenario_A_risk_separation.",
        "Figure M10. Scenario B. External time-dependent AUC and Brier score at 2, 5, and 10 years for all four models. Discrimination is strongest at the early horizon. File: M10_scenario_B_timing.",
        "Figure M11. Scenario C. Left: internal cause-specific 5-year AUC for bone, visceral, and central nervous system first sites, labeled inside the bars so that error bars remain readable. Right: external first-site counts only. File: M11_scenario_C_first_site.",
    ]
    for leg in legends:
        add_nofirst(doc, leg, size=11, space_after=6)

    add_h(doc, "References")
    refs = [
        "Hudis CA, Barlow WE, Costantino JP, et al. Proposal for standardized definitions for efficacy end points in adjuvant breast cancer trials: the STEEP system. J Clin Oncol. 2007;25(15):2127-2132.",
        "Curtis C, Shah SP, Chin SF, et al. The genomic and transcriptomic architecture of 2,000 breast tumours reveals novel subgroups. Nature. 2012;486(7403):346-352.",
        "Pereira B, Chin SF, Rueda OM, et al. The somatic mutation profiles of 2,433 breast cancers refine their genomic and transcriptomic landscapes. Nat Commun. 2016;7:11479.",
        "Galea MH, Blamey RW, Elston CE, Ellis IO. The Nottingham Prognostic Index in primary breast cancer. Breast Cancer Res Treat. 1992;22(3):207-219.",
        "Paik S, Shak S, Tang G, et al. A multigene assay to predict recurrence of tamoxifen-treated, node-negative breast cancer. N Engl J Med. 2004;351(27):2817-2826.",
        "Sparano JA, Gray RJ, Makower DF, et al. Adjuvant chemotherapy guided by a 21-gene expression assay in breast cancer. N Engl J Med. 2018;379(2):111-121.",
        "Cardoso F, van’t Veer LJ, Bogaerts J, et al. 70-gene signature as an aid to treatment decisions in early-stage breast cancer. N Engl J Med. 2016;375(8):717-729.",
        "Filipits M, Rudas M, Jakesz R, et al. A new molecular predictor of distant recurrence in ER-positive, HER2-negative breast cancer adds independent information to conventional clinical risk factors. Clin Cancer Res. 2011;17(18):6012-6020.",
        "Early Breast Cancer Trialists’ Collaborative Group (EBCTCG). Comparisons between different polychemotherapy regimens for early breast cancer: meta-analyses of long-term outcome among 100,000 women in 123 randomised trials. Lancet. 2012;379(9814):432-444.",
        "Pan H, Gray R, Braybrooke J, et al. 20-year risks of breast-cancer recurrence after stopping endocrine therapy at 5 years. N Engl J Med. 2017;377(19):1836-1846.",
        "Cox DR. Regression models and life-tables. J R Stat Soc Series B Stat Methodol. 1972;34(2):187-220.",
        "Ishwaran H, Kogalur UB, Blackstone EH, Lauer MS. Random survival forests. Ann Appl Stat. 2008;2(3):841-860.",
        "Uno H, Cai T, Pencina MJ, D’Agostino RB, Wei LJ. On the C-statistics for evaluating overall adequacy of risk prediction procedures with censored survival data. Stat Med. 2011;30(10):1105-1117.",
        "Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating prediction models. Med Decis Making. 2006;26(6):565-574.",
        "Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ. 2024;385:e078378.",
        "Lundberg SM, Lee SI. A unified approach to interpreting model predictions. Adv Neural Inf Process Syst. 2017;30:4765-4774.",
        "Dowsett M, Sestak I, Regan MM, et al. Integration of clinical variables for the prediction of late distant recurrence in patients with estrogen receptor-positive breast cancer treated with 5 years of endocrine therapy: CTS5. J Clin Oncol. 2018;36(19):1941-1948.",
    ]
    for i, ref in enumerate(refs, start=1):
        add_nofirst(doc, f"{i}. {ref}", size=11, space_after=4)

    add_h(doc, "Author notes for revision (delete before submission)")
    notes = [
        "Insert author names, affiliations, corresponding author, word counts, running title, and conflict-of-interest and funding statements.",
        "Insert the ethics approval identifier for the 80-patient cohort and state the consent approach.",
        "Name the hospital in the methods only if the authorship group agrees to identify the site.",
        "Figure call-outs use the file prefixes M01–M11 so they match PMJ/manu/figures. Renumber to Figure 1–11 when the target journal’s order is fixed. Suggested narrative order is M01, M02, M03, M04, M05, M06, then M08 with M09–M11, and M07 as biology.",
        "Do not upgrade scenario C to an external performance claim. Do not quote external AUCs without the bootstrap intervals in Table 2.",
        "The calibration slope (approximately 1.65 raw; approximately 1.00 after recalibration) is a main limitation and should stay in the discussion if the results section is shortened.",
        "SHAP in Figure M03 is the Cox linear attribution, not TreeSHAP of the random survival forest. Keep that sentence if a reviewer asks which model is explained.",
        "Software versions were those in the analysis environment: Python 3.10.11, scikit-survival 0.25.0, scikit-learn 1.7.1, lifelines 0.30.0, shap 0.49.1, pandas 2.3.1, NumPy 1.26.4, Matplotlib 3.10.3.",
        "This draft was written to be edited. It is deliberately longer than a typical original article so that cuts can be made from a complete raw report rather than from an outline.",
    ]
    for n in notes:
        add_nofirst(doc, n, size=11, space_after=4)

    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    build()
