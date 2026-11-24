"""Scan database.xlsx for model-ready cohort sizes."""
from __future__ import annotations

import numpy as np
import pandas as pd

df = pd.read_excel("database.xlsx", sheet_name="hastalar")
n = len(df)
print("=== GENEL ===")
print(f"Toplam hasta: {n}")
print("Kolonlar:", list(df.columns))
print()

core_feat = [
    "yas",
    "menopoz",
    "tumor_boyutu_mm",
    "pozitif_ln",
    "grade",
    "histoloji",
    "er",
    "cerrahi",
    "kemoterapi",
    "hormon_tedavisi",
    "radyoterapi",
]
path_feat = ["pr", "her2"]
all_feat = core_feat + path_feat

for c in df.columns:
    if df[c].dtype == object:
        df[c] = df[c].replace({"": np.nan, " ": np.nan})

core_ok = df[core_feat].notna().all(axis=1)
path_ok = df[path_feat].notna().all(axis=1)
full_ok = core_ok & path_ok

print("=== OZELLIK DOLULUK ===")
print(f"Klinik cekirdek (Table5, PR/HER2 haric) dolu: {int(core_ok.sum())}")
print(f"PR+HER2 dolu: {int(path_ok.sum())}")
print(f"Klinik+PR+HER2 (complete-case) dolu: {int(full_ok.sum())}")
print()
miss = (
    df[all_feat + ["uzak_metastaz", "uzak_metastaz_gun", "ilk_uzak_metastaz_bolgesi", "nuks_oykusu_tam"]]
    .isna()
    .sum()
    .sort_values(ascending=False)
)
print("Eksik sayilari:")
print(miss.to_string())
print()

df["gun"] = pd.to_numeric(df["uzak_metastaz_gun"], errors="coerce")
df["ay"] = pd.to_numeric(df["uzak_metastaz_ay"], errors="coerce")
df["met"] = pd.to_numeric(df["uzak_metastaz"], errors="coerce")
YEAR = 365.25

print("=== MODEL 1: Metastaz var/yok (binary) ===")
print(
    f"DR etiketi dolu: {int(df['met'].notna().sum())} | "
    f"met=1: {int((df.met == 1).sum())} | met=0: {int((df.met == 0).sum())}"
)
print(
    f"  + klinik cekirdek: {int(core_ok.sum())}  "
    f"(met1={int((core_ok & (df.met == 1)).sum())}, met0={int((core_ok & (df.met == 0)).sum())})"
)
print(
    f"  + klinik+PR+HER2: {int(full_ok.sum())}  "
    f"(met1={int((full_ok & (df.met == 1)).sum())}, met0={int((full_ok & (df.met == 0)).sum())})"
)
print()


def exclusive_label(row):
    if pd.isna(row.met) or pd.isna(row.gun):
        return None
    if row.met == 1:
        y = row.gun / YEAR
        if y < 1:
            return "y1"
        if y < 2:
            return "y2"
        if y < 3:
            return "y3"
        if y < 4:
            return "y4"
        return "y5_plus"
    if row.gun >= 5 * YEAR:
        return "no_met_ge5y"
    return "censor_lt5y"


df["ex_label"] = df.apply(exclusive_label, axis=1)
print("=== MODEL 2A: Zaman siniflari (birbirini dislayan) ===")
print("Tum veri:")
print(df["ex_label"].value_counts(dropna=False).to_string())
print()
usable_ex = df["ex_label"].isin(["y1", "y2", "y3", "y4", "y5_plus", "no_met_ge5y"])
print(f"Etiketi net (erken sansur haric): {int(usable_ex.sum())}")
print(f"  + klinik cekirdek: {int((usable_ex & core_ok).sum())}")
print(df.loc[usable_ex & core_ok, "ex_label"].value_counts().to_string())
print(f"  + klinik+PR+HER2: {int((usable_ex & full_ok).sum())}")
print(df.loc[usable_ex & full_ok, "ex_label"].value_counts().to_string())
print()

print("=== MODEL 2B: Sadece metastaz olacaklarda zaman sinifi (y1..y5+) ===")
met1 = df.met == 1
print(f"met=1 toplam: {int(met1.sum())}")
print(df.loc[met1, "ex_label"].value_counts().to_string())
print(f"  + klinik: {int((met1 & core_ok).sum())}")
print(df.loc[met1 & core_ok, "ex_label"].value_counts().to_string())
print(f"  + klinik+PR+HER2: {int((met1 & full_ok).sum())}")
print(df.loc[met1 & full_ok, "ex_label"].value_counts().to_string())
print()

print("=== MODEL 2C: Kumule ufuk (k. yildan once met olacak mi?) ===")
print("Kural: event <k yil -> 1; k yila kadar event yok / event k'dan sonra -> 0; FU<k ve event yok -> haric")
for k in [1, 2, 3, 4, 5]:
    pos = (df.met == 1) & (df.gun < k * YEAR)
    neg = ((df.met == 0) & (df.gun >= k * YEAR)) | ((df.met == 1) & (df.gun >= k * YEAR))
    usable = pos | neg
    early = (df.met == 0) & (df.gun < k * YEAR)
    print(
        f"  {k}. yildan once: usable={int(usable.sum())} | "
        f"pos={int(pos.sum())} | neg={int(neg.sum())} | erken_sansur={int(early.sum())}"
    )
    print(
        f"      +klinik: {int((usable & core_ok).sum())} "
        f"(pos={int((usable & core_ok & pos).sum())}, neg={int((usable & core_ok & neg).sum())})"
    )
    print(
        f"      +klinik+PR+HER2: {int((usable & full_ok).sum())} "
        f"(pos={int((usable & full_ok & pos).sum())}, neg={int((usable & full_ok & neg).sum())})"
    )
print()

print("=== MODEL 3: Metastaz bolgesi ===")
site = df["ilk_uzak_metastaz_bolgesi"]
print(f"Bolge dolu: {int(site.notna().sum())}")
print(f"  met=1 & bolge dolu: {int(((df.met == 1) & site.notna()).sum())}")
print(f"  met=1 & bolge bos: {int(((df.met == 1) & site.isna()).sum())}")
print("nuks_oykusu_tam:")
print(df["nuks_oykusu_tam"].value_counts(dropna=False).to_string())


def coarse(s):
    if pd.isna(s):
        return None
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


df["site_coarse"] = site.map(coarse)
print("Ince (ham) bolge top 15:")
print(site.value_counts().head(15).to_string())
print("Kaba grup:")
print(df["site_coarse"].value_counts(dropna=False).to_string())
print(f"Bolge modeli + klinik: {int((site.notna() & core_ok).sum())}")
print(df.loc[site.notna() & core_ok, "site_coarse"].value_counts().to_string())
print(f"Bolge modeli + klinik+PR+HER2: {int((site.notna() & full_ok).sum())}")
print(df.loc[site.notna() & full_ok, "site_coarse"].value_counts().to_string())

print()
print("=== BORU HATTI UYGUNLUK (ozet) ===")
print(f"M1 binary complete-case: {int(full_ok.sum())}")
print(f"M2 exclusive 6-class complete-case: {int((usable_ex & full_ok).sum())}")
print(f"M2 only-met time complete-case: {int((met1 & full_ok).sum())}")
print(f"M3 site fine complete-case: {int((site.notna() & full_ok).sum())}")
print(f"M3 site coarse complete-case: {int((df.site_coarse.notna() & full_ok).sum())}")

summary_rows = []
summary_rows.append(
    {
        "model": "M1_binary_met",
        "feature_set": "klinik",
        "n": int(core_ok.sum()),
        "notes": f"met1={int((core_ok & (df.met == 1)).sum())}, met0={int((core_ok & (df.met == 0)).sum())}",
    }
)
summary_rows.append(
    {
        "model": "M1_binary_met",
        "feature_set": "klinik+PR+HER2",
        "n": int(full_ok.sum()),
        "notes": f"met1={int((full_ok & (df.met == 1)).sum())}, met0={int((full_ok & (df.met == 0)).sum())}",
    }
)
summary_rows.append(
    {
        "model": "M2_exclusive_6class",
        "feature_set": "klinik",
        "n": int((usable_ex & core_ok).sum()),
        "notes": str(df.loc[usable_ex & core_ok, "ex_label"].value_counts().to_dict()),
    }
)
summary_rows.append(
    {
        "model": "M2_exclusive_6class",
        "feature_set": "klinik+PR+HER2",
        "n": int((usable_ex & full_ok).sum()),
        "notes": str(df.loc[usable_ex & full_ok, "ex_label"].value_counts().to_dict()),
    }
)
summary_rows.append(
    {
        "model": "M2_time_among_met",
        "feature_set": "klinik+PR+HER2",
        "n": int((met1 & full_ok).sum()),
        "notes": str(df.loc[met1 & full_ok, "ex_label"].value_counts().to_dict()),
    }
)
summary_rows.append(
    {
        "model": "M3_site_fine",
        "feature_set": "klinik+PR+HER2",
        "n": int((site.notna() & full_ok).sum()),
        "notes": "ince anatomik",
    }
)
summary_rows.append(
    {
        "model": "M3_site_coarse",
        "feature_set": "klinik+PR+HER2",
        "n": int((df.site_coarse.notna() & full_ok).sum()),
        "notes": str(df.loc[df.site_coarse.notna() & full_ok, "site_coarse"].value_counts().to_dict()),
    }
)
for k in [1, 2, 3, 4, 5]:
    pos = (df.met == 1) & (df.gun < k * YEAR)
    neg = ((df.met == 0) & (df.gun >= k * YEAR)) | ((df.met == 1) & (df.gun >= k * YEAR))
    usable = pos | neg
    summary_rows.append(
        {
            "model": f"M2C_before_y{k}",
            "feature_set": "klinik+PR+HER2",
            "n": int((usable & full_ok).sum()),
            "notes": f"pos={int((usable & full_ok & pos).sum())}, neg={int((usable & full_ok & neg).sum())}",
        }
    )

pd.DataFrame(summary_rows).to_excel("cohort_model_ozet.xlsx", index=False)
print("\nKaydedildi: cohort_model_ozet.xlsx")
