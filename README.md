# breastpredict — METABRIC DRFI risk modelleme

Meme kanserinde **uzak nüks / meme kanserine bağlı ölüm (STEEP DRFI)** riskini klinik + patoloji (ER/PR/HER2) ile tahmin etme çalışması.

## Hızlı başlangıç (başka bilgisayar)

```bash
git clone https://github.com/g0kh/breastpredict.git
cd breastpredict
python -m venv .venv
# Windows:
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Ana (güncel) pipeline:

```bash
python run_drfi_pipeline.py
```

Çıktılar: [`sonuclar/`](sonuclar/)  
External veri toplama şablonu: [`sonuclar/External_Validasyon_Veri_Sablonu.xlsx`](sonuclar/External_Validasyon_Veri_Sablonu.xlsx)

## Ne yapıyoruz? (3 senaryo)

| Senaryo | Soru | Güncel çözüm | Durum |
|---|---|---|---|
| **A** | Nüks/met riski var mı? | DRFI sağkalım (RSF / Cox_ER / Cox) | Ana model — kullan |
| **B** | 1–5 yılda risk? | Aynı DRFI modelinden kümülatif risk (5y/10y tdAUC) | A ile birleşik |
| **C** | İlk metastaz nerede? | Competing-risk (kemik / visseral / SSS) | İkinci öncelik |

Eski binary M1 / 6-sınıf M2A / M2B kurguları **yöntemsel olarak geçersiz** sayıldı; arşiv scriptleri duruyor ama **güncel sonuç = `sonuclar/`**.

## Güncel performans (internal, özet)

Temiz kohort ≈ **3208** hasta, DRFI olay ≈ **1251**.

| Model | 5y tdAUC | Uno C (10y) |
|---|---:|---:|
| NPI_Cox (baseline) | 0.69 | 0.65 |
| Cox | 0.71 | 0.67 |
| Cox_ER_ayrı | 0.73 | 0.67 |
| **RSF** | **0.73** | 0.67 |

External validasyon için önerilen: **Cox_ER_ayrı + RSF + NPI baseline** (rekalibrasyon şart; pre-trastuzumab dönem).

## Veri

- [`database.xlsx`](database.xlsx) — birleştirilmiş klinik liste (3240 hasta)
- [`brca_metabric/`](brca_metabric/) — Supp Table 5/7 + cBioPortal klinik
- Omics (mRNA/CNA/metilasyon) repoda yok (boyut limiti)

## Önemli scriptler

| Dosya | Rol |
|---|---|
| `run_drfi_pipeline.py` | **Güncel** DRFI + site competing-risk → `sonuclar/` |
| `demo_duzeltmeler.py` | Metodolojik kanıt / hızlı deneme |
| `create_external_template.py` | External Excel şablonu üretir |
| `run_full_modeling.py` / `run_report2.py` | Eski binary/hibrit arşiv (M1–M2B) |
| `scan_cohorts.py` | Kohort uygunluk taraması |

## Çıktı klasörleri

- **`sonuclar/`** — güncel rapor (`report_drfi.html`), tablolar, figürler, external şablon
- `outputs/` — eski Report1/2 ara çıktıları
- `BreastPredict_Rapor.html`, `report2.html`, `NIHAI_RAPOR_DENETIM.md` — arşiv raporlar

## Metodoloji (kısa)

- Dışla: evre 4, BENIGN/PHYL/DCIS, T≤0
- Sonlanım: DRFI = uzak nüks **veya** meme kanseri ölümü; başka ölüm = sansür
- Özellik: age, log(size), log1p(LN), grade, ER/PR/HER2, menopoz, lobular, CT/HT var-yok (+ missing indicator); **RT/cerrahi yok**
- Impute/scale **fold içinde**; class_weight yok
- Metrik: time-dependent AUC, Uno C, IPCW Brier, kalibrasyon, DCA

Devam notları: [`DEVAM.md`](DEVAM.md)
