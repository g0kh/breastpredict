# Devam notları — güncel durum

Tarih: 2026-11-24 (güncellendi).

## Bu PC’de tamamlananlar

1. Repo klonlandı; Python 3.12 + `requirements.txt` kuruldu.
2. `database.xlsx` tarandı; eski M1/M2A/M2B/M3 binary-çoklu sınıf pipeline (Report1/2) çalıştırıldı.
3. Metodolojik düzeltmeler (`demo_duzeltmeler.py` + eleştiri) uygulandı → **`run_drfi_pipeline.py`**.
4. Güncel çıktılar **`sonuclar/`** altında.
5. External validasyon veri toplama Excel’i üretildi.

## Güncel “doğru” çerçeve

- **Kullan:** DRFI sağkalım (Senaryo A/B) + site competing-risk (Senaryo C)
- **Kullanma (ana sonuç olarak):** M1 binary, M2A 6-sınıf, M2B “5y binary + kısa FU at”
- External aday modeller: Cox_ER_ayrı, RSF, NPI_Cox baseline

## Başka PC’de devam

```bash
git clone https://github.com/g0kh/breastpredict.git
cd breastpredict
python -m pip install -r requirements.txt
python run_drfi_pipeline.py   # isteğe bağlı yeniden üretim
```

1. Hastanede `sonuclar/External_Validasyon_Veri_Sablonu.xlsx` doldur.
2. External kohortu aynı özellik / DRFI tanımı ile skorla; **rekalibrasyon** yap (özellikle HER2 / trastuzumab dönemi).
3. Metrik: 5y/10y tdAUC, Uno C, IPCW Brier, kalibrasyon, DCA — accuracy/F1 değil.
4. İstersen 1–4y tdAUC satırlarını `run_drfi_pipeline.py` içine ekle.

## Bilinen sınırlar

- CTS5 yok (Ki-67 yok).
- Pre-trastuzumab eğitim → modern HER2+’ta rekalibrasyon şart.
- Kohort 3 (eksik nüks öyküsü) gürültülü; duyarlılık analizi `sonuclar/` içinde.
- Omics repoda yok.

## Arşiv dosyaları

`BreastPredict_Rapor.html`, `report2.html`, `outputs/`, `run_full_modeling.py`, `run_report2.py` — tarihsel; ana iddia `sonuclar/report_drfi.html`.
