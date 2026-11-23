# breastpredict — METABRIC klinik hasta listesi

Bu depo, meme kanseri METABRIC verisinden türetilmiş klinik hasta listesini ve modelleme notlarını içerir.

## Ana çıktı

- [`database.xlsx`](database.xlsx)
  - Sayfa `hastalar`: 3.240 hasta, hasta başına 1 satır
  - Sayfa `sozluk`: sütun kaynağı, birim, eksik kuralı

## Kaynaklar (repoda)

- `brca_metabric/NIHMS1520488-supplement-Supp_Table_5.txt` — omurga klinik + uzak metastaz süresi (`TDR`/`DR`)
- `brca_metabric/NIHMS1520488-supplement-Supp_Table_7.txt` — nüks bölgesi satırları
- `brca_metabric/brca_metabric/data_clinical_patient.txt` — cBioPortal hasta klinik
- `brca_metabric/brca_metabric/data_clinical_sample.txt` — PR / HER2

## Repoda olmayan (çok büyük)

mRNA, metilasyon, CNA ve `brca_metabric.tar.gz` GitHub limitinden dolayı yok. Klinik modelleme için gerekmez.

## Devam notları

Ayrıntılı kararlar ve modelleme planı: [`DEVAM.md`](DEVAM.md)
