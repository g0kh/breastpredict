# Cursor konuşma özeti — diğer bilgisayarda devam

Tarih: 2026-11-23/24. Amaç: METABRIC’ten klinik hasta listesi → metastaz zamanı / risk / bölge modellemesi.

## Ne yapıldı

1. cBioPortal METABRIC klinik tabloları incelendi (2509 hasta).
2. Rueda 2019 ek tabloları eklendi / tarandı:
   - Supp Table 5: 3240 hasta; uzak metastaz süresi gün cinsinden (`TDR`, `DR`)
   - Supp Table 7: nüks bölgesi; ilk uzak bölge 618 hastada çıkarılabilir
3. `database.xlsx` üretildi (omurga = Table 5; PR/HER2 = cBioPortal sample; ilk uzak bölge = Table 7).
4. Ki-67 ve LVI bu paketlerde yok → vazgeçildi; sütun açılmadı (sözlükte not var).

## `database.xlsx` sütunları

`hasta_id`, `yas`, `menopoz`, `tumor_boyutu_mm`, `pozitif_ln`, `grade`, `histoloji`, `er`, `pr`, `her2`, `cerrahi`, `kemoterapi`, `hormon_tedavisi`, `radyoterapi`, `uzak_metastaz`, `uzak_metastaz_gun`, `uzak_metastaz_ay`, `ilk_uzak_metastaz_bolgesi`, `nuks_oykusu_tam`

## Önemli sayılar

| Tanım | n |
|---|---|
| Toplam hasta | 3240 |
| Uzak metastaz var (DR=1) | 1080 |
| Uzak metastaz yok (DR=0) | 2160 |
| İlk uzak bölge dolu | 618 |
| Klinik alanlar + PR/HER2 dolu (bölge şart değil / DR=0’da bölge OK) | ~1681 |
| PR/HER2 eksik (Table5–cBio örtüşmesi dışı) | ~1260 |

### Zaman dilimi sınıfları (önerilen etiketsel yaklaşım)

Kullanıcı hedefi: 1. yıl / 2. yıl / 3. yıl / 4. yıl / 5. yıl ve sonrası + metastazsız.

| Sınıf | n (tüm veri, yaklaşık) |
|---|---|
| 1. yıl içinde | 116 |
| 2. yıl | 213 |
| 3. yıl | 181 |
| 4. yıl | 125 |
| 5. yıl ve sonrası | 445 |
| Metastaz yok ve takip ≥5 yıl | 1856 |
| Metastaz yok ama takip &lt;5 yıl | 304 → **etiketten çıkar** (sansür; “yok” yazılamaz) |

Complete-case (PR/HER2 dahil) ve etiketi net: ~1662 (no_met 1089; y1–y5_plus: 53/112/100/70/238).

## Modelleme hükümleri (konuşmadan)

- **Metastaz zamanı / risk (survival veya 6 sınıf):** sayılar genelde yeterli; olay ~570–1080.
- **İnce anatomik ilk bölge:** 416–618, dengesiz → zayıf; kaba grup (kemik / visseral / SSS) daha gerçekçi.
- Tedavi değişkenlerinde endikasyon karışıklığına dikkat.
- `uzak_metastaz=0` iken `uzak_metastaz_gun` = sansür süresi, olay tarihi değil.
- `NO/NA` tedavi kodları “verilmedi” ile “bilinmiyor”u ayırmaz.
- Sınıflar dengesiz → accuracy yerine ağırlıklı / makro-F1 veya survival tercih edilmeli.

## Sonraki adım (diğer PC)

1. `git clone https://github.com/g0kh/breastpredict.git`
2. `database.xlsx` ile etiket üret (5 zaman + no_met; erken sansürlü DR0 çıkar).
3. Survival veya sınıflandırma baseline (Cox / RSF / GBM).
4. İstenirse bölge modeli ayrı, kaba etiketle.

## Notlar

- MB-0550: Table5 `TDR=2767`, Table7 ilk DISTANT SITE zamanı 3970; sürede Table5, bölgede Table7 kullanıldı.
- ER klinik (Table5) ile cBioPortal ER_IHC/ER_STATUS her zaman birebir değil; listede ER = Table5.
- HER2 = cBioPortal `HER2_STATUS` (SNP6 kopya sayısı değil).
