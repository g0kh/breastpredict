"""Create hospital data-collection Excel for external validation."""
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "sonuclar" / "External_Validasyon_Veri_Sablonu.xlsx"

wb = Workbook()

# styles
header_fill = PatternFill("solid", fgColor="1F5A4E")
header_font = Font(color="FFFFFF", bold=True, size=11)
req_fill = PatternFill("solid", fgColor="F7EFE4")
opt_fill = PatternFill("solid", fgColor="EAF2F0")
site_fill = PatternFill("solid", fgColor="EDE4F5")
thin = Border(
    left=Side(style="thin", color="CCCCCC"),
    right=Side(style="thin", color="CCCCCC"),
    top=Side(style="thin", color="CCCCCC"),
    bottom=Side(style="thin", color="CCCCCC"),
)
wrap = Alignment(wrap_text=True, vertical="center")

# ---------------------------------------------------------------------------
# Sheet 1: Talimatlar
# ---------------------------------------------------------------------------
ws0 = wb.active
ws0.title = "1_Talimatlar"
ws0["A1"] = "EXTERNAL VALİDASYON — HASTANE VERİ TOPLAMA TALİMATI"
ws0["A1"].font = Font(bold=True, size=16, color="1F5A4E")
ws0.merge_cells("A1:B1")

talimatlar = [
    ("Amaç", "Geliştirilen DRFI (uzak nüks veya meme kanserine bağlı ölüm) risk modelini dış kohortta test etmek."),
    ("Ana senaryo", "Senaryo A/B: Tanı sonrası 1–5–10 yılda DRFI riski. Aynı hasta listesi yeterlidir."),
    ("İkinci senaryo (opsiyonel)", "Senaryo C: İlk uzak metastaz bölgesi (kemik / visseral / beyin). Mor sütunlar."),
    ("Kimler DAHİL", "İnvaziv meme kanseri, primer tedavi amaçlı tanı, evre 0–3 (evre 4 DEĞİL), takip süresi bilinen hastalar."),
    ("Kimler DIŞLANIR", "Tanıda uzak metastaz (evre 4); DCIS; benign; fillodes (phyllodes); takip süresi 0 veya bilinmeyen."),
    ("Zaman sıfır", "Tanı tarihi (veya primer cerrahi tarihi — tüm kohortta aynı kuralı kullanın ve Sozluk'ta belirtin)."),
    ("DRFI olay tanımı (STEEP)", "Olay=1: ilk uzak metastaz/nüks VEYA meme kanserine bağlı ölüm. Başka nedenle ölüm = olay değil (sansür)."),
    ("Süre nasıl yazılır", "Olay olduysa: tanıdan olaya kadar gün. Olay olmadıysa: tanıdan son kontrole (veya başka nedenle ölüme) kadar gün."),
    ("Eksik veri", "Bilinmiyorsa hücreyi BOŞ bırakın. 'Yok' ile 'bilinmiyor'u karıştırmayın. PR/HER2 için ayrı durum sütunları var."),
    ("Tedavi kodu", "Sadece VAR/YOK. Rejim adı (CMF vb.) gerekmez. Radyoterapi ve cerrahi tipi bu model için zorunlu değil."),
    ("Kaç hasta?", "Mümkünse ≥200 (ideal ≥300). Olay sayısı mümkünse ≥50 DRFI olayı."),
    ("KVKK / anonimleştirme", "İsim yazmayın. Merkez içi anonim hasta_kodu kullanın."),
    ("Doldurulacak sayfa", "2_Veri_Girisi — her satır bir hasta. Başlık satırını silmeyin / değiştirmeyin."),
]
ws0["A3"] = "Başlık"
ws0["B3"] = "Açıklama"
ws0["A3"].fill = header_fill
ws0["B3"].fill = header_fill
ws0["A3"].font = header_font
ws0["B3"].font = header_font
for i, (a, b) in enumerate(talimatlar, start=4):
    ws0[f"A{i}"] = a
    ws0[f"B{i}"] = b
    ws0[f"A{i}"].font = Font(bold=True)
    ws0[f"A{i}"].alignment = wrap
    ws0[f"B{i}"].alignment = wrap
ws0.column_dimensions["A"].width = 28
ws0.column_dimensions["B"].width = 100
ws0.row_dimensions[1].height = 28
for r in range(4, 4 + len(talimatlar)):
    ws0.row_dimensions[r].height = 40

# ---------------------------------------------------------------------------
# Sheet 2: Veri girisi
# ---------------------------------------------------------------------------
ws = wb.create_sheet("2_Veri_Girisi")

# (col_key, header_tr, group) group: zorunlu / onerilen / bolge
columns = [
    ("hasta_kodu", "hasta_kodu", "zorunlu"),
    ("tani_tarihi", "tani_tarihi", "zorunlu"),
    ("yas_tani", "yas_tani", "zorunlu"),
    ("menopoz", "menopoz", "zorunlu"),
    ("tumor_boyutu_mm", "tumor_boyutu_mm", "zorunlu"),
    ("pozitif_lenf_nodu_sayisi", "pozitif_lenf_nodu_sayisi", "zorunlu"),
    ("grade", "grade", "zorunlu"),
    ("histoloji", "histoloji", "zorunlu"),
    ("ER", "ER", "zorunlu"),
    ("PR", "PR", "zorunlu"),
    ("HER2", "HER2", "zorunlu"),
    ("kemoterapi_verildi", "kemoterapi_verildi", "zorunlu"),
    ("hormonoterapi_verildi", "hormonoterapi_verildi", "zorunlu"),
    ("evre", "evre", "zorunlu"),
    ("DRFI_olay", "DRFI_olay", "zorunlu"),
    ("DRFI_sure_gun", "DRFI_sure_gun", "zorunlu"),
    ("son_durum", "son_durum", "onerilen"),
    ("olum_nedeni", "olum_nedeni", "onerilen"),
    ("NPI", "NPI", "onerilen"),
    ("radyoterapi_verildi", "radyoterapi_verildi", "onerilen"),
    ("cerrahi_tipi", "cerrahi_tipi", "onerilen"),
    ("ilk_uzak_metastaz_oldu", "ilk_uzak_metastaz_oldu", "bolge"),
    ("ilk_uzak_metastaz_gun", "ilk_uzak_metastaz_gun", "bolge"),
    ("ilk_uzak_bolge", "ilk_uzak_bolge", "bolge"),
    ("notlar", "notlar", "onerilen"),
]

fills = {"zorunlu": req_fill, "onerilen": opt_fill, "bolge": site_fill}

# legend row
ws["A1"] = "RENK:"
ws["B1"] = "Turuncu = ZORUNLU (Senaryo A/B)"
ws["C1"] = "Yeşil = önerilen"
ws["D1"] = "Mor = Senaryo C (bölge, opsiyonel)"
ws["B1"].fill = req_fill
ws["C1"].fill = opt_fill
ws["D1"].fill = site_fill
ws.merge_cells("D1:F1")

for col_i, (key, header, group) in enumerate(columns, start=1):
    cell = ws.cell(2, col_i, header)
    cell.fill = header_fill
    cell.font = header_font
    cell.alignment = Alignment(horizontal="center", wrap_text=True, vertical="center")
    cell.border = thin
    # group color on row 3
    gcell = ws.cell(3, col_i, group)
    gcell.fill = fills[group]
    gcell.alignment = Alignment(horizontal="center")
    gcell.border = thin
    ws.column_dimensions[get_column_letter(col_i)].width = max(14, len(header) + 2)

ws.row_dimensions[2].height = 30
ws.freeze_panes = "B4"

# data validations for 200 rows
dv_meno = DataValidation(type="list", formula1='"pre,post,bilinmiyor"', allow_blank=True)
dv_grade = DataValidation(type="list", formula1='"1,2,3"', allow_blank=True)
dv_er = DataValidation(type="list", formula1='"pozitif,negatif,bilinmiyor"', allow_blank=True)
dv_bin = DataValidation(type="list", formula1='"evet,hayir,bilinmiyor"', allow_blank=True)
dv_hist = DataValidation(
    type="list",
    formula1='"IDC,ILC,IDC+ILC,diger_invaziv,DCIS_DISLA,benign_DISLA,fillodes_DISLA"',
    allow_blank=True,
)
dv_evre = DataValidation(type="list", formula1='"1,2,3,4_DISLA,bilinmiyor"', allow_blank=True)
dv_drfi = DataValidation(type="list", formula1='"0,1"', allow_blank=True)
dv_son = DataValidation(
    type="list",
    formula1='"yasiyor,olum_meme_kanseri,olum_diger,kayip_takip"',
    allow_blank=True,
)
dv_olum = DataValidation(type="list", formula1='"meme_kanseri,diger,bilinmiyor,uygulamaz"', allow_blank=True)
dv_cer = DataValidation(type="list", formula1='"meme_koruyucu,mastektomi,yok,bilinmiyor"', allow_blank=True)
dv_bolge = DataValidation(
    type="list",
    formula1='"kemik,viseral,beyin_SSS,kemik+viseral,diger,bilinmiyor"',
    allow_blank=True,
)

for dv, colname in [
    (dv_meno, "menopoz"),
    (dv_grade, "grade"),
    (dv_er, "ER"),
    (dv_er, "PR"),
    (dv_er, "HER2"),
    (dv_bin, "kemoterapi_verildi"),
    (dv_bin, "hormonoterapi_verildi"),
    (dv_bin, "radyoterapi_verildi"),
    (dv_bin, "ilk_uzak_metastaz_oldu"),
    (dv_hist, "histoloji"),
    (dv_evre, "evre"),
    (dv_drfi, "DRFI_olay"),
    (dv_son, "son_durum"),
    (dv_olum, "olum_nedeni"),
    (dv_cer, "cerrahi_tipi"),
    (dv_bolge, "ilk_uzak_bolge"),
]:
    idx = [c[0] for c in columns].index(colname) + 1
    letter = get_column_letter(idx)
    dv.add(f"{letter}4:{letter}503")
    ws.add_data_validation(dv)

# pre-create 50 empty bordered rows for comfort
for r in range(4, 54):
    for c in range(1, len(columns) + 1):
        ws.cell(r, c).border = thin
        ws.cell(r, c).fill = fills[columns[c - 1][2]]

# ---------------------------------------------------------------------------
# Sheet 3: Sozluk
# ---------------------------------------------------------------------------
ws2 = wb.create_sheet("3_Sozluk")
ws2["A1"] = "Sütun"
ws2["B1"] = "Zorunlu mu?"
ws2["C1"] = "Ne yazılacak?"
ws2["D1"] = "İzin verilen değerler / birim"
ws2["E1"] = "Neden gerekli?"
for col in range(1, 6):
    ws2.cell(1, col).fill = header_fill
    ws2.cell(1, col).font = header_font

sozluk = [
    ("hasta_kodu", "ZORUNLU", "Anonim merkez kodu", "örn. EXT-001", "Eşleştirme; kimlik yok"),
    ("tani_tarihi", "ZORUNLU", "Tanı tarihi", "YYYY-AA-GG", "Takip süresinin başlangıcı"),
    ("yas_tani", "ZORUNLU", "Tanıdaki yaş", "yıl (örn. 54.2)", "Model özelliği"),
    ("menopoz", "ZORUNLU", "Menopoz durumu tanıda", "pre / post / bilinmiyor", "Model özelliği"),
    ("tumor_boyutu_mm", "ZORUNLU", "Primer tümör en büyük çap", "mm", "Model özelliği (log dönüşüm)"),
    ("pozitif_lenf_nodu_sayisi", "ZORUNLU", "Pozitif aksiller LN sayısı", "0,1,2… (yoksa 0)", "Model özelliği"),
    ("grade", "ZORUNLU", "Histolojik grade", "1 / 2 / 3", "Model özelliği"),
    ("histoloji", "ZORUNLU", "Histolojik tip", "IDC / ILC / IDC+ILC / diger_invaziv (DCIS/benign/fillodes = çalışmaya alma)", "Lobüler ayrımı"),
    ("ER", "ZORUNLU", "Östrojen reseptörü", "pozitif / negatif / bilinmiyor", "Klinik+patoloji"),
    ("PR", "ZORUNLU", "Progesteron reseptörü", "pozitif / negatif / bilinmiyor", "Klinik+patoloji"),
    ("HER2", "ZORUNLU", "HER2 durumu", "pozitif / negatif / bilinmiyor (IHC/FISH sonucu)", "Klinik+patoloji; modern kohortta trastuzumab notu önemli"),
    ("kemoterapi_verildi", "ZORUNLU", "Adjuvan/neoadjuvan KT", "evet / hayir / bilinmiyor", "Var/yok (rejim adı gerekmez)"),
    ("hormonoterapi_verildi", "ZORUNLU", "Endokrin tedavi", "evet / hayir / bilinmiyor", "Var/yok"),
    ("evre", "ZORUNLU", "Klinik/patolojik evre", "1/2/3 (4 = dışla)", "Dışlama kontrolü"),
    ("DRFI_olay", "ZORUNLU", "DRFI olayı oldu mu?", "1 = uzak nüks VEYA meme kanseri ölümü; 0 = olmadı", "Ana sonuç (etiket)"),
    ("DRFI_sure_gun", "ZORUNLU", "Tanıdan olay/sansüre gün", "tam sayı gün (ay×30.4375 ile de çevrilebilir)", "Sağkalım süresi"),
    ("son_durum", "önerilen", "Son bilinen durum", "yasiyor / olum_meme_kanseri / olum_diger / kayip_takip", "DRFI kontrolü"),
    ("olum_nedeni", "önerilen", "Ölüm nedeni", "meme_kanseri / diger / bilinmiyor / uygulamaz", "DeathBreast ayrımı"),
    ("NPI", "önerilen", "Nottingham Prognostic Index", "sayı (varsa)", "Baseline karşılaştırma"),
    ("radyoterapi_verildi", "önerilen", "RT verildi mi?", "evet / hayir / bilinmiyor", "Zorunlu değil (modelde yok)"),
    ("cerrahi_tipi", "önerilen", "Cerrahi", "meme_koruyucu / mastektomi / yok / bilinmiyor", "Zorunlu değil"),
    ("ilk_uzak_metastaz_oldu", "Senaryo C", "Uzak met gelişti mi?", "evet / hayir / bilinmiyor", "Bölge modeli"),
    ("ilk_uzak_metastaz_gun", "Senaryo C", "İlk uzak met zamanı", "gün", "Bölge modeli"),
    ("ilk_uzak_bolge", "Senaryo C", "İlk uzak met yeri", "kemik / viseral / beyin_SSS / kemik+viseral / diger", "Bölge modeli"),
    ("notlar", "önerilen", "Serbest not", "metin", "Özel durumlar"),
]
for i, row in enumerate(sozluk, start=2):
    for j, v in enumerate(row, start=1):
        ws2.cell(i, j, v).alignment = wrap
        ws2.cell(i, j).border = thin
ws2.column_dimensions["A"].width = 28
ws2.column_dimensions["B"].width = 14
ws2.column_dimensions["C"].width = 36
ws2.column_dimensions["D"].width = 55
ws2.column_dimensions["E"].width = 40

# ---------------------------------------------------------------------------
# Sheet 4: Ornek
# ---------------------------------------------------------------------------
ws3 = wb.create_sheet("4_Ornek_Satirlar")
for col_i, (_, header, group) in enumerate(columns, start=1):
    cell = ws3.cell(1, col_i, header)
    cell.fill = header_fill
    cell.font = header_font

examples = [
    {
        "hasta_kodu": "EXT-001",
        "tani_tarihi": "2018-03-12",
        "yas_tani": 58,
        "menopoz": "post",
        "tumor_boyutu_mm": 22,
        "pozitif_lenf_nodu_sayisi": 1,
        "grade": 2,
        "histoloji": "IDC",
        "ER": "pozitif",
        "PR": "pozitif",
        "HER2": "negatif",
        "kemoterapi_verildi": "evet",
        "hormonoterapi_verildi": "evet",
        "evre": 2,
        "DRFI_olay": 0,
        "DRFI_sure_gun": 2190,
        "son_durum": "yasiyor",
        "olum_nedeni": "uygulamaz",
        "NPI": 4.2,
        "radyoterapi_verildi": "evet",
        "cerrahi_tipi": "meme_koruyucu",
        "ilk_uzak_metastaz_oldu": "hayir",
        "ilk_uzak_metastaz_gun": "",
        "ilk_uzak_bolge": "",
        "notlar": "6 yıl metastazsız takip",
    },
    {
        "hasta_kodu": "EXT-002",
        "tani_tarihi": "2016-07-01",
        "yas_tani": 47,
        "menopoz": "pre",
        "tumor_boyutu_mm": 35,
        "pozitif_lenf_nodu_sayisi": 4,
        "grade": 3,
        "histoloji": "IDC",
        "ER": "negatif",
        "PR": "negatif",
        "HER2": "pozitif",
        "kemoterapi_verildi": "evet",
        "hormonoterapi_verildi": "hayir",
        "evre": 3,
        "DRFI_olay": 1,
        "DRFI_sure_gun": 820,
        "son_durum": "olum_meme_kanseri",
        "olum_nedeni": "meme_kanseri",
        "NPI": 5.8,
        "radyoterapi_verildi": "evet",
        "cerrahi_tipi": "mastektomi",
        "ilk_uzak_metastaz_oldu": "evet",
        "ilk_uzak_metastaz_gun": 780,
        "ilk_uzak_bolge": "kemik",
        "notlar": "Önce kemik met, sonra exitus",
    },
    {
        "hasta_kodu": "EXT-003",
        "tani_tarihi": "2019-01-20",
        "yas_tani": 72,
        "menopoz": "post",
        "tumor_boyutu_mm": 18,
        "pozitif_lenf_nodu_sayisi": 0,
        "grade": 2,
        "histoloji": "ILC",
        "ER": "pozitif",
        "PR": "negatif",
        "HER2": "negatif",
        "kemoterapi_verildi": "hayir",
        "hormonoterapi_verildi": "evet",
        "evre": 1,
        "DRFI_olay": 0,
        "DRFI_sure_gun": 1100,
        "son_durum": "olum_diger",
        "olum_nedeni": "diger",
        "NPI": 3.1,
        "radyoterapi_verildi": "hayir",
        "cerrahi_tipi": "meme_koruyucu",
        "ilk_uzak_metastaz_oldu": "hayir",
        "ilk_uzak_metastaz_gun": "",
        "ilk_uzak_bolge": "",
        "notlar": "Başka nedenle ölüm = DRFI olay 0, süre sansür",
    },
]
keys = [c[0] for c in columns]
for r, ex in enumerate(examples, start=2):
    for c, k in enumerate(keys, start=1):
        ws3.cell(r, c, ex.get(k, ""))

ws3["A6"] = "Not: Bunlar uydurma örneklerdir. Gerçek veri 2_Veri_Girisi sayfasına girilecektir."
ws3["A6"].font = Font(italic=True, color="666666")

# ---------------------------------------------------------------------------
# Sheet 5: Kontrol listesi
# ---------------------------------------------------------------------------
ws4 = wb.create_sheet("5_Teslim_Kontrol")
ws4["A1"] = "Teslim öncesi kontrol listesi"
ws4["A1"].font = Font(bold=True, size=14, color="1F5A4E")
checks = [
    "Evre 4 hastalar listeden çıkarıldı mı?",
    "DCIS / benign / fillodes çıkarıldı mı?",
    "DRFI_olay ve DRFI_sure_gun tüm satırlarda dolu mu?",
    "Meme kanseri ölümleri DRFI_olay=1 olarak işaretlendi mi?",
    "Başka nedenle ölümler DRFI_olay=0 ve süre sansür olarak mı işlendi?",
    "PR/HER2 bilinmiyorsa boş veya 'bilinmiyor' mu? (uydurma pozitif/negatif yok)",
    "Tarih formatı tutarlı mı (YYYY-AA-GG)?",
    "İsim/TCKN yok; sadece hasta_kodu var mı?",
    "Merkez adı / veri dönemi (yıllar) notlara yazıldı mı?",
    "Trastuzumab / anti-HER2 kullanım dönemi belirtildi mi? (modern kohort için kritik)",
]
ws4["A3"] = "Kontrol"
ws4["B3"] = "Evet/Hayır"
ws4["A3"].fill = header_fill
ws4["B3"].fill = header_fill
ws4["A3"].font = header_font
ws4["B3"].font = header_font
for i, t in enumerate(checks, start=4):
    ws4[f"A{i}"] = t
    ws4[f"B{i}"] = ""
ws4.column_dimensions["A"].width = 90
ws4.column_dimensions["B"].width = 12
dv_eh = DataValidation(type="list", formula1='"Evet,Hayir"', allow_blank=True)
dv_eh.add("B4:B20")
ws4.add_data_validation(dv_eh)

ws4["A16"] = "Merkez adı:"
ws4["A17"] = "Veri yılları (örn. 2015-2022):"
ws4["A18"] = "İletişim (opsiyonel):"
ws4["A19"] = "Zaman sıfır kuralı (tanı / cerrahi):"

OUT.parent.mkdir(parents=True, exist_ok=True)
wb.save(OUT)
print(f"Wrote {OUT}")
