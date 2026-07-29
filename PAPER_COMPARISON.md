# Perbandingan Implementasi dengan Paper Acuan

**Task dari dosen pembimbing:** *"cek dengan paper"*

**Paper acuan:**
Daniele Rege Cambrin, Luca Colomba, Paolo Garza —
*"Magnifier: A Multi-grained Neural Network-based Architecture for Burned Area
Delineation"*, IEEE JSTARS 2025 (DOI 10.1109/JSTARS.2025.3565819).
Kode Tugas Akhir ini merupakan turunan dari repo penulis paper tersebut
(`github.com/DarthReca/magnifier-california`).

**Dataset Indonesia** dalam paper = referensi [64]:
Prabowo et al., *"Deep learning dataset for estimating burned areas: Case study,
Indonesia"*, Data 7(6):78, 2022 — dataset yang dipakai di `data/indonesia/`.

> **Catatan penting soal posisi penelitian ini terhadap paper.**
> Paper Magnifier menguji backbone **MobileNetV3, ResNet, dan MiT (SegFormer)**.
> Paper **tidak** memakai **MobileViT**. Jadi *"cek dengan paper"* di sini **bukan**
> mereplikasi Magnifier persis, melainkan:
> 1. memastikan **protokol** (dataset, loss, optimizer, cross-validation, metrik)
>    konsisten dengan paper, dan
> 2. membandingkan hasil model **DeepLabV3+ + MobileViT** (kontribusi TA ini)
>    terhadap **baseline single-model DeepLabV3+** paper pada dataset Indonesia.

---

## 1. Ringkasan — apa yang SUDAH sesuai paper ✅

| Aspek | Paper | Implementasi kita | Status |
|---|---|---|---|
| Base decoder | DeepLabV3+ (ASPP + atrous conv) | `smp.DeepLabV3Plus`, ASPP rates (6,12,18), output_stride 16 | ✅ Sesuai |
| Loss function | Asymmetric Unified Focal (AUF) | `AsymmetricUnifiedFocalLoss` | ✅ Sesuai |
| Hyperparam AUF | λ=0.5, δ=0.6, γ=0.1 | `weight=0.5, delta=0.6, gamma=0.1` | ✅ **Persis** |
| Inisialisasi | Random init (bukan RGB, tanpa pretrain) | `encoder_weights: null` | ✅ Sesuai |
| Optimizer | AdamW | AdamW | ✅ Sesuai |
| Scheduler | Polynomial LR | `PolynomialLR` | ✅ (lihat catatan §2) |
| Cross-validation | 5-fold: 1 fold test, 1 fold val, 3 fold train | `test_fold=k`, `val=(k+1)%5`, sisanya train | ✅ **Persis** |
| Task | Binary burned (1) / non-burned (0) | `n_classes=2`, mask 0/1 | ✅ Sesuai |
| Ukuran citra | 227 citra, 512×512 | 227 citra, resize 512×512 | ✅ Sesuai |
| Metrik utama | IoU & F1 kelas burned, ± std antar fold | IoU, F1 (+ precision/recall/acc) | ✅ Sesuai (lihat §2 soal std) |

**Kesimpulan:** kerangka metodologi (loss, optimizer, CV, metrik) **konsisten dengan
paper**. Bagus untuk ditunjukkan ke dosen.

---

## 2. Temuan yang perlu DIPERBAIKI / diperhatikan ⚠️

Diurutkan dari yang paling berdampak.

### ✅ TEMUAN 1 (KRITIS) — SUDAH DIPERBAIKI (2026-07-10) — Statistik normalisasi tidak cocok dengan data asli
File: `lightning_modules/indonesia_datamodule.py:28-29`

> **STATUS: SELESAI.** Mean/std dihitung ulang dari seluruh 227 citra. Setelah
> perbaikan, input ternormalisasi jadi **mean ≈ −0.10, std ≈ 0.91** (sebelumnya
> mean ≈ +11, std ≈ 8). Nilai baru:
> `MEANS = [11592.3, 10282.3, 9015.6, 7896.5, 18340.9, 11298.5]`,
> `STDS = [6431.2, 6780.0, 6697.5, 7130.1, 7922.3, 6089.8]`.
> _(Detail masalah aslinya di bawah, untuk dokumentasi thesis.)_


Kode meng-hardcode:
```python
LANDSAT8_MEANS = [1123, 1117, 1050, 2500, 2200, 1600]
LANDSAT8_STDS  = [ 650,  620,  600, 1200, 1100,  900]
```
tetapi **mean per-band data asli** (diukur dari 20 citra pertama) adalah:
```
band0: 9103   band1: 7823   band2: 6775   band3: 5882
band4: 16000  band5: 10411  band6: 5893   band7: 197
```
Skalanya **berbeda ~6–10×**. Akibatnya z-score `(x - mean) / std` menghasilkan
nilai input yang jauh dari 0 (mis. `(9103 - 1123)/650 ≈ +12`), bukan ternormalisasi
di sekitar 0. Ini **merusak proses training** dan konsisten dengan gejala model
"over-predict burned" pada checkpoint pendek (IoU ≈ 0.46).

**Aksi:** hitung ulang mean/std per-band dari data asli (script sederhana),
lalu masukkan ke `LANDSAT8_MEANS/STDS`. Atau normalisasi per-citra.

### ✅ TEMUAN 2 (KRITIS) — SUDAH DIPERBAIKI (2026-07-17) — Jumlah channel: 6 → 8

> **STATUS: SELESAI.** `in_channels`/`n_channels` = **8** di semua config model,
> `LANDSAT8_MEANS/STDS` diperluas ke 8 nilai. Terverifikasi: dataset menghasilkan
> tensor `(8, 512, 512)`, model build 1.45M param, `fast_dev_run` lolos.

**Masalahnya:** `_normalise()` memakai `n_bands = min(image_bands=8, stats_bands=6) = 6`
→ `image[:6]`. Karena statistik hanya punya 6 nilai, dua band terakhir terbuang
**diam-diam**. Komentar config lama menulis `# Landsat-8: B2 B3 B4 B5 B6 B7` —
mengira SWIR2 ikut. Ternyata tidak.

**Urutan band dipastikan secara empiris dari seluruh 227 citra** (bukan dari asumsi):

| Band | Isi | Bukti |
|---|---|---|
| 0–3 | B1 Coastal, B2 Blue, B3 Green, B4 Red | mean menurun 11592 → 10282 → 9016 → 7897 (pola hamburan atmosfer) |
| 4 | **B5 NIR** | mean **18340.9**, melonjak jauh di atas tetangganya — signature vegetasi |
| 5 | B6 SWIR1 | mean 11298.5 |
| 6 | **B7 SWIR2** | mean 6352.3; 39.017 nilai unik, rentang 0–60000 → band spektral kontinu |
| 7 | **B9 Cirrus** | mean 485.8; ~8k nilai unik, 0% piksel nol, korelasi tetangga **0.80** → bukan mask |

**Kunci pembuktian:** lonjakan NIR ada di **indeks 4**. Kalau stack dimulai dari B2
seperti asumsi kode lama, NIR (B5) seharusnya di indeks 3. Karena berada di indeks 4,
stack dimulai dari **B1** — sehingga `image[:6]` = B1–B6 dan **B7 (SWIR2) terbuang**.

**Dampaknya fatal:** indeks baku area terbakar adalah
**NBR = (B5 − B7) / (B5 + B7)**. Model punya NIR tapi tidak punya SWIR2, jadi
**secara fisik mustahil** menghitung indeks paling penting untuk tugasnya.

**Soal band 7 (indeks terakhir):** sempat diduga QA/mask, tetapi tidak lolos uji —
mask QA bernilai diskrit (0–15), sedangkan band ini punya ~8.000 nilai unik, tanpa
piksel nol, dan berstruktur spasial halus (korelasi tetangga 0,80). Korelasinya
terhadap semua band permukaan ≈ **0,0** (0,013–0,056), yang justru khas **B9 Cirrus**:
serapan uap air membuatnya tidak melihat permukaan. Ini konsisten dengan stack
Landsat-8 30 m standar (**B1,B2,B3,B4,B5,B6,B7,B9**; B8 Pan dilewat karena 15 m) =
tepat **8 channel** seperti Tabel I paper.

**Catatan untuk analisis:** `std` B9 dalam satu citra hanya **0,013** (praktis
konstan) → informasi kebakarannya nyaris nol. Kenaikan performa diharapkan datang
dari **SWIR2**, bukan B9. B9 tetap disertakan demi kesetaraan protokol dengan paper.

**Statistik baru** (mean/std per band, seluruh piksel dari 227 citra — metode
diverifikasi identik dengan nilai lama, selisih 0,0 pada 6 band pertama):
```python
LANDSAT8_MEANS = torch.tensor([11592.3, 10282.3, 9015.6, 7896.5, 18340.9, 11298.5, 6352.3, 485.8])
LANDSAT8_STDS  = torch.tensor([ 6431.2,  6780.0,  6697.5, 7130.1,  7922.3,  6089.8, 4739.9, 855.6])
```

⚠️ **Checkpoint lama tidak kompatibel.** Konvolusi pertama berubah 6→8 channel, jadi
`lightning_logs/version_6/checkpoints/*.ckpt` (IoU 0.6436) tidak bisa di-load. Simpan
angkanya sebagai baris ablasi **"6 band / tanpa SWIR2"** untuk dibandingkan dengan
hasil 8 band.

### 🔴 TEMUAN 7 (BARU, KRITIS) — Kebocoran spasial: tile satu scene tersebar antar fold

Tanpa `splits.parquet`, fold dibagi lewat urutan abjad mod 5
(`indonesia_datamodule.py:107-109`). Nama file berpola
`L8_<path/row>_<tanggal>_<no.tile>`, sehingga tile dari **citra & tanggal yang sama**
bernomor urut dan **dijamin** jatuh ke fold berbeda-beda.

Hasil pengukuran atas 227 citra:

| Metrik | Nilai |
|---|---|
| Scene unik (path/row + tanggal) | **81** (dari 227 citra) |
| Scene dengan >1 tile | 36 |
| Citra yang berada dalam scene multi-tile | **182 (80,2%)** |
| **Test fold 0 yang punya "saudara" di training** | **34 dari 46 (73,9%)** |

Contoh: scene `117062_240919` punya **22 tile** → fold `[0,1,2,3,4,0,1,2,3,4,…]`,
tersebar ke train, val, **dan** test sekaligus.

**Artinya:** model dilatih pada tile bersebelahan dari kebakaran yang sama, hari yang
sama, kondisi atmosfer yang sama — lalu diuji pada tile tetangganya. Ini lebih dekat
ke **menghafal** daripada generalisasi, sehingga **IoU 0.6436 kemungkinan besar
terlalu tinggi**.

⚠️ **`scripts/sanity_and_splits.py` TIDAK menyelesaikan ini** — script itu mengacak
**per tile** (`random.shuffle(ids)` lalu `i % 5`), bukan per scene. Kebocoran tetap
terjadi, hanya polanya jadi acak.

⚠️ **Bug tambahan:** script menulis kolom **`id`** (baris 90), sedangkan datamodule
membaca kolom **`files`** (baris 105). Karena dibungkus `except Exception` yang
menangkap semua error, `splits.parquet` akan **gagal dibaca tanpa peringatan** dan
diam-diam jatuh kembali ke directory scan mod-5.

**Aksi:** (a) perbaiki nama kolom + ganti except diam-diam jadi peringatan eksplisit;
(b) buat pembagi fold **per-scene** (semua tile satu scene → satu fold; 81 scene ÷ 5 ≈
16 scene/fold); (c) laporkan **kedua** protokol di thesis:

| Protokol | Guna |
|---|---|
| Random split (sama seperti paper) | perbandingan apple-to-apple dengan Tabel IV |
| Scene-aware split | angka generalisasi yang jujur |

Selisih keduanya = kontribusi metodologis: *"kebocoran spasial menaikkan IoU secara
semu sebesar X poin"*.

**Catatan penting:** perlu dicek apakah paper Magnifier juga memakai split acak per
tile. Kalau ya, angka paper (73,5 / 58,1 / 60,9) mengandung bias yang sama dan
perbandingan random-split tetap sah.

### 🟠 TEMUAN 3 — Learning rate ablation ResNet tidak sesuai paper
Paper §IV-E: LR **ResNet-DeepLabV3+ = 0.01**, Mobile-DeepLabV3+ = 0.0001.
Config kita `deeplabv3plus_resnet50.yaml` memakai `learning_rate: 1.0e-4` (0.0001).

**Aksi:** untuk ablasi ResNet-50 yang adil vs paper, pakai `learning_rate=1.0e-2`.
Untuk MobileViT/MobileNetV3 (kelas "mobile"), `1.0e-4` sudah sesuai pola paper.

### 🟡 TEMUAN 4 — Parameter scheduler sedikit berbeda
File: `neural_net/deeplabv3plus_mobilevit.py:318-319`
- Paper: Polynomial LR **power = 1**, **55** iterasi.
- Kode: **power = 0.9**, `total_iters = max_epochs (default 60)`.

**Aksi (opsional, untuk replikasi ketat):** set `power=1.0` dan `max_epochs=55`.

### 🟠 TEMUAN 5 — Pelaporan harus mean ± std 5-fold — **TOOLING SUDAH DIPERBAIKI, RUN MASIH PENDING**

> **STATUS: `mode=crossval` sudah diperbaiki & diuji (2026-07-17). Run 5-fold
> sebenarnya BELUM dijalankan** — ini yang tersisa untuk angka final thesis.

Paper melaporkan **rata-rata ± std antar fold**, dan std-nya besar (±3,4 s/d ±7,6 —
lihat §3). Hasil 1 fold **tidak cukup** untuk mengklaim unggul atas MobileNetV3.

**Tiga bug ditemukan di `mode=crossval` (semua sudah diperbaiki):**

1. **Trainer dibuat tanpa `callbacks`** → tidak ada `ModelCheckpoint` ber-monitor,
   padahal baris berikutnya memanggil `ckpt_path="best"`. Akibatnya "best" jatuh ke
   **checkpoint epoch terakhir**, bukan epoch dengan `val_loss` terbaik → protokolnya
   **berbeda** dari `mode=train`, sehingga angkanya tidak sebanding.
   → Diperbaiki: `callbacks=make_callbacks()`.
2. **Callback dibagi antar fold.** `ModelCheckpoint`/`EarlyStopping` menyimpan state
   (`best_model_path`, counter patience), jadi state fold 0 bocor ke fold berikutnya.
   → Diperbaiki: factory `make_callbacks()` membuat objek baru tiap fold.
3. **`logger=False`** → tidak ada `metrics.csv` per fold (tidak bisa bikin kurva
   loss/IoU untuk thesis), dan ringkasan hanya di-`print` sehingga hilang bila
   terminal ditutup setelah run belasan jam.
   → Diperbaiki: `CSVLogger` per fold + ringkasan ditulis ke `summary.csv`.

**Terverifikasi** lewat smoke test 5 fold (exit 0). Artefak yang dihasilkan:
```
lightning_logs/crossval_<model_name>/
├── summary.csv        ← metric, fold0..fold4, mean, std  (siap jadi Tabel IV)
├── fold0/metrics.csv  ← kurva per fold
└── ... fold4/metrics.csv
```

**Aksi (tersisa):** jalankan di GPU, lalu laporkan `mean ± std` IoU & F1:
```bash
python main.py mode=crossval model=deeplabv3plus_mobilevit_xxs dataset=indonesia trainer.max_epochs=55
```

### 🟢 TEMUAN 6 — Metrik efisiensi
Paper memakai **GFLOPs** sebagai ukuran biaya komputasi. Kita melaporkan
`inference_time_ms` + jumlah parameter. Keduanya valid, tetapi jika ingin
tabel yang langsung sebanding dengan paper, tambahkan GFLOPs
(mis. lewat `fvcore`/`thop`).

---

## 3. Target angka dari paper (acuan hasil)

**Paper Tabel IVa — DeepLabV3+, kolom Indonesia** (F1 / IoU, dalam %):

| Backbone (single model) | Params | F1 | IoU |
|---|---|---|---|
| MobileNetV3-Small | 0.93M | 73.3 ± 6.1 | 58.1 ± 7.6 |
| MobileNetV3-Large | 2.97M | 75.5 ± 4.9 | 60.9 ± 6.2 |
| ResNet-18 | 11M | 83.7 ± 2.8 | 72.0 ± 4.1 |
| ResNet-101 | 42M | 82.4 ± 3.9 | 70.2 ± 5.5 |
| **Magnifier-ResNet18 (best paper)** | 22M | **84.7 ± 2.2** | **73.5 ± 3.4** |

**Cara membaca untuk TA ini:** model kita **DeepLabV3+ + MobileViT-XXS** (~1.45M
param) sekelas backbone "mobile". Target realistis setelah training penuh:
**IoU ≈ 0.58–0.72, F1 ≈ 0.73–0.84** — idealnya menyamai/melebihi MobileNetV3-Small
pada jumlah parameter yang sebanding, sebagai bukti keunggulan hybrid CNN-Transformer
MobileViT.

> ⚠️ **Lawan pembanding yang benar adalah baris "mobile", BUKAN 73.5/84.7.**
> Angka 73.5 / 84.7 itu **Magnifier-ResNet18 (22M param, arsitektur multi-model)** —
> model terbaik paper, bukan sekelas MobileViT-XXS (1.45M).

**Hasil sementara — run 55 epoch, 6 band, 1 fold (`lightning_logs/version_6`):**

| Model | Params | F1 | IoU |
|---|---|---|---|
| MobileNetV3-Small (paper) | 0.93M | 73.3 ± 6.1 | 58.1 ± 7.6 |
| MobileNetV3-Large (paper) | 2.97M | 75.5 ± 4.9 | 60.9 ± 6.2 |
| **MobileViT-XXS (TA ini)** | **1.45M** | **78.32** | **64.36** |
| ResNet-18 (paper) | 11M | 83.7 ± 2.8 | 72.0 ± 4.1 |

Sudah **melampaui kedua baseline mobile** — unggul **+3.5 IoU** atas MobileNetV3-Large
dengan **setengah** jumlah parameter. Ini persis klaim TA ini, dan masuk rentang target.

**Tetapi angka ini BELUM bisa ditulis sebagai hasil final,** karena tiga hal:
1. **Baru 1 fold**, bukan mean ± std 5-fold (Temuan 5). Std paper mencapai ±7,6 —
   1 fold tidak cukup untuk mengklaim unggul.
2. **Masih 6 band** — SWIR2 terbuang (Temuan 2, sudah diperbaiki tapi belum dilatih ulang).
3. **Mengandung kebocoran spasial** (Temuan 7) — 73,9% test fold punya saudara di
   training, sehingga 64.36 kemungkinan terlalu tinggi.

`test_inference_time = 244,8 ms/citra` pada run itu adalah **angka CPU** — jangan
dipakai untuk klaim efisiensi; ukur ulang di GPU.

---

## 4. Rekomendasi urutan aksi

**Sudah selesai:**
- ✅ Temuan 1 — normalisasi (mean/std dihitung ulang dari data asli, 2026-07-10)
- ✅ Temuan 2 — band 6 → 8, SWIR2 dipulihkan + B9 ditambahkan (2026-07-17)
- ✅ Temuan 4 — run 55 epoch sudah dijalankan (scheduler `power` masih 0.9, lihat bawah)
- ✅ Temuan 5 (tooling) — `mode=crossval` diperbaiki & diuji; run-nya masih pending

**Berikutnya, sesuai prioritas:**

1. **Latih ulang 1 fold dengan 8 band** → bandingkan langsung dengan 64.36 (6 band).
   Ini mengukur dampak SWIR2 dan sekaligus jadi baris ablasi thesis.
   ```bash
   python main.py mode=train model=deeplabv3plus_mobilevit_xxs dataset=indonesia trainer.max_epochs=55
   ```
2. **Jalankan `mode=crossval`** (Temuan 5) → `mean ± std` IoU/F1 untuk Tabel IV.
   Biayanya 5× satu run → **wajib GPU**.
3. **Perbaiki kebocoran spasial** (Temuan 7) → pembagi fold per-scene + bug kolom
   `id`/`files`. Laporkan random-split **dan** scene-aware side-by-side.
4. **Ukur ulang inference time di GPU** (angka CPU 244 ms tidak layak untuk klaim
   efisiensi model "mobile").
5. Perbaiki LR ablasi ResNet ke 0.01 (Temuan 3) sebelum menjalankan ablasi.
6. (Opsional) scheduler `power=0.9` → `1.0` (Temuan 4); tambahkan GFLOPs (Temuan 6).

**Tabel ablasi yang layak masuk thesis:**

| Konfigurasi | IoU | F1 | Catatan |
|---|---|---|---|
| 6 band (tanpa SWIR2), 1 fold | 0.6436 | 0.7832 | `version_6` — checkpoint tidak kompatibel dgn config 8-band |
| 8 band (dengan SWIR2), 1 fold | — | — | mengukur dampak SWIR2 |
| 8 band, 5-fold CV | — ± — | — ± — | **angka final, format Tabel IV** |
| 8 band, scene-aware split | — ± — | — ± — | angka generalisasi jujur |
