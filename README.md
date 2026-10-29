# 🫁 Göğüs Röntgeni ile Zatürre (Pneumonia) Tespiti

## İkili Model Karşılaştırması: CNN vs Vision Transformer (ViT)

> **Ders:** Bilgisayarlı Görme — Dönem Projesi  
> **Amaç:** Göğüs röntgeni görüntülerinden zatürre (pneumonia) hastalığını tespit etmek için iki farklı derin öğrenme modelini eğitmek ve performanslarını karşılaştırmak.

---

## 📁 Proje Klasör Yapısı

Projeyi çalıştırmadan **önce** klasör yapın şu şekilde olmalı:

```
DualModel-Pneumonia-Detection/
├── main.py                              ← Ana program dosyası (tüm kod burada)
├── README.md                            ← Bu dosya
├── data/
│   ├── train/                           ← Eğitim verileri
│   │   ├── NORMAL/                      ← Sağlıklı akciğer röntgenleri
│   │   │   ├── IM-0001.jpeg
│   │   │   └── ...
│   │   └── PNEUMONIA/                   ← Zatürre teşhisi konmuş röntgenler
│   │       ├── person1_bacteria_1.jpeg
│   │       └── ...
│   └── test/                            ← Test verileri
│       ├── NORMAL/
│       └── PNEUMONIA/
│
│   *** Çalıştırdıktan sonra oluşan dosyalar: ***
│
├── egitim_karsilastirma_grafikleri.png   ← CNN vs ViT eğitim grafikleri
├── tahmin_gorsellestirmesi.png           ← 10 rastgele test görüntüsü tahminleri
└── en_iyi_vit.pth                       ← En iyi ViT model ağırlıkları
```

---

## 🚀 Nasıl Çalıştırılır?

### PyCharm ile (En Kolay Yol):
1. PyCharm'da projeyi aç
2. Sol taraftan `main.py` dosyasına çift tıkla
3. Üst menüden **Run → Run 'main.py'** seçeneğine tıkla (veya **Ctrl+Shift+F10** / **⌃⇧R**)
4. Alt taraftaki **Run** penceresinde eğitim ilerlemesini canlı takip edebilirsin

### Terminal ile:
```bash
cd /Users/yusuf/PycharmProjects/DualModel-Pneumonia-Detection
.venv/bin/python main.py
```

### ⏱️ Tahmini Süre:
- CNN eğitimi: ~5-15 dakika (bilgisayar hızına göre)
- ViT eğitimi: ~10-30 dakika (GPU yoksa daha uzun sürebilir)
- **Toplam:** Yaklaşık 20-45 dakika

---

## 📖 main.py Kodunun Detaylı Açıklaması

Kodun tamamı mantıksal bölümlere ve fonksiyonlara ayrılmıştır. Her bölüm ne yaptığını aşağıda bulabilirsin:

---

### 1️⃣ Kütüphane İthalleri (Satır 26-57)

```python
import tensorflow as tf    # CNN modeli için
import torch               # ViT modeli için
import timm                # Hazır ViT modeli yüklemek için
import cv2                 # Görüntü okuma ve işleme
import matplotlib.pyplot   # Grafik çizimi
```

Projede **iki ayrı derin öğrenme kütüphanesi** kullanılıyor:
- **TensorFlow/Keras** → CNN modelini sıfırdan inşa etmek için
- **PyTorch + timm** → Hazır eğitilmiş ViT modelini yükleyip ince ayar yapmak için

Bu iki kütüphanenin farklı veri formatları beklediğini kodun içinde otomatik olarak hallediyoruz.

---

### 2️⃣ Sabit Değerler / Konfigürasyon (Satır 59-85)

```python
GORUNTU_BOYUTU = 224       # Tüm görüntüler 224x224 piksel olacak
BATCH_BOYUTU = 32          # Bir seferde 32 görüntü işlenir
CNN_EPOCH_SAYISI = 15      # CNN 15 tur eğitilir
VIT_EPOCH_SAYISI = 5       # ViT 5 tur yeterli (zaten önceden öğrenmiş)
```

**Epoch nedir?** Modelin tüm eğitim verisini baştan sona bir kez görmesine 1 epoch denir. CNN sıfırdan öğrendiği için 15 epoch lazım; ViT zaten milyonlarca görüntüyle önceden eğitildiği için 5 epoch yeterli.

**Batch nedir?** Tüm veriyi tek seferde belleğe sığdıramayız. Bu yüzden 32'şerli gruplar halinde modele veriyoruz.

---

### 3️⃣ Veri Okuma Fonksiyonları (Satır 88-143)

#### `goruntu_yollarini_ve_etiketleri_oku(dizin)`
- `data/train/NORMAL/` klasöründeki tüm dosyaları bulur → etiket = **0** (sağlıklı)
- `data/train/PNEUMONIA/` klasöründeki tüm dosyaları bulur → etiket = **1** (hasta)
- **CSV dosyası kullanılmaz!** Etiket doğrudan klasör adından gelir.

#### `goruntuleri_yukle_cnn(yollar, etiketler)`
- Görüntüleri **gri seviye** (siyah-beyaz, tek kanal) olarak okur
- Her görüntüyü **224×224** piksel boyutuna getirir
- Piksel değerlerini **0-255** aralığından **0-1** aralığına normalize eder (255'e böler)
- CNN tek kanal bekler çünkü röntgenler zaten siyah-beyazdır, ekstra renk bilgisi gereksiz

---

### 4️⃣ CNN Modeli — Sıfırdan İnşa (Satır 146-228)

#### `cnn_modeli_olustur()`
Katman katman şu mimariyi kurar:

```
Giriş (224x224x1 gri görüntü)
    ↓
[Conv2D 32 filtre] → [BatchNorm] → [MaxPooling 2x2]    ← 1. Blok
    ↓
[Conv2D 64 filtre] → [BatchNorm] → [MaxPooling 2x2]    ← 2. Blok
    ↓
[Conv2D 128 filtre] → [BatchNorm] → [MaxPooling 2x2]   ← 3. Blok
    ↓
[Flatten]              ← 3D matrisi tek satıra düzleştir
    ↓
[Dense 256 nöron]      ← Tam bağlı katman
    ↓
[Dropout %50]          ← Ezberlemeyi önle (rastgele nöron kapat)
    ↓
[Dense 1, sigmoid]     ← Çıkış: 0.0 (Normal) ile 1.0 (Pneumonia) arası
```

**Katmanlar ne işe yarar?**
| Katman | Görevi |
|--------|--------|
| **Conv2D** | Görüntüdeki kenarları, dokuları, şekilleri öğrenir (filtre/özellik çıkarma) |
| **MaxPooling2D** | Görüntüyü küçültür, önemli bilgileri korur, hesaplamayı azaltır |
| **BatchNormalization** | Eğitimi hızlandırır ve kararlı hale getirir |
| **Flatten** | 2D özellik haritasını tek boyutlu vektöre çevirir |
| **Dense** | Tüm nöronları birbirine bağlar, karar verme katmanı |
| **Dropout** | Rastgele nöronları kapatarak modelin ezberleme yapmasını önler |

#### `cnn_modelini_egit()`
- **EarlyStopping:** Doğrulama kaybı 5 epoch boyunca iyileşmezse eğitimi erken durdurur (gereksiz eğitimi önler)
- **ReduceLROnPlateau:** Model tıkanırsa öğrenme oranını yarıya düşürür
- `shuffle=True` ile her epoch'ta veri sırası karıştırılır

#### `cnn_tahmin_yap()`
- Sigmoid çıktısı 0.5'ten büyükse → **PNEUMONIA**, küçükse → **NORMAL**
- Güven oranı: Modelin kararından ne kadar emin olduğu (ör: %92)

---

### 5️⃣ ViT Modeli — Hazır Model + İnce Ayar (Satır 231-396)

#### Vision Transformer (ViT) Nedir?
CNN'den farklı olarak ViT, görüntüyü küçük parçalara (patch) böler ve bunları bir cümledeki kelimeler gibi işler. Google tarafından geliştirilmiştir ve büyük veri setlerinde CNN'den daha iyi performans gösterir.

#### `RontgenVeriSeti` (PyTorch Dataset sınıfı)
- ViT modeli **RGB (3 kanal, renkli)** görüntü bekler
- Gri röntgenler otomatik olarak 3 kanala dönüştürülür (`.convert("RGB")`)
- Bu, CNN ile ViT arasındaki en büyük fark: CNN tek kanal alır, ViT üç kanal alır

#### `vit_donusumleri_olustur()`
Eğitim sırasında **veri artırma (data augmentation)** uygulanır:
- Rastgele yatay çevirme
- Rastgele döndürme (±10°)
- Parlaklık/kontrast değişimi
- ImageNet normalizasyonu (ViT bu değerlerle eğitildiği için aynısını uygulamak şart)

#### `vit_modeli_olustur()`
- `vit_tiny_patch16_224` modeli indirilir (önceden ImageNet ile eğitilmiş)
- Son sınıflandırma katmanı 1000 sınıftan → **2 sınıfa** (Normal/Pneumonia) değiştirilir
- Bu işleme **Transfer Öğrenme (Transfer Learning)** denir

#### `vit_modelini_egit()`
- Her epoch sonunda doğrulama doğruluğu kontrol edilir
- En iyi model `en_iyi_vit.pth` dosyasına kaydedilir
- Cosine Annealing ile öğrenme oranı kademeli olarak azalır

#### `vit_tahmin_yap()`
- Softmax fonksiyonu ile olasılık dağılımı hesaplanır
- En yüksek olasılıklı sınıf tahmin olarak seçilir

---

### 6️⃣ Grafik Çizimi (Satır 399-455)

#### `egitim_grafiklerini_ciz()`

4 adet grafik yan yana çizilir ve `egitim_karsilastirma_grafikleri.png` olarak kaydedilir:

```
┌─────────────────────┬─────────────────────┐
│  CNN — Kayıp (Loss) │  ViT — Kayıp (Loss) │
│  Mavi: Eğitim       │  Mavi: Eğitim       │
│  Turuncu: Doğrulama │  Turuncu: Doğrulama │
├─────────────────────┼─────────────────────┤
│  CNN — Doğruluk     │  ViT — Doğruluk     │
│  Yeşil: Eğitim      │  Yeşil: Eğitim      │
│  Turuncu: Doğrulama │  Turuncu: Doğrulama │
└─────────────────────┴─────────────────────┘
```

**Loss (Kayıp):** Modelin ne kadar yanlış yaptığını gösterir. Eğitim boyunca **düşmesi** beklenir.  
**Accuracy (Doğruluk):** Modelin ne kadar doğru tahmin ettiğini gösterir. Eğitim boyunca **artması** beklenir.

---

### 7️⃣ Tahmin Görselleştirmesi (Satır 458-523)

#### `tahmin_gorsellestirme()`

Test setinden **rastgele 10 röntgen görüntüsü** seçilir. Her birinin üzerinde şu bilgiler yazar:

```
Gerçek: PNEUMONIA
CNN: PNEUMONIA, Güven: %94.2
ViT: PNEUMONIA, Güven: %98.7
```

**Renk kodları:**
- 🟢 **Yeşil:** Her iki model de doğru tahmin etmiş
- 🔴 **Kırmızı:** Her iki model de yanlış tahmin etmiş
- 🟠 **Turuncu:** Biri doğru, biri yanlış

Sonuç `tahmin_gorsellestirmesi.png` olarak kaydedilir.

---

### 8️⃣ Ana Program Akışı — `main()` (Satır 526-618)

Tüm adımlar sırayla çalışır:

```
ADIM 1: Veri okuma → Klasörlerden görüntüler okunur, karıştırılır, %80/20 bölünür
    ↓
ADIM 2: CNN eğitimi → Gri seviye görüntülerle 15 epoch eğitim
    ↓
ADIM 3: ViT eğitimi → RGB görüntülerle 5 epoch ince ayar
    ↓
ADIM 4: Grafik çizimi → 4'lü karşılaştırma grafiği PNG olarak kaydedilir
    ↓
ADIM 5: Tahmin görseli → 10 test görüntüsü üzerinde sonuçlar gösterilir
    ↓
SONUÇ ÖZETİ: Her iki modelin test doğruluğu ekrana yazdırılır
```

---

## 📊 Çalıştırdıktan Sonra Nereye Bakacaksın?

### 1. PyCharm Run Penceresi (Alt taraf)
Eğitim sırasında şunları göreceksin:
```
ADIM 2: CNN modeli eğitimi (Keras)
Epoch 1/15
130/130 ━━━━━━━━━━━━━━━━ 45s - accuracy: 0.7823 - loss: 0.4521 - val_accuracy: 0.8590 - val_loss: 0.3312
...
★ CNN Test Doğruluğu: 0.8942 (89.42%)

ADIM 3: ViT modeli ince ayar (Fine-Tuning)
Epoch [1/5]
  Eğitim   → Kayıp: 0.3210, Doğruluk: 0.8734
  Doğrulama → Kayıp: 0.1980, Doğruluk: 0.9320
...
★ ViT Test Doğruluğu: 0.9423 (94.23%)
```

### 2. Oluşan Dosyalar (Proje klasöründe)

| Dosya | Ne İçerir | Hocaya Gösterilecek mi? |
|-------|-----------|------------------------|
| `egitim_karsilastirma_grafikleri.png` | CNN vs ViT loss ve accuracy grafikleri | ✅ **Evet — en önemli çıktı** |
| `tahmin_gorsellestirmesi.png` | 10 röntgen üzerinde tahmin sonuçları | ✅ **Evet** |
| `en_iyi_vit.pth` | ViT'in kayıtlı ağırlıkları | Opsiyonel |

### 3. Son Çıktı (Programın sonunda)
```
══════════════════════════════════════════════════
  SONUÇ ÖZETİ
══════════════════════════════════════════════════
  CNN Test Doğruluğu : 89.42%
  ViT Test Doğruluğu : 94.23%
  Fark (ViT - CNN)   : +4.81%
```

---

## 🧠 İki Model Arasındaki Temel Farklar

| Özellik | CNN (Keras) | ViT (PyTorch + timm) |
|---------|-------------|----------------------|
| Yaklaşım | Sıfırdan öğrenme | Transfer öğrenme (ince ayar) |
| Giriş | Gri seviye (1 kanal) | RGB (3 kanal) |
| Mimari | Evrişim + Havuzlama | Dikkat mekanizması (Attention) |
| Parametre | ~2-3 milyon | ~5-6 milyon (ama önceden eğitimli) |
| Beklenen sonuç | İyi (~85-90%) | Daha iyi (~90-95%) |
| Eğitim süresi | Daha kısa | Daha uzun |

---

## ⚙️ Gerekli Kütüphaneler

```
tensorflow >= 2.10
torch >= 2.0
torchvision >= 0.15
timm >= 0.9
opencv-python-headless >= 4.5
matplotlib >= 3.5
scikit-learn >= 1.0
numpy >= 1.21
Pillow >= 9.0
tqdm >= 4.60
```

---

## 📝 Notlar

- Kod ilk çalıştırmada ViT modelini internetten indirdiği için **internet bağlantısı gereklidir**.
- GPU yoksa eğitim CPU üzerinde yapılır; biraz daha uzun sürer ama sorunsuz çalışır.
- Rastgelelik tohumu (seed=42) sabitlendiği için her çalıştırmada **aynı sonuçları** alırsın.
- Grafik dosyaları her çalıştırmada üzerine yazılır (overwrite).

