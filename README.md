# 🫁 Göğüs Röntgeni ile Zatürre (Pneumonia) Tespiti

## İkili Model Karşılaştırması: CNN vs Vision Transformer (ViT)


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

### Terminal ile:
```bash
cd /Users/yusuf/PycharmProjects/DualModel-Pneumonia-Detection
.venv/bin/python main.py
```

### ⏱️ Tahmini Süre:
- CNN eğitimi: ~5-15 dakika (bilgisayar hızına göre)
- ViT eğitimi: ~10-30 dakika (GPU yoksa daha uzun sürebilir)
- **Toplam:** Yaklaşık 10-45 dakika



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

