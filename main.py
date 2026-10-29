"""
=============================================================================
  Göğüs Röntgeni ile Zatürre (Pneumonia) Tespiti
  İkili Model Karşılaştırması: CNN (Keras) vs Vision Transformer (ViT)
  Üniversite - Bilgisayarlı Görme Dersi Dönem Projesi
=============================================================================

Proje Özeti:
  - Göğüs röntgeni görüntüleri klasör yapısından okunur (CSV yok).
  - Model 1: Sıfırdan inşa edilen CNN (Conv2D, MaxPooling2D, Flatten, Dense).
  - Model 2: Önceden eğitilmiş ViT modeli üzerinde ince ayar (fine-tuning).
  - Her iki modelin eğitim/doğrulama grafikleri yan yana karşılaştırılır.
  - Test setinden rastgele 10 görüntü üzerinde tahmin görselleştirilir.

Veri Seti Klasör Yapısı:
  data/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/
"""

# ========================== KÜTÜPHANE İTHALLERİ ==========================
import os
import random
import warnings

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # GUI olmadan grafik oluşturmak için
import matplotlib.pyplot as plt

# --- Keras / TensorFlow ---
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# --- PyTorch / Transformers / timm ---
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm

from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Gereksiz uyarıları kapat
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ========================== SABİT DEĞERLER (CONFIG) ==========================
# Proje kök dizini (main.py'nin bulunduğu yer)
PROJE_DIZINI = os.path.dirname(os.path.abspath(__file__))
VERI_DIZINI = os.path.join(PROJE_DIZINI, "data")
EGITIM_DIZINI = os.path.join(VERI_DIZINI, "train")
TEST_DIZINI = os.path.join(VERI_DIZINI, "test")

# Görüntü ve eğitim parametreleri
GORUNTU_BOYUTU = 224          # Hem CNN hem ViT için 224x224
BATCH_BOYUTU = 32
CNN_EPOCH_SAYISI = 15
VIT_EPOCH_SAYISI = 5          # ViT ince ayar için daha az epoch yeterli
OGRENME_ORANI_CNN = 1e-3
OGRENME_ORANI_VIT = 2e-5      # İnce ayar için düşük öğrenme oranı
DOGRULAMA_ORANI = 0.2         # Eğitim verisinden %20 doğrulama için ayrılır

# Sınıf isimleri — klasör adlarıyla birebir eşleşmeli
SINIF_ISIMLERI = ["NORMAL", "PNEUMONIA"]

# Cihaz seçimi (GPU varsa kullan, yoksa CPU)
CIHAZ = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")

# Tekrar üretilebilirlik için rastgelelik tohumları
RASTGELELIK_TOHUMU = 42
random.seed(RASTGELELIK_TOHUMU)
np.random.seed(RASTGELELIK_TOHUMU)
tf.random.set_seed(RASTGELELIK_TOHUMU)
torch.manual_seed(RASTGELELIK_TOHUMU)


# ========================== VERİ OKUMA FONKSİYONLARI ==========================

def goruntu_yollarini_ve_etiketleri_oku(dizin):
    """
    Belirtilen dizindeki alt klasörlerden görüntü dosya yollarını ve etiketlerini okur.
    Etiketler, klasör adından (NORMAL=0, PNEUMONIA=1) otomatik olarak çıkarılır.
    CSV dosyası kullanılmaz.

    Args:
        dizin (str): Üst klasör yolu (örn: data/train)

    Returns:
        yollar (list): Görüntü dosya yollarının listesi
        etiketler (list): Her görüntüye karşılık gelen sayısal etiketler (0 veya 1)
    """
    yollar = []
    etiketler = []
    for sinif_indeksi, sinif_adi in enumerate(SINIF_ISIMLERI):
        sinif_klasoru = os.path.join(dizin, sinif_adi)
        if not os.path.isdir(sinif_klasoru):
            print(f"[UYARI] Klasör bulunamadı: {sinif_klasoru}")
            continue
        for dosya_adi in os.listdir(sinif_klasoru):
            dosya_yolu = os.path.join(sinif_klasoru, dosya_adi)
            # Sadece geçerli görüntü dosyalarını al
            if dosya_adi.lower().endswith((".jpeg", ".jpg", ".png", ".bmp")):
                yollar.append(dosya_yolu)
                etiketler.append(sinif_indeksi)
    print(f"  → {dizin} dizininden {len(yollar)} görüntü okundu.")
    return yollar, etiketler


def goruntuleri_yukle_cnn(yollar, etiketler, boyut=GORUNTU_BOYUTU):
    """
    CNN modeli için görüntüleri gri seviye (tek kanal) olarak yükler,
    boyutlandırır ve 0-1 arasına normalize eder.

    Args:
        yollar (list): Görüntü dosya yolları
        etiketler (list): Sayısal etiketler
        boyut (int): Hedef görüntü boyutu (kare)

    Returns:
        goruntu_dizisi (np.ndarray): (N, boyut, boyut, 1) şeklinde normalize dizi
        etiket_dizisi (np.ndarray): (N,) şeklinde etiket dizisi
    """
    goruntuler = []
    gecerli_etiketler = []
    for yol, etiket in zip(yollar, etiketler):
        # Gri seviye olarak oku (CNN için tek kanal yeterli)
        goruntu = cv2.imread(yol, cv2.IMREAD_GRAYSCALE)
        if goruntu is None:
            continue  # Bozuk dosyaları atla
        # 224x224 boyutuna getir
        goruntu = cv2.resize(goruntu, (boyut, boyut))
        goruntuler.append(goruntu)
        gecerli_etiketler.append(etiket)

    # NumPy dizisine çevir, kanal boyutunu ekle, 0-1 arasına normalize et
    goruntu_dizisi = np.array(goruntuler, dtype=np.float32) / 255.0
    goruntu_dizisi = np.expand_dims(goruntu_dizisi, axis=-1)  # (N, 224, 224, 1)
    etiket_dizisi = np.array(gecerli_etiketler, dtype=np.int32)
    return goruntu_dizisi, etiket_dizisi


# ========================== CNN MODELİ (KERAS) ==========================

def cnn_modeli_olustur(giris_boyutu=(GORUNTU_BOYUTU, GORUNTU_BOYUTU, 1)):
    """
    Sıfırdan basit bir CNN mimarisi kurar.
    Hoca kuralı gereği Conv2D, MaxPooling2D, Flatten ve Dense katmanları kullanılır.

    Mimari:
      Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Conv2D(128) → MaxPool →
      Flatten → Dense(256) → Dropout → Dense(1, sigmoid)

    Args:
        giris_boyutu (tuple): Giriş tensörünün şekli (yükseklik, genişlik, kanal)

    Returns:
        model: Derlenmiş Keras Sequential modeli
    """
    model = Sequential([
        Input(shape=giris_boyutu),

        # 1. Evrişim Bloğu
        Conv2D(32, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # 2. Evrişim Bloğu
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # 3. Evrişim Bloğu
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # Düzleştirme ve Tam Bağlı Katmanlar
        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid")  # İkili sınıflandırma
    ])

    model.compile(
        optimizer=Adam(learning_rate=OGRENME_ORANI_CNN),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def cnn_modelini_egit(model, X_egitim, y_egitim, X_dogrulama, y_dogrulama):
    """
    CNN modelini eğitir. Erken durdurma ve öğrenme oranı azaltma geri çağırmaları
    ile aşırı öğrenme (overfitting) kontrol altına alınır.

    Args:
        model: Derlenmiş Keras modeli
        X_egitim, y_egitim: Eğitim verisi ve etiketleri
        X_dogrulama, y_dogrulama: Doğrulama verisi ve etiketleri

    Returns:
        gecmis: Eğitim geçmişi (loss, accuracy, val_loss, val_accuracy)
    """
    geri_cagirmalar = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            verbose=1
        )
    ]

    gecmis = model.fit(
        X_egitim, y_egitim,
        validation_data=(X_dogrulama, y_dogrulama),
        epochs=CNN_EPOCH_SAYISI,
        batch_size=BATCH_BOYUTU,
        callbacks=geri_cagirmalar,
        shuffle=True,  # Hoca kuralı: veri mutlaka karıştırılmalı
        verbose=1
    )
    return gecmis


def cnn_tahmin_yap(model, X_test):
    """
    CNN modeli ile test verisi üzerinde tahmin yapar.

    Args:
        model: Eğitilmiş Keras modeli
        X_test (np.ndarray): Test görüntüleri

    Returns:
        tahmin_siniflar (np.ndarray): Tahmin edilen sınıflar (0 veya 1)
        tahmin_olasiliklar (np.ndarray): Tahmin güven oranları (0.0 - 1.0)
    """
    ham_tahminler = model.predict(X_test, verbose=0).flatten()
    tahmin_siniflar = (ham_tahminler >= 0.5).astype(int)
    # Güven: Tahmin edilen sınıfın olasılığı
    tahmin_olasiliklar = np.where(
        tahmin_siniflar == 1, ham_tahminler, 1 - ham_tahminler
    )
    return tahmin_siniflar, tahmin_olasiliklar


# ========================== ViT MODELİ (PyTorch + timm) ==========================

class RontgenVeriSeti(Dataset):
    """
    PyTorch Dataset sınıfı.
    ViT modeli RGB (3 kanal) giriş beklediği için görüntüler renkli olarak yüklenir.
    """

    def __init__(self, dosya_yollari, etiketler, donusum=None):
        """
        Args:
            dosya_yollari (list): Görüntü dosya yolları
            etiketler (list): Sayısal etiketler
            donusum: torchvision transforms nesnesi
        """
        self.dosya_yollari = dosya_yollari
        self.etiketler = etiketler
        self.donusum = donusum

    def __len__(self):
        return len(self.dosya_yollari)

    def __getitem__(self, indeks):
        # Görüntüyü RGB olarak aç (ViT 3 kanal bekler)
        goruntu = Image.open(self.dosya_yollari[indeks]).convert("RGB")
        if self.donusum:
            goruntu = self.donusum(goruntu)
        etiket = self.etiketler[indeks]
        return goruntu, etiket


def vit_donusumleri_olustur():
    """
    ViT modeli için görüntü dönüşüm pipeline'ları oluşturur.
    - Eğitim: Rastgele çevirme ve renk dönüşümü ile veri artırma (augmentation)
    - Doğrulama/Test: Sadece yeniden boyutlandırma ve normalize etme

    Returns:
        egitim_donusumu, deger_donusumu: torchvision.transforms.Compose nesneleri
    """
    # ImageNet normalizasyon değerleri (ViT bu değerlerle önceden eğitilmiştir)
    ortalama = [0.485, 0.456, 0.406]
    std_sapma = [0.229, 0.224, 0.225]

    egitim_donusumu = transforms.Compose([
        transforms.Resize((GORUNTU_BOYUTU, GORUNTU_BOYUTU)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),             # 0-1 arasına normalize eder
        transforms.Normalize(ortalama, std_sapma)
    ])

    deger_donusumu = transforms.Compose([
        transforms.Resize((GORUNTU_BOYUTU, GORUNTU_BOYUTU)),
        transforms.ToTensor(),
        transforms.Normalize(ortalama, std_sapma)
    ])

    return egitim_donusumu, deger_donusumu


def vit_modeli_olustur(sinif_sayisi=2):
    """
    timm kütüphanesi ile önceden eğitilmiş (pretrained) ViT modelini yükler.
    Son sınıflandırma katmanını bizim görevimize göre değiştirir.

    Args:
        sinif_sayisi (int): Çıktı sınıf sayısı

    Returns:
        model: PyTorch ViT modeli
    """
    # Küçük ViT modeli — hızlı eğitim ve yeterli doğruluk sağlar
    model = timm.create_model(
        "vit_tiny_patch16_224",  # Hafif ViT varyantı
        pretrained=True,
        num_classes=sinif_sayisi
    )
    model = model.to(CIHAZ)
    print(f"  → ViT modeli '{CIHAZ}' cihazına yüklendi.")
    return model


def vit_bir_epoch_egit(model, veri_yukleyici, kayip_fonk, optimizer):
    """
    ViT modelini tek bir epoch boyunca eğitir.

    Args:
        model: PyTorch modeli
        veri_yukleyici: DataLoader nesnesi
        kayip_fonk: Kayıp fonksiyonu
        optimizer: Optimizasyon algoritması

    Returns:
        ort_kayip (float): Epoch ortalama kayıp değeri
        ort_dogruluk (float): Epoch ortalama doğruluk oranı
    """
    model.train()
    toplam_kayip = 0.0
    toplam_dogru = 0
    toplam_ornek = 0

    for goruntuler, etiketler in tqdm(veri_yukleyici, desc="  Eğitim", leave=False):
        goruntuler = goruntuler.to(CIHAZ)
        etiketler = etiketler.to(CIHAZ)

        optimizer.zero_grad()
        ciktilar = model(goruntuler)
        kayip = kayip_fonk(ciktilar, etiketler)
        kayip.backward()
        optimizer.step()

        toplam_kayip += kayip.item() * goruntuler.size(0)
        tahminler = ciktilar.argmax(dim=1)
        toplam_dogru += (tahminler == etiketler).sum().item()
        toplam_ornek += goruntuler.size(0)

    ort_kayip = toplam_kayip / toplam_ornek
    ort_dogruluk = toplam_dogru / toplam_ornek
    return ort_kayip, ort_dogruluk


def vit_degerlendir(model, veri_yukleyici, kayip_fonk):
    """
    ViT modelini doğrulama veya test veri seti üzerinde değerlendirir.

    Args:
        model: PyTorch modeli
        veri_yukleyici: DataLoader nesnesi
        kayip_fonk: Kayıp fonksiyonu

    Returns:
        ort_kayip (float): Ortalama kayıp
        ort_dogruluk (float): Ortalama doğruluk
    """
    model.eval()
    toplam_kayip = 0.0
    toplam_dogru = 0
    toplam_ornek = 0

    with torch.no_grad():
        for goruntuler, etiketler in tqdm(veri_yukleyici, desc="  Doğrulama", leave=False):
            goruntuler = goruntuler.to(CIHAZ)
            etiketler = etiketler.to(CIHAZ)

            ciktilar = model(goruntuler)
            kayip = kayip_fonk(ciktilar, etiketler)

            toplam_kayip += kayip.item() * goruntuler.size(0)
            tahminler = ciktilar.argmax(dim=1)
            toplam_dogru += (tahminler == etiketler).sum().item()
            toplam_ornek += goruntuler.size(0)

    ort_kayip = toplam_kayip / toplam_ornek
    ort_dogruluk = toplam_dogru / toplam_ornek
    return ort_kayip, ort_dogruluk


def vit_modelini_egit(model, egitim_yukleyici, dogrulama_yukleyici):
    """
    ViT modelinin tam eğitim döngüsünü yürütür.

    Args:
        model: PyTorch ViT modeli
        egitim_yukleyici: Eğitim DataLoader'ı
        dogrulama_yukleyici: Doğrulama DataLoader'ı

    Returns:
        gecmis (dict): Epoch bazında kayıp ve doğruluk değerlerini içeren sözlük
    """
    kayip_fonk = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=OGRENME_ORANI_VIT)

    # Öğrenme oranı programlayıcı — kademeli azalma
    programlayici = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=VIT_EPOCH_SAYISI
    )

    gecmis = {
        "egitim_kayip": [], "egitim_dogruluk": [],
        "dogrulama_kayip": [], "dogrulama_dogruluk": []
    }

    en_iyi_dogruluk = 0.0

    for epoch in range(VIT_EPOCH_SAYISI):
        print(f"\n  Epoch [{epoch + 1}/{VIT_EPOCH_SAYISI}]")

        # Eğitim adımı
        eg_kayip, eg_dogruluk = vit_bir_epoch_egit(
            model, egitim_yukleyici, kayip_fonk, optimizer
        )

        # Doğrulama adımı
        dg_kayip, dg_dogruluk = vit_degerlendir(
            model, dogrulama_yukleyici, kayip_fonk
        )

        programlayici.step()

        # Geçmişe kaydet
        gecmis["egitim_kayip"].append(eg_kayip)
        gecmis["egitim_dogruluk"].append(eg_dogruluk)
        gecmis["dogrulama_kayip"].append(dg_kayip)
        gecmis["dogrulama_dogruluk"].append(dg_dogruluk)

        print(f"    Eğitim   → Kayıp: {eg_kayip:.4f}, Doğruluk: {eg_dogruluk:.4f}")
        print(f"    Doğrulama → Kayıp: {dg_kayip:.4f}, Doğruluk: {dg_dogruluk:.4f}")

        # En iyi modeli takip et
        if dg_dogruluk > en_iyi_dogruluk:
            en_iyi_dogruluk = dg_dogruluk
            torch.save(model.state_dict(), os.path.join(PROJE_DIZINI, "en_iyi_vit.pth"))
            print(f"    ✓ En iyi model kaydedildi (Doğruluk: {en_iyi_dogruluk:.4f})")

    # En iyi ağırlıkları geri yükle
    model.load_state_dict(
        torch.load(os.path.join(PROJE_DIZINI, "en_iyi_vit.pth"), weights_only=True)
    )
    return gecmis


def vit_tahmin_yap(model, veri_yukleyici):
    """
    ViT modeli ile test verisi üzerinde tahmin yapar.

    Args:
        model: Eğitilmiş PyTorch modeli
        veri_yukleyici: Test DataLoader'ı

    Returns:
        tum_tahminler (np.ndarray): Tahmin edilen sınıflar
        tum_olasiliklar (np.ndarray): Tahmin güven oranları
        tum_etiketler (np.ndarray): Gerçek etiketler
    """
    model.eval()
    tum_tahminler = []
    tum_olasiliklar = []
    tum_etiketler = []

    with torch.no_grad():
        for goruntuler, etiketler in veri_yukleyici:
            goruntuler = goruntuler.to(CIHAZ)
            ciktilar = model(goruntuler)
            olasiliklar = torch.softmax(ciktilar, dim=1)
            guvenler, tahminler = olasiliklar.max(dim=1)

            tum_tahminler.extend(tahminler.cpu().numpy())
            tum_olasiliklar.extend(guvenler.cpu().numpy())
            tum_etiketler.extend(etiketler.numpy())

    return (
        np.array(tum_tahminler),
        np.array(tum_olasiliklar),
        np.array(tum_etiketler)
    )


# ========================== GÖRSELLEŞTİRME FONKSİYONLARI ==========================

def egitim_grafiklerini_ciz(cnn_gecmis, vit_gecmis, kayit_yolu):
    """
    Her iki modelin eğitim/doğrulama kayıp ve doğruluk grafiklerini
    yan yana çizdirip PNG olarak kaydeder.

    Grafik Düzeni (2x2):
      [CNN Kayıp]        [ViT Kayıp]
      [CNN Doğruluk]     [ViT Doğruluk]

    Args:
        cnn_gecmis: Keras model.fit() dönüş nesnesi
        vit_gecmis (dict): ViT eğitim geçmişi sözlüğü
        kayit_yolu (str): PNG dosyasının kaydedileceği yol
    """
    fig, eksenler = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("CNN vs ViT — Eğitim ve Doğrulama Grafikleri", fontsize=16, fontweight="bold")

    # ---- CNN Kayıp Grafiği ----
    ek = eksenler[0, 0]
    ek.plot(cnn_gecmis.history["loss"], label="Eğitim Kaybı", color="steelblue", linewidth=2)
    ek.plot(cnn_gecmis.history["val_loss"], label="Doğrulama Kaybı", color="coral", linewidth=2)
    ek.set_title("CNN — Kayıp (Loss)", fontsize=13)
    ek.set_xlabel("Epoch")
    ek.set_ylabel("Kayıp")
    ek.legend()
    ek.grid(True, alpha=0.3)

    # ---- ViT Kayıp Grafiği ----
    ek = eksenler[0, 1]
    ek.plot(vit_gecmis["egitim_kayip"], label="Eğitim Kaybı", color="steelblue", linewidth=2)
    ek.plot(vit_gecmis["dogrulama_kayip"], label="Doğrulama Kaybı", color="coral", linewidth=2)
    ek.set_title("ViT — Kayıp (Loss)", fontsize=13)
    ek.set_xlabel("Epoch")
    ek.set_ylabel("Kayıp")
    ek.legend()
    ek.grid(True, alpha=0.3)

    # ---- CNN Doğruluk Grafiği ----
    ek = eksenler[1, 0]
    ek.plot(cnn_gecmis.history["accuracy"], label="Eğitim Doğruluğu", color="seagreen", linewidth=2)
    ek.plot(cnn_gecmis.history["val_accuracy"], label="Doğrulama Doğruluğu", color="darkorange", linewidth=2)
    ek.set_title("CNN — Doğruluk (Accuracy)", fontsize=13)
    ek.set_xlabel("Epoch")
    ek.set_ylabel("Doğruluk")
    ek.legend()
    ek.grid(True, alpha=0.3)

    # ---- ViT Doğruluk Grafiği ----
    ek = eksenler[1, 1]
    ek.plot(vit_gecmis["egitim_dogruluk"], label="Eğitim Doğruluğu", color="seagreen", linewidth=2)
    ek.plot(vit_gecmis["dogrulama_dogruluk"], label="Doğrulama Doğruluğu", color="darkorange", linewidth=2)
    ek.set_title("ViT — Doğruluk (Accuracy)", fontsize=13)
    ek.set_xlabel("Epoch")
    ek.set_ylabel("Doğruluk")
    ek.legend()
    ek.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(kayit_yolu, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n✓ Karşılaştırma grafikleri kaydedildi: {kayit_yolu}")


def tahmin_gorsellestirme(
    test_yollari, gercek_etiketler,
    cnn_tahminler, cnn_guvenler,
    vit_tahminler, vit_guvenler,
    kayit_yolu
):
    """
    Test setinden rastgele 10 röntgen görüntüsü seçer ve her birinin üzerinde
    gerçek sınıf, tahmin edilen sınıf ve güven oranını gösterir.
    İki satır halinde çizilir: üst satır CNN, alt satır ViT tahminleri.

    Args:
        test_yollari (list): Test görüntülerinin dosya yolları
        gercek_etiketler: Gerçek sınıf etiketleri
        cnn_tahminler, cnn_guvenler: CNN tahmin sonuçları
        vit_tahminler, vit_guvenler: ViT tahmin sonuçları
        kayit_yolu (str): PNG dosyasının kaydedileceği yol
    """
    # Rastgele 10 indeks seç
    toplam = len(test_yollari)
    secilen_indeksler = random.sample(range(toplam), min(10, toplam))

    fig, eksenler = plt.subplots(2, 5, figsize=(22, 9))
    fig.suptitle(
        "Rastgele 10 Test Görüntüsü — Tahmin Sonuçları",
        fontsize=16, fontweight="bold"
    )

    for i, indeks in enumerate(secilen_indeksler):
        # Görüntüyü renkli oku (görselleştirme için)
        goruntu = cv2.imread(test_yollari[indeks])
        goruntu = cv2.cvtColor(goruntu, cv2.COLOR_BGR2RGB)
        goruntu = cv2.resize(goruntu, (224, 224))

        gercek = SINIF_ISIMLERI[gercek_etiketler[indeks]]

        # --- Üst Satır: CNN Tahminleri ---
        if i < 5:
            ek = eksenler[0, i]
        else:
            ek = eksenler[1, i - 5]

        ek.imshow(goruntu)
        ek.axis("off")

        cnn_sinif = SINIF_ISIMLERI[cnn_tahminler[indeks]]
        cnn_guven = cnn_guvenler[indeks] * 100
        vit_sinif = SINIF_ISIMLERI[vit_tahminler[indeks]]
        vit_guven = vit_guvenler[indeks] * 100

        # Başlık rengini doğru/yanlış tahminine göre belirle
        cnn_renk = "green" if cnn_sinif == gercek else "red"
        vit_renk = "green" if vit_sinif == gercek else "red"

        baslik = (
            f"Gerçek: {gercek}\n"
            f"CNN: {cnn_sinif}, Güven: %{cnn_guven:.1f}\n"
            f"ViT: {vit_sinif}, Güven: %{vit_guven:.1f}"
        )

        # Başlık rengini belirle (her iki model de doğruysa yeşil)
        if cnn_sinif == gercek and vit_sinif == gercek:
            baslik_renk = "green"
        elif cnn_sinif != gercek and vit_sinif != gercek:
            baslik_renk = "red"
        else:
            baslik_renk = "darkorange"

        ek.set_title(baslik, fontsize=9, color=baslik_renk, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(kayit_yolu, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Tahmin görselleştirmesi kaydedildi: {kayit_yolu}")


# ========================== ANA PROGRAM ==========================

def main():
    """
    Programın ana akış fonksiyonu.
    Tüm adımlar sırasıyla çalıştırılır:
      1. Veri okuma ve ön işleme
      2. CNN modeli oluşturma ve eğitme
      3. ViT modeli oluşturma ve ince ayar (fine-tuning)
      4. Karşılaştırmalı grafiklerin çizilmesi
      5. Tahmin görselleştirmesi
    """
    print("=" * 70)
    print("  GÖĞÜS RÖNTGENİ İLE ZATÜRRE TESPİTİ")
    print("  İkili Model Karşılaştırması: CNN vs ViT")
    print("=" * 70)

    # ----------------------------------------------------------------
    # ADIM 1: VERİ OKUMA VE ÖN İŞLEME
    # ----------------------------------------------------------------
    print("\n" + "─" * 50)
    print("ADIM 1: Veri okuma ve ön işleme")
    print("─" * 50)

    # Eğitim ve test yollarını klasör adlarından oku
    egitim_yollari, egitim_etiketleri = goruntu_yollarini_ve_etiketleri_oku(EGITIM_DIZINI)
    test_yollari, test_etiketleri = goruntu_yollarini_ve_etiketleri_oku(TEST_DIZINI)

    # Verileri karıştır (hoca kuralı)
    birlesik = list(zip(egitim_yollari, egitim_etiketleri))
    random.shuffle(birlesik)
    egitim_yollari, egitim_etiketleri = zip(*birlesik)
    egitim_yollari, egitim_etiketleri = list(egitim_yollari), list(egitim_etiketleri)

    # Eğitim verisini eğitim + doğrulama olarak böl
    (eg_yollar, dg_yollar,
     eg_etiketler, dg_etiketler) = train_test_split(
        egitim_yollari, egitim_etiketleri,
        test_size=DOGRULAMA_ORANI,
        random_state=RASTGELELIK_TOHUMU,
        stratify=egitim_etiketleri  # Sınıf dengesini koru
    )

    print(f"\n  Eğitim seti   : {len(eg_yollar)} görüntü")
    print(f"  Doğrulama seti: {len(dg_yollar)} görüntü")
    print(f"  Test seti     : {len(test_yollari)} görüntü")

    # ----------------------------------------------------------------
    # ADIM 2: CNN MODELİ (KERAS — GRİ SEVİYE)
    # ----------------------------------------------------------------
    print("\n" + "─" * 50)
    print("ADIM 2: CNN modeli eğitimi (Keras)")
    print("─" * 50)

    # CNN için görüntüleri gri seviye olarak yükle ve normalize et
    print("\n  CNN için görüntüler yükleniyor (gri seviye, tek kanal)...")
    X_eg_cnn, y_eg_cnn = goruntuleri_yukle_cnn(eg_yollar, eg_etiketler)
    X_dg_cnn, y_dg_cnn = goruntuleri_yukle_cnn(dg_yollar, dg_etiketler)
    X_test_cnn, y_test_cnn = goruntuleri_yukle_cnn(test_yollari, test_etiketleri)
    print(f"  Eğitim şekli : {X_eg_cnn.shape}")
    print(f"  Doğrulama    : {X_dg_cnn.shape}")
    print(f"  Test şekli   : {X_test_cnn.shape}")

    # CNN modelini oluştur ve eğit
    print("\n  CNN mimarisi oluşturuluyor...")
    cnn_model = cnn_modeli_olustur()
    cnn_model.summary()

    print("\n  CNN eğitimi başlıyor...\n")
    cnn_gecmis = cnn_modelini_egit(cnn_model, X_eg_cnn, y_eg_cnn, X_dg_cnn, y_dg_cnn)

    # CNN test tahminleri
    cnn_tahminler, cnn_guvenler = cnn_tahmin_yap(cnn_model, X_test_cnn)
    cnn_test_dogruluk = np.mean(cnn_tahminler == y_test_cnn)
    print(f"\n  ★ CNN Test Doğruluğu: {cnn_test_dogruluk:.4f} ({cnn_test_dogruluk * 100:.2f}%)")

    # ----------------------------------------------------------------
    # ADIM 3: ViT MODELİ (PyTorch + timm — RGB)
    # ----------------------------------------------------------------
    print("\n" + "─" * 50)
    print("ADIM 3: ViT modeli ince ayar (Fine-Tuning)")
    print("─" * 50)

    # ViT için dönüşüm pipeline'ları oluştur
    egitim_donusumu, deger_donusumu = vit_donusumleri_olustur()

    # PyTorch Dataset ve DataLoader'ları oluştur
    egitim_veri_seti = RontgenVeriSeti(eg_yollar, eg_etiketler, donusum=egitim_donusumu)
    dogrulama_veri_seti = RontgenVeriSeti(dg_yollar, dg_etiketler, donusum=deger_donusumu)
    test_veri_seti = RontgenVeriSeti(test_yollari, test_etiketleri, donusum=deger_donusumu)

    egitim_yukleyici = DataLoader(
        egitim_veri_seti, batch_size=BATCH_BOYUTU, shuffle=True, num_workers=0
    )
    dogrulama_yukleyici = DataLoader(
        dogrulama_veri_seti, batch_size=BATCH_BOYUTU, shuffle=False, num_workers=0
    )
    test_yukleyici = DataLoader(
        test_veri_seti, batch_size=BATCH_BOYUTU, shuffle=False, num_workers=0
    )

    # ViT modelini yükle ve eğit
    print("\n  Önceden eğitilmiş ViT modeli yükleniyor...")
    vit_model = vit_modeli_olustur(sinif_sayisi=2)

    print("\n  ViT ince ayar eğitimi başlıyor...")
    vit_gecmis = vit_modelini_egit(vit_model, egitim_yukleyici, dogrulama_yukleyici)

    # ViT test tahminleri
    vit_tahminler, vit_guvenler, vit_gercek = vit_tahmin_yap(vit_model, test_yukleyici)
    vit_test_dogruluk = np.mean(vit_tahminler == vit_gercek)
    print(f"\n  ★ ViT Test Doğruluğu: {vit_test_dogruluk:.4f} ({vit_test_dogruluk * 100:.2f}%)")

    # ----------------------------------------------------------------
    # ADIM 4: KARŞILAŞTIRMALI GRAFİKLER
    # ----------------------------------------------------------------
    print("\n" + "─" * 50)
    print("ADIM 4: Karşılaştırmalı grafiklerin çizilmesi")
    print("─" * 50)

    grafik_yolu = os.path.join(PROJE_DIZINI, "egitim_karsilastirma_grafikleri.png")
    egitim_grafiklerini_ciz(cnn_gecmis, vit_gecmis, grafik_yolu)

    # ----------------------------------------------------------------
    # ADIM 5: TAHMİN GÖRSELLEŞTİRMESİ
    # ----------------------------------------------------------------
    print("\n" + "─" * 50)
    print("ADIM 5: Tahmin görselleştirmesi (Rastgele 10 test görüntüsü)")
    print("─" * 50)

    tahmin_gorsel_yolu = os.path.join(PROJE_DIZINI, "tahmin_gorsellestirmesi.png")
    tahmin_gorsellestirme(
        test_yollari, test_etiketleri,
        cnn_tahminler, cnn_guvenler,
        vit_tahminler, vit_guvenler,
        tahmin_gorsel_yolu
    )

    # ----------------------------------------------------------------
    # SONUÇ ÖZETİ
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  SONUÇ ÖZETİ")
    print("=" * 70)
    print(f"  CNN Test Doğruluğu : {cnn_test_dogruluk * 100:.2f}%")
    print(f"  ViT Test Doğruluğu : {vit_test_dogruluk * 100:.2f}%")
    print(f"  Fark (ViT - CNN)   : {(vit_test_dogruluk - cnn_test_dogruluk) * 100:+.2f}%")
    print("─" * 70)
    print(f"  Kaydedilen Dosyalar:")
    print(f"    1. {grafik_yolu}")
    print(f"    2. {tahmin_gorsel_yolu}")
    print("=" * 70)
    print("  ✓ Proje başarıyla tamamlandı!")
    print("=" * 70)


if __name__ == "__main__":
    main()

