import librosa
from librosa.util.exceptions import ParameterError
import numpy as np
from scipy.signal import wiener
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import random
from transformers import pipeline
from sklearn.preprocessing import StandardScaler
from deep_translator import GoogleTranslator
import audioread
from transformers import logging
import warnings
from librosa.feature import spectral_contrast, chroma_stft
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from googletrans import Translator
from textblob import TextBlob
warnings.filterwarnings("ignore", category=RuntimeWarning)

MODEL_DOSYASI = "ses_tanima_modeli.pkl"
logging.set_verbosity_error()
# Gürültü azaltma fonksiyonu
def gürültü_azaltma(y, verbose=False):
    if np.var(y) < 1e-6:  # Çok düşük varyans kontrolü
        if verbose:
            print("Gürültü azaltma atlandı (düşük varyans)")
        return y  # Gürültü azaltma uygulamadan döndür
    try:
        return wiener(y, mysize=3)  # Gürültü azaltma uygulanıyor
    except Exception as e:
        if verbose:
            print(f"Gürültü azaltma sırasında hata: {e}")
        return y  # Gürültü azaltma başarısızsa orijinal sinyali döndür

# Ses özelliklerini çıkarma fonksiyonu
def ses_ozelliklerini_cikar(y, sr, n_mfcc=13):
    try:
        # Gürültü azaltma
        y = gürültü_azaltma(y)

        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc.T, axis=0)

        # Mel-spektrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, fmax=8000)
        mel_mean = np.mean(mel_spec.T, axis=0)

        # ZCR ve RMS
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        rms = np.mean(librosa.feature.rms(y=y))

        # Ek özellikler
        spectral_contrast_features = np.mean(spectral_contrast(y=y, sr=sr).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)

        # Tüm özellikleri birleştir
        return np.hstack([mfcc_mean, mel_mean, zcr, rms, spectral_contrast_features, chroma])
    except Exception as e:
        print(f"Ses özellikleri çıkarılamadı: {e}")
        return None
    
# Ses dosyasını küçük parçalara ayırma fonksiyonu
def split_audio(dosya_adi, window_size=1, hop_length=1):
    try:
        y, sr = librosa.load(dosya_adi, sr=None)
    except audioread.NoBackendError:
        print(f"Hata: {dosya_adi} ses formatı desteklenmiyor.")
        return [], None
    except Exception as e:
        print(f"Beklenmeyen hata oluştu: {e}")
        return [], None

    samples_per_window = int(window_size * sr)
    samples_per_hop = int(hop_length * sr)
    windows = [y[i:i + samples_per_window] for i in range(0, len(y), samples_per_hop)]
    return windows, sr

# Veri kümesi oluşturma fonksiyonu
def veri_kumesi_olusturma(ses_dizin, window_size=1, hop_length=1, verbose=False):
    veriler = []
    toplam_dosya = 0
    islenen_dosya = 0
    for klasor in os.listdir(ses_dizin):
        klasor_yolu = os.path.join(ses_dizin, klasor)
        if os.path.isdir(klasor_yolu):
            for dosya in os.listdir(klasor_yolu):
                if dosya.endswith(".wav"):
                    dosya_yolu = os.path.join(klasor_yolu, dosya)
                    try:
                            # Dosya geçerli mi kontrol edin
                            with audioread.audio_open(dosya_yolu):
                                pass
                    except Exception:
                            if verbose:
                                print(f"Toplam dosya: {toplam_dosya}, Başarıyla işlenen dosyalar: {islenen_dosya}")
                            continue          
                    if verbose:
                        print(f"İşleniyor: {dosya_yolu}")
                    try:
                        windows, sr = split_audio(dosya_yolu, window_size, hop_length)
                        for window in windows:
                            ozellikler = ses_ozelliklerini_cikar(y=window, sr=sr)
                            if ozellikler is not None:  # Özellikler boş değilse ekle
                                veriler.append([ozellikler, klasor])
                    except Exception as e:
                        if verbose:
                            print(f"Hata oluştu: {e}")
    return veriler

# Eğitim veri seti hazırlama fonksiyonu
def egitim_verisi_hazirla(ses_dizin, window_size=1, hop_length=1):
    print("Eğitim veri seti hazırlanıyor...")
    veri_kumesi = veri_kumesi_olusturma(ses_dizin, window_size, hop_length)
    print(f"Toplam örnek sayısı: {len(veri_kumesi)}")
    return veri_kumesi

# Modeli eğitme fonksiyonu
def modeli_egit(veri_kumesi, model_dosyasi=MODEL_DOSYASI):
    try:
        print("Model eğitiliyor...")

        # Özellik ve etiketleri ayır
        X = np.array([veri[0] for veri in veri_kumesi])
        y = np.array([veri[1] for veri in veri_kumesi])

        # Boş veri kontrolü
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Eğitim için yeterli veri bulunamadı!")

        print(f"Eğitim veri seti boyutları: {X.shape}, {y.shape}")

        # Etiketleri encode et
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        # Eğitim ve test setlerine ayır
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        # Normalize et
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        print(f"Eğitim seti boyutu: {X_train.shape}, Test seti boyutu: {X_test.shape}")

        # Logistic Regression modelini eğit
        model = LogisticRegression(max_iter=1000, C=1.0)
        model.fit(X_train, y_train)

        # Eğitim doğruluğunu yazdır
        train_accuracy = model.score(X_train, y_train)
        print(f"Eğitim doğruluğu: {train_accuracy:.2f}")

        # Modeli kaydet
        with open(model_dosyasi, "wb") as f:
            pickle.dump({"model": model, "label_encoder": label_encoder, "scaler": scaler}, f)

        print(f"Model kaydedildi: {model_dosyasi}")

    except Exception as e:
        print(f"Model eğitimi sırasında hata oluştu: {e}")

# Ses tanıma fonksiyonu
def ses_tanima(model_dosyasi, dosya_adi, esik=0.8):
    try:
        with open(model_dosyasi, "rb") as f:
            data = pickle.load(f)
            model = data["model"]
            label_encoder = data["label_encoder"]
            scaler = data["scaler"]

        # Ses dosyasını yükle
        y, sr = librosa.load(dosya_adi, sr=None)

        windows, _ = split_audio(dosya_adi, window_size=1, hop_length=1)

        tahminler = []
        tum_olasiliklar = []
        for window in windows:
            ozellikler = ses_ozelliklerini_cikar(window, sr)
            if ozellikler is None:
                return "Ses özellikleri çıkarılamadı."

            # Veriyi scaler ile normalize et
            ozellikler = scaler.transform([ozellikler])

            # Tahmin yap
            tahmin_olasiliklari = model.predict_proba(ozellikler)[0]
            max_olasilik = max(tahmin_olasiliklari)
            tahmin = model.predict(ozellikler)[0]

            tum_olasiliklar.append(tahmin_olasiliklari)
            if max_olasilik < esik:
                tahminler.append("Tanımlanamayan Ses")
            else:
                tahminler.append(tahmin)

        # Ortalama olasılıkları hesapla
        ortalama_olasiliklar = np.mean(tum_olasiliklar, axis=0)
        tahmin_etiketleri = label_encoder.classes_

        return tahmin_etiketleri, ortalama_olasiliklar

    except Exception as e:
        return f"Hata: {e}"

# kelime_sayisi_hesapla fonksiyonu ekleniyor
def kelime_sayisi_hesapla(metin):
    """
    Verilen metni kullanarak kelime sayısını hesaplar.
    """
    if metin:
        # Fazladan boşlukları temizle
        kelimeler = metin.strip().split()
        return len(kelimeler)
    return 0
def combine_predictions(ses_duygu, ses_olasilik, metin_duygu, metin_olasilik):
    duygular = ["mutlu", "üzgün", "nötr"]
    skorlar = {duygu: 0 for duygu in duygular}

    # Ağırlıkları dengeleyin
    skorlar[ses_duygu] += ses_olasilik * 0.5
    skorlar[metin_duygu] += metin_olasilik * 0.5

    toplam_skor = sum(skorlar.values())
    yuzdeler = {duygu: (skor / toplam_skor) * 100 for duygu, skor in skorlar.items()}

    en_yuksek_duygu = max(yuzdeler, key=yuzdeler.get)
    return en_yuksek_duygu, yuzdeler[en_yuksek_duygu]

def duygu_analizi(metin):
    """
    Verilen metni Türkçe'den İngilizce'ye çevirerek duygu analizi yapar ve yüzdelik oran döndürür.
    """
    try:
        # Türkçe metni İngilizce'ye çevir
        translator = Translator()
        translated_text = translator.translate(metin, src='tr', dest='en').text

        # Çevrilen metin üzerinde duygu analizi
        blob = TextBlob(translated_text)
        polarite = blob.sentiment.polarity  # Polarite -1 (negatif) ile 1 (pozitif) arasında

        # Duygu oranlarını hesaplama
        if polarite > 0:
            pozitif = polarite * 100
            negatif = 0
        elif polarite < 0:
            pozitif = 0
            negatif = abs(polarite) * 100
        else:
            pozitif = 0
            negatif = 0

        notr = 100 - (pozitif + negatif)  # Nötr oranı kalan kısmı tamamlar

        # Sonuç döndür
        return {
            "Mutlu": round(pozitif, 2),
            "Üzgün": round(negatif, 2),
        }
    except Exception as e:
        return f"Duygu analizi sırasında hata: {e}"

    
   #Konu analizi için model yükleme 
topic_analyzer = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
# Konu analizi fonksiyonu
def metin_konusu_analizi(metin):
    try:
        translated_text = GoogleTranslator(source='tr', target='en').translate(metin)
        categories = ["Education", "Health", "Technology", "Sports", "Art"]
        result = topic_analyzer(translated_text, candidate_labels=categories)
        return result["labels"], result["scores"]
    except Exception as e:
        return [], []

def plot_topic_analysis(categories, scores, canvas):
    try:
        canvas.figure.clear()
        ax = canvas.figure.add_subplot(111)
        ax.pie(scores, labels=categories, startangle=140, autopct='%1.1f%%', colors=plt.cm.Paired(np.arange(len(categories))))
        ax.set_title("Konu Analizi Sonucu")
        canvas.draw()
    except Exception as e:
        print(f"Grafik oluşturulamadı: {e}")