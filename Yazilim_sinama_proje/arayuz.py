import wave
import numpy as np
import sounddevice as sd
import speech_recognition as sr
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QLabel
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from model import veri_kumesi_olusturma, modeli_egit, ses_tanima, kelime_sayisi_hesapla,duygu_analizi,metin_konusu_analizi
from model import MODEL_DOSYASI
import librosa
import matplotlib.pyplot as plt

VERI_DIZIN = "veriler"
KAYIT_DOSYASI = "kayit.wav"
# Veri kümesini oluştur ve modeli eğit
veri_kumesi = veri_kumesi_olusturma(VERI_DIZIN, verbose=True)
modeli_egit(veri_kumesi)
class RealTimeAudioRecorder(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gerçek Zamanlı Ses Kaydı ve Tanıma")
        self.setGeometry(100, 100, 800, 600)

        # Ana widget ve layout
        ana_widget = QWidget()
        self.setCentralWidget(ana_widget)
        ana_layout = QVBoxLayout()

        # Butonlar
        buton_layout = QHBoxLayout()

        # Kaydı Başlat Butonu
        self.kayit_butonu = QPushButton("Kaydı Başlat")
        self.kayit_butonu.clicked.connect(self.start_recording)
        self.kayit_butonu.setStyleSheet(
            "background-color: #4CAF50; color: white; font-size: 16px; padding: 10px; border-radius: 8px;"
        )
        buton_layout.addWidget(self.kayit_butonu)

        # Kaydı Durdur Butonu
        self.durdur_butonu = QPushButton("Kaydı Durdur")
        self.durdur_butonu.clicked.connect(self.stop_recording)
        self.durdur_butonu.setStyleSheet(
            "background-color: #f44336; color: white; font-size: 16px; padding: 10px; border-radius: 8px;"
        )
        buton_layout.addWidget(self.durdur_butonu)

        # Tahmin Yap Butonu
        self.tahmin_butonu = QPushButton("Konuşmacıyı Tahmin Et")
        self.tahmin_butonu.clicked.connect(self.make_prediction)
        self.tahmin_butonu.setStyleSheet(
            "background-color: #2196F3; color: white; font-size: 16px; padding: 10px; border-radius: 8px;"
        )
        buton_layout.addWidget(self.tahmin_butonu)

        # Kelime Sayısını Hesapla Butonu
        self.kelime_sayisi_butonu = QPushButton("Kelime Sayısını Hesapla")
        self.kelime_sayisi_butonu.clicked.connect(self.calculate_word_count)
        self.kelime_sayisi_butonu.setStyleSheet(
            "background-color: #FF9800; color: white; font-size: 16px; padding: 10px; border-radius: 8px;"
        )
        buton_layout.addWidget(self.kelime_sayisi_butonu)

        ana_layout.addLayout(buton_layout)

        # Duygu Tahmini Butonu
        self.duygu_tahmini_butonu = QPushButton("Duygu Tahmini Yap")
        self.duygu_tahmini_butonu.clicked.connect(self.analyze_emotion)
        self.duygu_tahmini_butonu.setStyleSheet(
            "background-color: #9C27B0; color: white; font-size: 16px; padding: 10px; border-radius: 8px;"
        )
        buton_layout.addWidget(self.duygu_tahmini_butonu)
        #Konu Analizi
        self.konuyu_analyze_butonu = QPushButton("Konuşulan Konuyu Analiz Et")
        self.konuyu_analyze_butonu.clicked.connect(self.analyze_topic)
        self.konuyu_analyze_butonu.setStyleSheet(
            "background-color: #FF5722; color: white; font-size: 16px; padding: 10px; border-radius: 8px;"
        )
        buton_layout.addWidget(self.konuyu_analyze_butonu)                
        
        # Tahmin Sonucu
        self.sonuc_label = QLabel("Sonuç: Henüz bir tahmin yapılmadı.")
        self.sonuc_label.setStyleSheet("font-size: 18px; color: #333;")
        ana_layout.addWidget(self.sonuc_label)

        # Grafik Alanı
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        ana_layout.addWidget(self.canvas)

        ana_widget.setLayout(ana_layout)

        # Ses kaydı ayarları
        self.fs = 44100  # Örnekleme frekansı
        self.audio_data = []  # Ses verisi
        self.recording_in_progress = False  # Kaydın durumu

        # Timer ayarı, her 100ms'de grafik güncelleme
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_graph)
        self.timer.start(100)  # Grafikleri 100ms aralıklarla güncelle

    def audio_callback(self, indata, frames, time, status):
        """Anlık ses verilerini toplar ve grafikleri günceller."""
        if self.recording_in_progress:
            self.audio_data.append(indata.copy())

    def start_recording(self):
        """Kaydı başlatır."""
        print("Kayıt başladı...")
        self.recording_in_progress = True
        self.audio_data = []  # Yeni bir kayıt başlatıyoruz
        self.stream = sd.InputStream(
            samplerate=self.fs, channels=1, dtype='int16', callback=self.audio_callback,
            blocksize=2048  # Ses kaydını verimli bir şekilde almak için blocksize
        )
        self.stream.start()

    def stop_recording(self):
        """Kaydı durdurur ve ses dosyasını kaydeder."""
        if not self.recording_in_progress:
            print("Kayıt zaten durduruldu.")
            return

        self.recording_in_progress = False
        self.stream.stop()
        self.stream.close()

        if len(self.audio_data) > 0:
            audio_data = np.concatenate(self.audio_data, axis=0)
            with wave.open("kayit.wav", "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.fs)
                wf.writeframes(audio_data.astype(np.int16).tobytes())
            print(f"Ses kaydedildi: kayit.wav")

    def make_prediction(self):
        """Tahmin yapar ve sonucu grafikle gösterir."""
        try:
            tahmin_etiketleri, olasiliklar = ses_tanima(MODEL_DOSYASI, "kayit.wav")

            if isinstance(tahmin_etiketleri, str):  # Hata mesajı varsa
                self.sonuc_label.setText(f"Sonuç: {tahmin_etiketleri}")
                return

            # Grafik oluşturma
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.pie(
                olasiliklar, 
                labels=tahmin_etiketleri, 
                autopct='%1.1f%%', 
                startangle=140, 
                colors=plt.cm.Paired(np.arange(len(tahmin_etiketleri)))
            )
            ax.set_title("Konuşmacı Tahmini")
            self.canvas.draw()

            # En yüksek olasılığa sahip etiketi göster
            en_yuksek_index = np.argmax(olasiliklar)
            en_yuksek_tahmin = tahmin_etiketleri[en_yuksek_index]
            en_yuksek_olasilik = olasiliklar[en_yuksek_index] * 100
            self.sonuc_label.setText(f"Sonuç: {en_yuksek_tahmin} ({en_yuksek_olasilik:.2f}%)")
        except Exception as e:
            self.sonuc_label.setText(f"Hata: {e}")

    def calculate_word_count(self):
        """Kelime sayısını hesaplar ve sonucu gösterir."""
        try:
            # Grafik alanını temizle
            self.figure.clear()
            self.canvas.draw()
            # Ses kaydını metne dönüştür
            metin = self.sesten_metne_donustur("kayit.wav")
            self.sonuc_label.setText(f"Metin: {metin}\n")  # Metni yeni bir satırda göster
            kelime_sayisi = kelime_sayisi_hesapla(metin)
            self.sonuc_label.setText(f"Metin: {metin}\nKelime Sayısı: {kelime_sayisi}")
        except Exception as e:
            self.sonuc_label.setText(f"Hata: {e}")

    def sesten_metne_donustur(self, dosya_adi):
        """Google Speech API kullanarak ses kaydını metne dönüştürür."""
        recognizer = sr.Recognizer()
        with sr.AudioFile(dosya_adi) as source:
            try:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data, language="tr-TR")
                return text
            except sr.UnknownValueError:
                return "Anlaşılamadı."
            except sr.RequestError:
                return "API erişim hatası."
    #duygu tahmini
    def analyze_emotion(self):
        """Metin üzerinde duygu analizi yapar ve sonucu gösterir."""
        try:
            # Ses kaydını metne dönüştür
            metin = self.sesten_metne_donustur("kayit.wav")
            if metin:
                # Model'deki duygu analizi fonksiyonunu çağır
                duygu_sonuclari = duygu_analizi(metin)

                # Duygu yüzdelerini birleştir
                duygu_metni = "\n".join([f"{duygu}: %{oran}" for duygu, oran in duygu_sonuclari.items()])
                self.sonuc_label.setText(f"Metin: {metin}\nDuygu Analizi:\n{duygu_metni}")
            else:
                self.sonuc_label.setText("Duygu analizi yapılacak metin bulunamadı.")
        except Exception as e:
            self.sonuc_label.setText(f"Hata: {e}")

    # 6. Story: Konunun analizi
    def analyze_topic(self):
        """Konuşulan konunun analizini yapar ve grafiği gösterir."""
        try:
            metin = self.sesten_metne_donustur(KAYIT_DOSYASI)
            if metin in ["Anlaşılamadı.", "API erişim hatası."]:
                self.sonuc_label.setText("Konunun analizi yapılamadı: Ses metne dönüştürülemedi.")
                return

            # Konu analizi yap
            categories, scores = metin_konusu_analizi(metin)
            if not categories or not scores:
                self.sonuc_label.setText("Konu analizi yapılamadı.")
                return

            # Mevcut grafikleri temizle ve yeni grafiği çiz
            self.figure.clear()
            ax = self.figure.add_subplot(1, 1, 1)
            ax.pie(scores, labels=categories, startangle=140, autopct='%1.1f%%', colors=plt.cm.Paired(np.arange(len(categories))))
            ax.set_title("Konu Analizi Sonucu")
            self.canvas.draw()

            # En iyi kategoriyi metin olarak göster
            best_category = categories[0]
            confidence = scores[0] * 100
            self.sonuc_label.setText(f"Konu: {best_category} ({confidence:.2f}%)")
        except Exception as e:
            self.sonuc_label.setText(f"Hata: {e}")

    def update_graph(self):
        """Grafikleri günceller."""
        if not self.recording_in_progress or len(self.audio_data) == 0:
            return

        audio_data = np.concatenate(self.audio_data, axis=0)

        # Dalga formunu çiz
        self.figure.clear()
        ax1 = self.figure.add_subplot(2, 1, 1)
        ax1.plot(audio_data[-self.fs:], color='blue')
        ax1.set_title("Dalga Formu")
        ax1.set_xlabel("Zaman (örnek)")
        ax1.set_ylabel("Genlik")

        # Histogramı çiz
        ax2 = self.figure.add_subplot(2, 1, 2)
        ax2.hist(audio_data, bins=50, color='blue', alpha=0.7)
        ax2.set_title("Genlik Histogramı")
        ax2.set_xlabel("Genlik")
        ax2.set_ylabel("Frekans")

        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication([])
    window = RealTimeAudioRecorder()
    window.show()
    app.exec_()
