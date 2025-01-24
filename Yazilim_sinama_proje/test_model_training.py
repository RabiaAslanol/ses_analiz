import unittest
import os
import wave
import numpy as np
from model import veri_kumesi_olusturma, modeli_egit, MODEL_DOSYASI

class TestModelTraining(unittest.TestCase):
    def setUp(self):
        """
        Testler başlamadan önce çalışır. Örnek veri dizinini ve sahte ses dosyalarını oluşturur.
        """
        self.veri_dizini = "ornek_veriler"
        os.makedirs(self.veri_dizini, exist_ok=True)  # Ana dizini oluştur

        # Örnek alt klasörler ve sahte ses dosyaları oluştur
        for i in range(2):  # 2 tane klasör (örnek konuşmacı)
            klasor = os.path.join(self.veri_dizini, f"klasor_{i}")
            os.makedirs(klasor, exist_ok=True)  # Alt klasörleri oluştur

            # Geçerli .wav dosyası oluştur
            dosya_yolu = os.path.join(klasor, f"ornek_{i}.wav")
            with wave.open(dosya_yolu, "wb") as wf:
                wf.setnchannels(1)  # Mono
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(44100)  # 44.1 kHz
                sine_wave = (np.sin(2 * np.pi * np.arange(44100) * 440 / 44100)).astype(np.int16)
                wf.writeframes(sine_wave.tobytes())
                
    def tearDown(self):
        """
        Testlerden sonra çalışır. Oluşturulan geçici dizinleri ve dosyaları temizler.
        """
        for root, dirs, files in os.walk(self.veri_dizini, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))  # Dosyaları sil
            for name in dirs:
                os.rmdir(os.path.join(root, name))  # Klasörleri sil
        os.rmdir(self.veri_dizini)  # Ana dizini sil

    def test_veri_kumesi_olusturma(self):
        """
        veri_kumesi_olusturma fonksiyonunu test eder.
        """
        veri_kumesi = veri_kumesi_olusturma(self.veri_dizini, verbose=True)
        self.assertGreater(len(veri_kumesi), 0)  # Veri kümesinin dolu olduğunu kontrol eder

    def test_modeli_egit(self):
        """
        modeli_egit fonksiyonunu test eder.
        """
        veri_kumesi = veri_kumesi_olusturma(self.veri_dizini, verbose=True)
        modeli_egit(veri_kumesi, MODEL_DOSYASI)
        self.assertTrue(os.path.exists(MODEL_DOSYASI))  # Model dosyasının kaydedildiğini doğrular

if __name__ == "__main__":
    unittest.main()
