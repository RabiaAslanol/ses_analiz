import unittest
from PyQt5.QtWidgets import QApplication
from arayuz import RealTimeAudioRecorder
from PyQt5.QtTest import QTest
from PyQt5.QtCore import Qt
import sys
import os
sys.path.append(os.path.abspath("model_klasoru"))  # "model_klasoru" ilgili klasörün yolu
from model import veri_kumesi_olusturma
ses_dizin = "veriler"
class TestGUI(unittest.TestCase):
    def setUp(self):
        self.ses_dizin = "veriler"  # Test için kullandığınız veri dizini

    def test_kayit_baslat_butonu(self):
        print("Kaydı Başlat butonuna tıklandı.")  # Sadece buton mesajı
        # İşlem yapabilirsiniz (örnek: dummy bir işlev çağrısı)
        veri_kumesi_olusturma(self.ses_dizin, verbose=False)

    def test_kelime_sayisi_hesapla_butonu(self):
        print("Kelime Sayısını Hesapla butonuna tıklandı.")  # Sadece buton mesajı
        veri_kumesi_olusturma(self.ses_dizin, verbose=False)

    def test_konusmaci_tahmini(self):
        print("Konuşmacıyı Tahmin Et butonuna tıklandı.")  # Sadece buton mesajı
        veri_kumesi_olusturma(self.ses_dizin, verbose=False)


if __name__ == "__main__":
    unittest.main()