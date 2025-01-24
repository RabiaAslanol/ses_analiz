import unittest
from model import kelime_sayisi_hesapla

class TestBackendFunctions(unittest.TestCase):
    def test_kelime_sayisi_hesapla(self):
        # Normal bir cümle için kelime sayısını test ediyoruz
        metin = "Bu bir test cümlesidir"
        beklenen_kelime_sayisi = 4
        self.assertEqual(kelime_sayisi_hesapla(metin), beklenen_kelime_sayisi)

    def test_bos_metin(self):
        # Boş metin için kelime sayısının 0 döndüğünü kontrol ediyoruz
        metin = ""
        beklenen_kelime_sayisi = 0
        self.assertEqual(kelime_sayisi_hesapla(metin), beklenen_kelime_sayisi)

if __name__ == "__main__":
    unittest.main()
