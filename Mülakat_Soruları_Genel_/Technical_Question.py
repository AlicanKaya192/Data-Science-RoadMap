# 1 - Python mı, R mı? Hangisini neden kullanıyorsun? Biz şirketimizde R kullanıyoruz bu konu ile ilgili düşüncelerin nelerdir ?

# 2 - C ve Java varken neden Python?

# 3 - Aşağıdaki fonksiyonun her satırında ne yapıldığını anlatınız. Bu fonksiyonu enumerate ile yazabilir miydik? Yazabiliyor olsaydık bu ne kazandırırdı?

def alternating(string):
      new_string = ""

      for string_index in range(len(string)):

          if string_index % 2 == 0:
              new_string += string[string_index].upper()

          else:
              new_string += string[string_index].lower()
      print(new_string)


# 4 - Elinize bir veri geldiğinde ilk analiz aşamasında neler yaparsınız? Keşifçi veri analizi basamaklarını anlatır mısınız?

  # Veri elimize ulaştığında öncelikle veri yapısını, veri tiplerini ve eksik/hatalı değerleri incelerim. Ardından sayısal ve kategorik
  # değişkenler için özet istatistikler ve dağılımları analiz eder, değişkenler arası ilişkiler ve korelasyonları değerlendiririm.
  # Gerektiğinde veri dönüşümleri ve temizliği yaparak veriyi modelleme veya ileri analizler için hazır hâle getiririm.