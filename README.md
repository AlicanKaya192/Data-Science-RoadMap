# Alican Kaya Python Learning Repository

[Portfolio](https://alican-kaya.com/) | [LinkedIn](https://www.linkedin.com/in/alican-kaya-881650234/)

---

![License](https://img.shields.io/badge/license-Custom-blue) ![Version](https://img.shields.io/badge/version-3.14.2-blue) ![Language](https://img.shields.io/badge/language-Python-yellow) ![GitHub](https://img.shields.io/badge/GitHub-AlicanKaya192/PythonTopics.git-black?logo=github)

---

<img src="https://github.com/user-attachments/assets/7c5aefab-2a2d-4d28-afb6-fb2863392e6f" width="640" />

## 📑 İçindekiler
* [📌 Repository Hakkında](#-repository-hakkında)
* [📚 Öğrenim Yol Haritası ve İçerikler](#-öğrenim-yol-haritası-ve-içerikler)
* [📂 Ekstra Projeler ve Kaynaklar](#-ekstra-projeler-ve-kaynaklar)
* [📖 Proje Durumu ve İlerleme](#-proje-durumu-ve-ilerleme)
* [💡 Önerilen Çalışma Yöntemleri](#-önerilen-çalışma-yöntemleri)
* [🤝 Katkıda Bulunma](#-katkıda-bulunma)

---

## 📌 Repository Hakkında

Bu repository, Python programlama dili öğrenim sürecimde oluşturduğum notları, örnek kodları ve projeleri içeren kapsamlı bir kaynaktır. **Veri Bilimi ve Makine Öğrenimi** yol haritasını takip ederek; temel Python konularından başlayıp, ileri seviye veri analizi, özellik mühendisliği ve makine öğrenimi modellerine kadar uzanan bir yapı sunmaktadır.

Amacım, bu süreçte öğrendiklerimi organize bir şekilde belgelemek ve benzer yoldan geçenler için faydalı bir rehber oluşturmaktır.

---

## 📚 Öğrenim Yol Haritası ve İçerikler

Repository içerisindeki klasörler, öğrenim sırasına göre numaralandırılmıştır. Aşağıdaki adımları takiperek sistematik bir şekilde ilerleyebilirsiniz.

### 1️⃣ Çalışma Ortamı Ayarları
Python geliştirme ortamının kurulması ve yönetilmesi ile ilgili temel adımlar.
- **1.1 - setting_up_working_environment.py:** Çalışma ortamı kurulumu ve temel ayarlar.
- **1.2 - What is a virtual environment:** Sanal ortamların (Virtual Environment) önemi ve kullanımı.
- **1.3 - Package Management:** `conda` ve `pip` ile paket yönetimi ve bağımlılıklar.

### 2️⃣ Veri Yapıları
Python'un temel yapı taşları olan veri tiplerinin detaylı incelenmesi.
- **data_structures.py:** String, List, Dictionary, Tuple ve Set veri yapıları, metodları ve kullanım alanları.

### 3️⃣ Fonksiyonlar, Koşullar, Döngüler ve Comprehensions
Programlama mantığının temelleri ve fonksiyonel programlama araçları.
- **functions_conditions_loops_comprehensions.py:** Fonksiyon tanımlama, `if-else` yapıları, döngüler, `zip`, `lambda`, `map`, `filter`, `reduce` ve Comprehension yapıları.

### 4️⃣ Egzersizler (Python ve List Comprehensions)
Öğrenilen temel konuların pekiştirilmesi için pratik çalışmalar.
- **4.1_Python_Exercises.py:** Temel Python konuları üzerine alıştırmalar.
- **4.2_List_Comprehension_Exercises.py:** `Car_crashes` ve diğer veri setleri üzerinde List Comprehension pratikleri.

### 5️⃣ Numpy
Bilimsel hesaplamalar ve çok boyutlu dizi işlemleri.
- **data_analysis_numpy.py:** Array yapısı, boyutlandırma, indeksleme, fancy index ve matematiksel işlemler.

### 6️⃣ Pandas
Veri analizi ve manipülasyonu için en temel kütüphane.
- **1 - data_analysis_pandas.py:** Series ve DataFrame yapıları, veri okuma, filtreleme, `loc` & `iloc`, `groupby`, `pivot_table`.
- **2 - Pandas_exercise.py:** Pandas fonksiyonları üzerine pekiştirici alıştırmalar.

### 7️⃣ Veri Görselleştirme (Matplotlib & Seaborn)
Veriyi anlamlandırmak ve sunmak için görselleştirme teknikleri.
- **Veri_Görselleştirme_Matplotlib&Seaborn.py:** Çizgi, sütun, histogram, scatter plot grafikleri ve özelleştirme teknikleri.

### 8️⃣ Gelişmiş Fonksiyonel Keşifçi Veri Analizi (EDA)
Veri setini sistematik olarak analiz etme metodolojisi.
- **gelişmiş_fonksiyonel_keşifçi_veri_analizi.py:** Genel resim, kategorik/sayısal değişken analizi, hedef değişken analizi ve korelasyon analizi.

### 9️⃣ CRM Analitik
Müşteri İlişkileri Yönetimi ve veri odaklı pazarlama stratejileri.
- **9.1 CRM Giriş:** CRM kavramı, KPI'lar ve Cohort Analizi.
- **9.2 RFM Analizi:** Recency, Frequency, Monetary metrikleri ile müşteri segmentasyonu.
- **9.3 CLTV (Müşteri Yaşam Boyu Değeri):** Müşteri değerinin hesaplanması ve tahmini (Prediction).
- **9.4 Projeler:**
    - `FLO_RFM.py`: FLO verisi ile RFM analizi ve müşteri segmentasyonu.
    - `FLO_CLTV_Prediction.py`: FLO verisi ile CLTV tahmini.

### 1️⃣2️⃣ Feature Engineering (Özellik Mühendisliği)
Ham veriden makine öğrenimi modelleri için anlamlı özellikler türetme sanatı.
- **12.1 Aykırı Değerler (Outliers):** Tespiti ve baskılama yöntemleri.
- **12.2 Eksik Değerler (Missing Values):** Eksik verilerin analizi ve doldurma stratejileri.
- **12.3 Encoding:** Label Encoding, One-Hot Encoding, Rare Encoding.
- **12.4 Feature Extraction:** Metin, tarih ve diğer verilerden yeni özellikler çıkarma.
- **12.6 Extra:** `Diabete_Feature_Engineering.py` ile diyabet veri seti üzerinde uçtan uca özellik mühendisliği uygulaması.

### 1️⃣3️⃣ Machine Learning (Makine Öğrenimi)
Veriden öğrenen modellerin kurulması ve değerlendirilmesi.
- **Projeler:**
    - `HOUSE_PRICE_PREDICTION`: Regresyon modelleri ile ev fiyatı tahmini.
    - `Telco_Churn`: Sınıflandırma modelleri ile müşteri terk analizi.
- **Değerlendirme:** Başarı metrikleri ve hata değerlendirme tabloları.

---

## 📂 Ekstra Projeler ve Kaynaklar

- **Armut ARL Projesi:** Birliktelik Kuralı Öğrenimi (Association Rule Learning) üzerine gerçek hayat senaryosu.
- **CheatSheets:** Python, Pandas, Numpy ve Git için hızlı başvuru kağıtları.
- **Datasets:** Çalışmalarda kullanılan veri setleri arşivi.
- **Mülakat Soruları:** Teknik mülakatlara hazırlık için soru ve çözümler.
- **Mentor Çözümleri:** Örnek problemlerin alternatif ve profesyonel çözümleri.

---

## 📖 Proje Durumu ve İlerleme

| Bölüm / Konu | Durum |
|--------------|-------|
| 1 - Çalışma Ortamı | ✅ Tamamlandı |
| 2 - Veri Yapıları | ✅ Tamamlandı |
| 3 - Fonksiyonlar & Döngüler | ✅ Tamamlandı |
| 4 - Egzersizler | ✅ Tamamlandı |
| 5 - Numpy | ✅ Tamamlandı |
| 6 - Pandas | ✅ Tamamlandı |
| 7 - Veri Görselleştirme | ✅ Tamamlandı |
| 8 - Keşifçi Veri Analizi (EDA) | ✅ Tamamlandı |
| 9 - CRM Analitik | ✅ Tamamlandı |
| 10 - Ölçümleme Problemleri | 🚧 Devam Ediyor |
| 11 - Tavsiye Sistemleri | ❌ Planlanıyor |
| 12 - Feature Engineering | 🚧 Devam Ediyor |
| 13 - Machine Learning | 🚧 Devam Ediyor |
| Time Series | ❌ Planlanıyor |
| SQL | ❌ Planlanıyor |

---

## 💡 Önerilen Çalışma Yöntemleri

1. **Sırayı Takip Edin:** Konular birbirinin üzerine inşa edildiği için klasör numaralarına göre ilerlemeniz tavsiye edilir.
2. **Uygulama Yapın:** Sadece kodları okumak yerine, `Datasets` klasöründeki verileri kullanarak kendi analizlerinizi yapın.
3. **Projeleri İnceleyin:** Özellikle `CRM` ve `Machine Learning` klasörlerindeki uçtan uca projeleri (pipeline) anlamaya çalışın.

### Algoritma ve Kod Pratiği Siteleri
* **Hackerrank:** Başlangıç ve orta seviye sorular için.
* **Codewars:** Küçük, pratik odaklı görevler.
* **Leetcode:** Orta ve ileri seviye kullanıcılar için (önce Hackerrank/Codewars yapılmalı).
* **Spoj:** Sadece sorular içerir, kod editörü yok. Diğer sitelerden sonra kullanılabilir.

> **Not:** Bu sitelere istediğiniz zaman girip ufak pratikler yapabilirsiniz. Veri seti pratiğine daha fazla vakit ayırmanız önerilir.

---

## 🤝 Katkıda Bulunma

Python öğrenimi sürecinde bu kaynakların geliştirilmesine katkıda bulunmak isteyenler için PR (Pull Request) ve issue'lar açmak tamamen açıktır.
