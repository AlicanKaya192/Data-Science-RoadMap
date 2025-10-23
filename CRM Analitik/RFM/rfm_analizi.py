##########################################################
# RFM ile Müşteri Segmentasyonu ( Customer Segmentation with RFM )
##########################################################

# 1. İş Problemi ( Business Problem )
# 2. Veriyi Anlama ( Data Understanding )
# 3. Veri Hazırlama ( Data Preparation )
# 4. RFM Metriklerinin Hesaplanması ( Calculating RFM Metrics )
# 5. RFM Skorlarının Hesaplanması ( Calculating RFM Scores )
# 6. RFM Segmentlerinin Oluşturulması ve Analiz Edilmesi ( Creating & Analysing RFM Segments )
# 7. Tüm Sürecin Fonksiyonlaştırılması


###########################################
# 1. İş Problemi ( Business Problem )
###########################################

# Bir e-ticaret şirketi müşterilerini segmentlere ayırıp bu segmentlere göre pazarlama stratejileri belirlemek istiyor.

# Veri Seti Hikayesi
# https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

# Online Retail II isimli veri seti İngiltere merkezli online bir satış mağazasının
# 01/12/2009 - 09/12/2011 tarihleri arasındaki satışlarını içeriyor.

# Değişkenler
#
# InvoiceNo: Fatura numarası. Her işleme yani faturaya ait eşsiz numara. C ile başlıyorsa iptal edilen işlem.
# StockCode: Ürün kodu. Her bir ürün için eşsiz numara.
# Description: Ürün ismi
# Quantity: ürün adedi. Faturalardaki ürünlerden kaçar tane satıldığını ifade etmektedir.
# InvoiceDate: Fatura tarihi ve zamanı.
# UnitPrice: Ürün fiyatı (Sterlin cinsinden)
# CustomerID: Eşsiz müşteri numarası
# Country: Ülke ismi. Müşterinin yaşadığı ülke.

###########################################
# 2. Veriyi Anlama ( Data Understanding )
###########################################

import datetime as dt
import pandas as pd

from Data_Structures.data_structures import names

# Tüm sütunları göster, sayıları 3 ondalıkla yazdır
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Excel dosyasını oku, Year 2009-2010 sayfasını al
df_ = pd.read_excel("C:/Users/ONI/PycharmProjects/PythonTopics/Datasets/RFM_VeriSeti/online_retail_II.xlsx",
                    sheet_name='Year 2009-2010')
df = df_.copy()  # Orijinal dataframe'i korumak için kopya oluştur

df.head()  # İlk 5 satırı görüntüle
df.shape  # Satır ve sütun sayısını göster
df.isnull().sum()  # Her sütunda eksik değer sayısını göster

# Eşsiz ürün sayısı
df["Description"].nunique()

# En çok tekrar eden ürünler
df["Description"].value_counts().head()

# Ürün bazında toplam satış miktarı
df.groupby("Description").agg({"Quantity": "sum"}).head()

# Ürün bazında toplam satış miktarına göre en çok satanlar
df.groupby("Description").agg({"Quantity": "sum"}).sort_values("Quantity", ascending=False).head()

# Fatura numarasının kaç farklı benzersiz değer içerdiğini sayıyoruz
df["Invoice"].nunique()

# Her bir satır için toplam fiyatı hesaplıyoruz (adet * birim fiyat)
df["TotalPrice"] = df["Quantity"] * df["Price"]

# Faturalara göre gruplama yapıp, her bir faturadaki toplam fiyatı hesaplıyoruz
# 'agg' ile TotalPrice sütununu topluyoruz
df.groupby("Invoice").agg({"TotalPrice": "sum"}).head()  # İlk 5 faturayı gösteriyoruz


###########################################
# 3. Veri Hazırlama ( Data Preparation )
###########################################

# Veri setinin boyutunu (satır sayısı, sütun sayısı) gösterir
df.shape

# Her bir sütunda kaç eksik (NaN) değer olduğunu sayar
df.isnull().sum()

# Eksik (NaN) değer içeren satırları veri setinden kalıcı olarak siler
df.dropna(inplace=True)

# --------------------------
# Sayısal sütunlar için temel istatistikleri özetler:
# count  : Veri sayısı
# mean   : Ortalama
# std    : Standart sapma
# min    : Minimum değer
# 25%,50%,75% : Çeyrek değerler
# max    : Maksimum değer
# .T ile transpoze ederek satır ve sütunları yer değiştirip okunabilirliği artırıyoruz
df.describe().T

# --------------------------
# Fatura numarası sütununda 'C' harfi içermeyen satırları seçiyoruz
# 'C' genellikle iade faturalarını temsil eder, onları filtreliyoruz
# ~ operatörü ile koşulun tersini alıyoruz
# na=False ile eksik değerleri otomatik olarak False sayıyoruz
df = df[~df["Invoice"].str.contains("C", na=False)]


###########################################
# 4. RFM Metriklerinin Hesaplanması ( Calculating RFM Metrics )
###########################################

# Recency, Frequency, Monetary

# Recency
# Müşterinin yeniliğini, sıcaklığını ifade ediyor.
# Bunun matematiksel karşılığı şudur, analizin yapıldığı tarih - ilgili müşterinin son satım almayı yaptığı tarihtir.

# Frequency
# Müşterinin yaptığı toplam satın almadır.

# Monetary
# Müşterinin yaptığı toplam satın almalar neticesinde bıraktığı toplam parasal değerdir.

df.head()

# Referans tarih olarak 11 Aralık 2010 tarihini belirliyoruz
today_date = dt.datetime(2010, 12, 11)

# Müşteri bazında RFM analizi için gruplama işlemi yapıyoruz
rfm = df.groupby("Customer ID").agg({
    # Recency (yenilik) metriği: her müşterinin son alışverişinden bu yana geçen gün sayısı
    "InvoiceDate": lambda InvoiceDate: (today_date - InvoiceDate.max()).days,

    # Frequency (alışveriş sıklığı) metriği: her müşterinin yaptığı benzersiz fatura sayısı
    "Invoice": lambda Invoice: Invoice.nunique(),

    # Monetary (harcama miktarı) metriği: her müşterinin toplam harcama tutarı
    "TotalPrice": lambda TotalPrice: TotalPrice.sum()
})

# İlk birkaç satırı görüntüleyerek veriye genel bir bakış yapıyoruz
rfm.head()

# Sütun adlarını RFM analizine uygun hale getiriyoruz
rfm.columns = ["recency", "frequency", "monetary"]

# Sayısal sütunlar için temel istatistikleri özetliyoruz (count, mean, std, min, max vb.)
# .T ifadesi ile satır ve sütunları transpoze ederek çıktıyı daha okunabilir hale getiriyoruz
rfm.describe().T

# Harcama tutarı (monetary) 0 veya negatif olan müşterileri analiz dışı bırakıyoruz
rfm = rfm[rfm["monetary"] > 0]

# Veri setinin boyutunu (satır, sütun) görüntülüyoruz
rfm.shape


###########################################
# 5. RFM Skorlarının Hesaplanması ( Calculating RFM Scores )
###########################################

# 'recency' değerlerini 5 eşit parçaya bölerek her müşteriye bir skor atıyoruz
# En düşük recency (yani en güncel alışveriş) değeri 5, en yüksek (yani en eski) değer 1 puan alır
rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])

# 'frequency' (alışveriş sıklığı) değerlerini 5 eşit gruba bölerek her müşteriye skor veriyoruz
# Önce rank(method="first") ile aynı frekansa sahip müşterilere sıralı (tekrarsız) bir sıra numarası veriyoruz
# Bu, qcut fonksiyonunun "eşit olmayan" tekrar değerlerinde hata vermesini engeller
# Daha sonra pd.qcut() ile veriyi 5 eşit parçaya böleriz
# En sık alışveriş yapan müşteriye 5, en az alışveriş yapan müşteriye 1 puan verilir
rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1])


# 'monetary' (toplam harcama) değerlerini 5 eşit gruba ayırarak her müşteriye bir skor veriyoruz
# En az harcama yapan müşteriye 1, en çok harcama yapan müşteriye 5 puan verilir
rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5])

# 'recency_score' ve 'frequency_score' değerlerini stringe çevirip birleştiriyoruz
# Böylece her müşteriye 2 basamaklı bir RFM skoru atanır (örneğin: "55", "42" gibi)
rfm["RFM_SCORE"] = (rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str))

# RFM tablosundaki sayısal sütunların istatistiksel özetini görüntülüyoruz
rfm.describe().T

# RFM skoru "55" olan müşterileri listeliyoruz
# Bu grup, en güncel (recency=5) ve en sık alışveriş yapan (frequency=5) müşterilerdir → "Sadık Müşteriler"
rfm[rfm["RFM_SCORE"] == "55"]

# RFM skoru "11" olan müşterileri listeliyoruz
# Bu grup, en uzun süredir alışveriş yapmayan (recency=1) ve en az alışveriş yapan (frequency=1) müşterilerdir → "Kaybedilmiş Müşteriler"
rfm[rfm["RFM_SCORE"] == "11"]


###########################################
# 6. RFM Segmentlerinin Oluşturulması ve Analiz Edilmesi ( Creating & Analysing RFM Segments )
###########################################

# 🔹 Regex (Regular Expression - Düzenli İfade):
# Metinlerde belirli desenleri (pattern) bulmak, eşleştirmek veya filtrelemek için kullanılan bir kurallar bütünüdür.

# Önemli Regex Sembolleri:
# .       → Herhangi bir karakter (tek karakter) ile eşleşir
# ^       → Metnin başlangıcını belirtir
# $       → Metnin sonunu belirtir
# []      → Köşeli parantez içindeki karakterlerden biriyle eşleşir (örnek: [abc] → a, b veya c)
# [0-9]   → 0 ile 9 arasındaki herhangi bir rakamla eşleşir
# [a-z]   → a’dan z’ye herhangi bir küçük harfle eşleşir
# [A-Z]   → A’dan Z’ye herhangi bir büyük harfle eşleşir
# |       → "veya" anlamına gelir (örnek: cat|dog → 'cat' veya 'dog')
# *       → Önceki karakterin 0 veya daha fazla tekrarını eşleştirir
# +       → Önceki karakterin 1 veya daha fazla tekrarını eşleştirir
# ?       → Önceki karakterin 0 veya 1 kez geçmesini sağlar
# {n}     → Önceki karakterin tam olarak n kez tekrar etmesini ister
# {n,m}   → Önceki karakterin en az n, en fazla m kez geçmesini ister
# ()      → Gruplama yapmak için kullanılır
# \d      → Herhangi bir rakam (0–9)
# \D      → Rakam olmayan karakter
# \s      → Boşluk karakteri (space, tab vb.)
# \S      → Boşluk olmayan karakter
# \w      → Harf, rakam veya alt çizgi (_)
# \W      → Harf, rakam veya alt çizgi olmayan karakter

# r"..."  → Raw string (ham string) ifadesi; ters eğik çizgilerin (\) özel anlamını kaldırır,
#           böylece regex ifadeleri doğrudan yazılabilir (örnek: r"\d+" yerine "\\d+")


seg_map = {
    r'[1-2][1-2]': 'hipernating',
    r'[1-2][3-4]': 'at_risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

# 'RFM_SCORE' sütunundaki değerleri, regex (düzenli ifade) tabanlı eşleştirme sözlüğü 'seg_map' yardımıyla
# ilgili müşteri segmenti isimleriyle değiştiriyoruz (örnek: "55" → "Champions", "11" → "Lost Customers")
rfm["segment"] = rfm["RFM_SCORE"].replace(seg_map, regex=True)

# Müşteri segmentlerine göre ortalama (mean) ve müşteri sayısı (count) bilgilerini özetliyoruz
# ["segment", "recency", "frequency", "monetary"] sütunlarını seçerken köşeli parantez içinde liste olarak yazmalıyız
rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])

# 'cant_loose' segmentine ait ilk birkaç müşteriyi görüntülüyoruz
rfm[rfm["segment"] == "cant_loose"].head()

# 'cant_loose' segmentine ait müşterilerin indeks (Customer ID) değerlerini getiriyoruz
# Bu sayede hangi müşterilerin bu segmente ait olduğunu kolayca görebiliriz
rfm[rfm["segment"] == "cant_loose"].index

# Boş bir DataFrame oluşturuyoruz
new_df = pd.DataFrame()

# 'new_customers' segmentinde yer alan müşterilerin indekslerini (Customer ID) yeni bir sütuna aktarıyoruz
new_df["new_customer_id"] = rfm[rfm["segment"] == "new_customers"].index

# Müşteri ID’lerini tam sayı (int) veri tipine dönüştürüyoruz
# (Bazı durumlarda indeksler float veya string olabilir)
new_df["new_customer_id"] = new_df["new_customer_id"].astype(int)

# Oluşturulan müşteri listesini 'new_customers.csv' adlı dosyaya kaydediyoruz
# Böylece yeni müşteri segmenti dışa aktarılmış olur
new_df.to_csv("new_customers.csv")


###########################################
# 7. Tüm Sürecin Fonksiyonlaştırılması
###########################################

def create_rfm(dataframe, csv=False):

    # VERİYİ HAZIRLAMA
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]

    # RFM METRİKLERİNİN HESAPLANMASI
    today_date = dt.datetime(2011, 12, 11)
    rfm = dataframe.groupby("Customer ID").agg({"InvoiceDate": lambda date: (today_date - date.max()).days,
                                                "Invoice": lambda num: num.nunique(),
                                                "TotalPrice": lambda price: price.sum()})

    rfm.columns = ["recency", "frequency", "monetary"]
    rfm = rfm[(rfm["monetary"] > 0)]

    # cltv_df skorları kategorik değere dönüştürülüp df'e eklendi
    rfm["RFM_SCORE"] = (rfm["recency_score"].astype(str) +
                        rfm["frequency_score"].astype(str))


    # SEGMENTLERİN İSİNLENDİRİLMESİ
    seg_map = {
        r'[1-2][1-2]': 'hipernating',
        r'[1-2][3-4]': 'at_risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }

    rfm["segment"] = rfm["RFM_SCORE"].replace(seg_map, regex=True)
    rfm = rfm[["recency", "frequency", "monetary", "segment"]]
    rfm.index = rfm.index.astype(int)

    if csv:
        rfm.to_csv("rfm.csv")
    return rfm

df = df_.copy()

rfm_new = create_rfm(df, csv=True)
