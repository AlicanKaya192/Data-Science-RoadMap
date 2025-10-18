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
df_ = pd.read_excel("C:/Users/ONI/PycharmProjects/PythonTopics/DataSets/RFM_VeriSeti/online_retail_II.xlsx",
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











