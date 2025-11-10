##################################################
# BG-NBD ve Gamma-Gamma ile CLTV Prediction
##################################################

# Bu çalışma, bir e-ticaret şirketi için Müşteri Yaşam Boyu Değerini (CLTV) tahmin etmeyi amaçlar.
# BG-NBD modeli ile müşterilerin gelecekteki satın alma sayısını,
# Gamma-Gamma modeli ile müşterilerin ortalama kârını tahmin edeceğiz.
# Daha sonra bu bilgiler ile CLTV hesaplanacak ve müşteriler segmentlere ayrılacaktır.

##################################################
# 1. Verinin Hazırlanması (Data Preperation)
##################################################

# Veri seti, İngiltere merkezli online bir satış mağazasının 01/12/2009 - 09/12/2011 tarihleri arasındaki satışlarını içeriyor.
# Değişkenler:
# - InvoiceNo: Fatura numarası. "C" ile başlıyorsa iptal edilen işlem
# - StockCode: Ürün kodu
# - Description: Ürün adı
# - Quantity: Ürün adedi
# - InvoiceDate: Fatura tarihi ve zamanı
# - UnitPrice: Ürün fiyatı
# - CustomerID: Müşteri numarası
# - Country: Ülke ismi

##################################
# Gerekli Kütüphane ve Fonksiyonlar
##################################

# pip install lifetimes (Terminal)
# !pip install lifetimes (Python Console)

import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler

# Pandas ayarları: Tüm sütunlar gösterilsin, satırlar genişletilsin, float formatı 4 basamaklı
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)


#########################
# Aykırı Değer Fonksiyonları
#########################

def outlier_thresholds(dataframe, variable):
    # 1. ve 99. yüzdelikleri kullanarak aykırı değer sınırlarını hesaplar
    quartile1 = dataframe[variable].quantile(0.01)  # Alt sınır için 1. yüzdelik
    quartile3 = dataframe[variable].quantile(0.99)  # Üst sınır için 99. yüzdelik
    interquantile_range = quartile3 - quartile1     # IQR: çeyrekler arası mesafe
    up_limit = quartile3 + 1.5 * interquantile_range  # Üst sınır
    low_limit = quartile1 - 1.5 * interquantile_range # Alt sınır
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    # Aykırı değerleri sınırlar ile değiştirir (kırpar)
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit  # Alt sınırdan düşük değerler kırpılır
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit    # Üst sınırdan yüksek değerler kırpılır


# Aykırı değerler, değişkenin genel dağılımının oldukça dışında kalan gözlemlerdir.
# Bunlar veri seti analizini yanıltabilir. Silmek yerine baskılayarak sınır değerlerine eşitlemek daha güvenli bir yaklaşımdır.


#########################
# Verinin Okunması
#########################

# Excel dosyasından veri okuma
df_ = pd.read_excel("Datasets/Müşteri_Yaşam_Boyu_Değeri_ve_Tahmini_Dataset/online_retail_II.xlsx",
                    sheet_name='Year 2010-2011')
df = df_.copy()  # Orijinal veri yedeği

# Veri setinin özet istatistikleri
df.describe().T
# Eksik değer sayısı kontrolü
df.isnull().sum()


#########################
# Veri Ön İşleme
#########################

df.dropna(inplace=True)  # Eksik verileri kaldır

# İptal edilen faturaları çıkar
df = df[~df["Invoice"].str.contains("C", na=False)]

# Negatif miktarlı ürünleri kaldır
df = df[df["Quantity"] > 0]

# Fiyatı sıfırdan büyük olan ürünleri seç
df = df[df["Price"] > 0]

# Aykırı değerleri kırp
replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

# Toplam fiyat değişkeni oluştur
df["total_price"] = df["Quantity"] * df["Price"]

# Analiz tarihi (bugünkü tarih yerine sabit tarih)
today_date = dt.datetime(2011, 12, 11)


###############################
# Lifetime Veri Yapısının Hazırlanması
###############################

# Müşteri bazında özet veri oluşturma
# - recency: son satın alma üzerinden geçen süre (gün cinsinden)
# - T: müşterinin yaşı, ilk satın almadan bugüne geçen süre
# - frequency: tekrarlayan satın alma sayısı (frequency > 1)
# - monetary: satın alma başına ortalama kazanç

cltv_df = df.groupby("Customer ID").agg({
    "InvoiceDate": [
        lambda InvoiceDate: (InvoiceDate.max() - InvoiceDate.min()).days,  # recency
        lambda InvoiceDate: (today_date - InvoiceDate.min()).days         # T (müşteri yaşı)
    ],
    "Invoice": lambda num: num.nunique(),  # frequency
    "total_price": lambda TotalPrice: TotalPrice.sum()  # toplam kazanç
})

# Çoklu index sütunları tek seviyeye indir
cltv_df.columns = cltv_df.columns.droplevel(0)
# Sütun isimlerini anlamlı şekilde değiştir
cltv_df.columns = ["recency", "T", "frequency", "monetary"]

# Monetary değerini satın alma başına ortalamaya çevir
cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]

# Özet istatistikler
cltv_df.describe().T

# Tek seferlik alışveriş yapan müşterileri çıkar
# CLTV modellemesi, tekrarlayan satın alma davranışını tahmin etmek için yapılır
cltv_df = cltv_df[(cltv_df["frequency"] > 1)]

# Recency ve T değerlerini haftalık hale çeviriyoruz
# Lifetimes kütüphanesi, zaman ölçümlerini genellikle haftalık kullanır
# recency: müşterinin son satın alma üzerinden geçen süre (hafta cinsinden)
cltv_df["recency"] = cltv_df["recency"] / 7

# T: müşterinin yaşı (analiz tarihinden ilk satın alma tarihine kadar geçen süre)
# Haftalık birim ile BG-NBD modeline uygun hale getirilir
cltv_df["T"] = cltv_df["T"] / 7
