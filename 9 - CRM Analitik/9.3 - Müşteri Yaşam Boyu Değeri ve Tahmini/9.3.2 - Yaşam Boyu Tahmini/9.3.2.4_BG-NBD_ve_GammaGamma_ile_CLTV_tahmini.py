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


##################################################
# 2. BG-NBD Modelinin Kurulması
##################################################

# BG-NBD (Beta Geometric / Negative Binomial Distribution) modeli,
# bir müşterinin gelecekte kaç defa satın alma yapacağını tahmin etmek için kullanılır.

# penalizer_coef: Aşırı öğrenmeyi (overfitting) önlemek için katsayılara uygulanan bir ceza terimidir.
# Çok küçük bir değer (örneğin 0.001), modelin esnekliğini azaltmadan daha dengeli bir öğrenme sağlar.
bgf = BetaGeoFitter(penalizer_coef=0.001)

# Modeli eğitiyoruz.
# frequency: Müşterinin geçmişte kaç kez satın alma yaptığı
# recency: Müşterinin son satın alması ile ilk satın alması arasındaki süre (haftalık)
# T: Analiz tarihine kadar müşterinin toplam yaşı (haftalık)
bgf.fit(cltv_df["frequency"],
        cltv_df["recency"],
        cltv_df["T"])


######################################################################
# 1 hafta içinde en çok satın alma yapması beklenen 10 müşteri
######################################################################

# conditional_expected_number_of_purchases_up_to_time() fonksiyonu,
# belirli bir zaman aralığında (örneğin 1 hafta) her müşterinin beklenen satın alma sayısını tahmin eder.
# Burada zaman birimi, daha önce recency ve T'yi haftalık olarak dönüştürdüğümüz için "hafta" anlamına gelir.
# Sonuçlar azalan sırada sıralanır ve en yüksek 10 müşteri listelenir.
bgf.conditional_expected_number_of_purchases_up_to_time(
    1,  # tahmin süresi: 1 hafta
    cltv_df["frequency"],
    cltv_df["recency"],
    cltv_df["T"]
).sort_values(ascending=False).head(10)

# Bu fonksiyonun daha kısası yok mu peki ? var. Bu işlemin aynısını predict() fonksiyonuyla da yapabiliriz.

bgf.predict(1,  # tahmin süresi: 1 hafta
            cltv_df["frequency"],
            cltv_df["recency"],
            cltv_df["T"]).sort_values(ascending=False).head(10)
# BG/NBD modeli için bu metot geçerlidir fakat Gamma Gamma modeli için geçerli değildir.

cltv_df["expected_purc_1_week"] = bgf.predict(1,  # tahmin süresi: 1 hafta
                                              cltv_df["frequency"],
                                              cltv_df["recency"],
                                              cltv_df["T"])


######################################################################
# 1 ay içinde en çok satın alma yapması beklenen 10 müşteri
######################################################################

bgf.predict(4,  # tahmin süresi: 4 hafta
            cltv_df["frequency"],
            cltv_df["recency"],
            cltv_df["T"]).sort_values(ascending=False).head(10)

cltv_df["expected_purc_1_month"] = bgf.predict(4,  # tahmin süresi: 4 hafta
                                              cltv_df["frequency"],
                                              cltv_df["recency"],
                                              cltv_df["T"])

# 1 aylık periyod da şirketimizin beklediği satış sayısı

cltv_df["expected_purc_1_month"] = bgf.predict(4,  # tahmin süresi: 4 hafta
                                              cltv_df["frequency"],
                                              cltv_df["recency"],
                                              cltv_df["T"]).sum()

#############################################
# Tahmin Sonuçlarının Değerlendirilmesi
#############################################

# BG-NBD modelinin tahmin performansını görselleştirmek için plot_period_transactions kullanıyoruz.
# Bu grafik, modelin geçmiş verilere ne kadar uyduğunu gösterir.
plot_period_transactions(bgf)
plt.show()  # Grafiği ekranda göster


#############################################
# 3. GAMMA-GAMMA Modelinin Kurulması
#############################################

# Hatırlatma:
# - BG/NBD modeli müşterinin satın alma sayısını tahmin eder.
# - Gamma-Gamma modeli ise müşterinin ortalama kârını (monetary value) tahmin eder.

# Gamma-Gamma modelini oluşturuyoruz
# penalizer_coef: overfitting'i önlemek için ceza katsayısı
ggf = GammaGammaFitter(penalizer_coef=0.01)

# Modeli eğitiyoruz
# frequency: tekrarlayan satın alma sayısı
# monetary: satın alma başına ortalama kazanç
ggf.fit(cltv_df["frequency"], cltv_df["monetary"])

# İlk 10 müşterinin beklenen ortalama kârını tahmin et
ggf.conditional_expected_average_profit(cltv_df["frequency"], cltv_df["monetary"]).head(10)

# Tahminleri azalan sırada sıralayarak en yüksek 10 müşteriyi görüntüle
ggf.conditional_expected_average_profit(cltv_df["frequency"], cltv_df["monetary"]).sort_values(ascending=False).head(10)

# Beklenen ortalama kâr değerlerini veri setine ekle
cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df["frequency"], cltv_df["monetary"])

# En yüksek beklenen ortalama kâr değerine sahip ilk 10 müşteriyi göster
cltv_df.sort_values("expected_average_profit", ascending=False).head(10)

##################################################
# 4. BG-NBD ve Gamma-Gamma modeli ile CLTV'nin hesaplanması
##################################################

# Customer Lifetime Value (CLTV) hesaplaması
# Burada BG-NBD modeli ile satın alma sayısını, Gamma-Gamma modeli ile ortalama kazancı
# kullanarak her müşteri için gelecekteki 3 aylık CLTV tahminini yapıyoruz.

cltv = ggf.customer_lifetime_value(
    bgf,  # BG-NBD modeli
    cltv_df["frequency"],  # Geçmişteki satın alma sayısı
    cltv_df["recency"],    # Son satın alma üzerinden geçen süre (haftalık)
    cltv_df["T"],          # Müşteri yaşı (haftalık)
    cltv_df["monetary"],   # Ortalama kazanç
    time=3,                # Tahmin süresi: 3 ay
    freq="W",              # Zaman birimi: hafta
    discount_rate=0.01     # Gelecekteki nakit akışlarının bugünkü değerini indirgeme oranı
)

# Hesaplanan CLTV verisinin ilk 5 satırını görüntüle
cltv.head()

# CLTV veri çerçevesinin index'ini resetle
cltv = cltv.reset_index()

# Orijinal müşteri verisi ile CLTV değerlerini birleştir
# Böylece hem müşteri bilgileri hem de CLTV tek tabloda olur
cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")

# En yüksek CLTV değerine sahip ilk 10 müşteriyi sıralayarak göster
cltv_final.sort_values(by="clv", ascending=False).head(10)


##################################################
# 5. CLTV'ye Göre Segmentlerin Oluşturulması
##################################################

# cltv_final veri çerçevesi, her müşterinin bilgilerini ve tahmini CLTV değerlerini içerir
cltv_final

# CLTV değerlerini ölçekleyip (scaled_clv) 4 segmente ayırıyoruz
# pd.qcut: Veriyi eşit büyüklükteki 4 gruba böler
# Labels: En düşükten en yükseğe segmentleri D, C, B, A olarak etiketliyoruz
cltv_final["segment"] = pd.qcut(cltv_final["scaled_clv"], 4, labels=["D", "C", "B", "A"])

# En yüksek CLTV değerine sahip ilk 50 müşteriyi görüntüle
cltv_final.sort_values(by="clv", ascending=False).head(50)

# Segmentlere göre özet istatistikler
# count: segmentte kaç müşteri var
# mean: segmentin ortalama CLTV değeri
# sum: segmentin toplam CLTV değeri
cltv_final.groupby("segment").agg({"count", "mean", "sum"})


##################################################
# 6. Çalışmanın Fonksiyonlaştırılması
##################################################

def create_cltv_p(dataframe, month=3):
    """
    Bu fonksiyon, verilen e-ticaret veri setinden müşteri yaşam boyu değerini (CLTV) tahmin eder.
    BG-NBD ve Gamma-Gamma modellerini kullanarak:
    - Beklenen satın alma sayısını
    - Beklenen ortalama kazancı
    - CLTV değerini
    - CLTV segmentlerini
    hesaplar ve tek bir veri çerçevesi olarak döndürür.
    """

    # 1. Veri Ön İşleme
    dataframe.dropna(inplace=True)  # Eksik verileri çıkar
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]  # İptal faturalar çıkar
    dataframe = dataframe[dataframe["Quantity"] > 0]  # Negatif miktarları çıkar
    dataframe = dataframe[dataframe["Price"] > 0]  # Fiyatı 0 olanları çıkar

    # Aykırı değerleri kırp
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")

    # Toplam fiyat değişkeni oluştur
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]

    today_date = dt.datetime(2011, 12, 11)  # Analiz tarihi

    # Müşteri bazında özet veri oluşturma
    cltv_df = dataframe.groupby("Customer ID").agg({
        "InvoiceDate": [
            lambda InvoiceDate: (InvoiceDate.max() - InvoiceDate.min()).days,  # recency
            lambda InvoiceDate: (today_date - InvoiceDate.min()).days         # T
        ],
        "Invoice": lambda num: num.nunique(),  # frequency
        "total_price": lambda TotalPrice: TotalPrice.sum()  # toplam kazanç
    })

    # Çoklu index sütunlarını düzleştir
    cltv_df.columns = cltv_df.columns.droplevel(0)
    cltv_df.columns = ["recency", "T", "frequency", "monetary"]

    # Ortalama kazancı satın alma başına çevir
    cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]

    # Tek seferlik alışveriş yapanları çıkar
    cltv_df = cltv_df[cltv_df["frequency"] > 1]

    # Zaman ölçümlerini haftalık birime çevir
    cltv_df["recency"] = cltv_df["recency"] / 7
    cltv_df["T"] = cltv_df["T"] / 7

    # 2. BG-NBD Modelinin Kurulması
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(cltv_df["frequency"],
            cltv_df["recency"],
            cltv_df["T"])

    # Beklenen satın alma sayısını tahmin et
    cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                                  cltv_df["frequency"],
                                                  cltv_df["recency"],
                                                  cltv_df["T"])

    cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                                   cltv_df["frequency"],
                                                   cltv_df["recency"],
                                                   cltv_df["T"])

    cltv_df["expected_purc_3_month"] = bgf.predict(12,
                                                   cltv_df["frequency"],
                                                   cltv_df["recency"],
                                                   cltv_df["T"])

    # 3. GAMMA-GAMMA Modelinin Kurulması
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(cltv_df["frequency"], cltv_df["monetary"])

    # Beklenen ortalama kazancı tahmin et
    cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(
        cltv_df["frequency"], cltv_df["monetary"]
    )

    # 4. CLTV'nin hesaplanması
    cltv = ggf.customer_lifetime_value(
        bgf,  # BG-NBD modeli
        cltv_df["frequency"],
        cltv_df["recency"],
        cltv_df["T"],
        cltv_df["monetary"],
        time=month,  # Tahmin süresi (ay)
        freq="W",    # Zaman birimi: hafta
        discount_rate=0.01
    )

    # CLTV veri çerçevesinin index'ini resetle
    cltv = cltv.reset_index()

    # Orijinal müşteri verisi ile CLTV değerlerini birleştir
    cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")

    # CLTV değerlerini ölçekleyip 4 segmente ayır
    cltv_final["segment"] = pd.qcut(cltv_final["scaled_clv"], 4, labels=["D", "C", "B", "A"])

    return cltv_final


# Fonksiyonun çalıştırılması
df = df_.copy()
cltv_final2 = create_cltv_p(df)

# Tahmin sonuçlarını CSV olarak kaydet
cltv_final2.to_csv("cltv_prediction.csv")
