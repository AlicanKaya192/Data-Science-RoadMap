################################
# Rating Products (Ürün Puanlama)
################################
# Bir ürünün veya hizmetin "gerçek" değerini nasıl hesaplarız?
# Sadece aritmetik ortalama almak yeterli midir? Yoksa yeni yorumlar eski yorumlardan
# veya kursu bitiren kişinin yorumu, yeni başlayandan daha mı değerlidir?

# Bu bölümde şu yöntemleri inceleyeceğiz:
# 1. Average (Basit Ortalama)
# 2. Time-Based Weighted Average (Zaman Ağırlıklı Ortalama)
# 3. User-Based Weighted Average (Kullanıcı Ağırlıklı Ortalama)
# 4. Weighted Rating (Karma Ağırlıklı Puanlama)


################################
# Uygulama: Kullanıcı ve Zaman Ağırlıklı Kurs Puanı Hesaplama
################################

import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

# Çıktıların daha okunaklı olması için pandas ayarlarını yapıyoruz.
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


# (50+ Saat) Python A-Z: Veri Bilimi ve Makine Öğrenimi Kursu Verisi
# Veri setini yüklüyoruz.
df = pd.read_csv("Datasets ( Genel )/course_reviews.csv")
df.head()
df.shape

# --- VERİYİ TANIMA ---
# Puanların (Rating) dağılımına bakıyoruz. İnsanlar genelde kaç puan vermiş?
df["Rating"].value_counts()

# Sorulan soru sayılarına bakıyoruz.
df['Questions Asked'].value_counts()

# Soru soranların verdiği puan ortalaması ile sormayanlarınki farklı mı?
# (Burada soru soranların daha ilgili olduğu ve puanlarının daha bilinçli olduğu varsayılabilir)
df.groupby('Questions Asked').agg({'Questions Asked': 'count',
                                     'Rating': 'mean'})

df.head()

################
# 1. Average (Basit Ortalama)
################
# En ilkel yöntemdir. Tüm puanları toplayıp kişi sayısına böleriz.
# Dezavantajı: 3 yıl önceki 1 puan ile dünkü 5 puanı eşit ağırlıkta tutar.
# Ürünün güncel kalitesini yansıtmayabilir.

df["Rating"].mean()


################
# 2. Time-Based Weighted Average (Zaman Ağırlıklı Ortalama)
################
# AMAÇ: Yakın zamanda yapılan yorumlara daha fazla ağırlık vermek.
# Neden? Çünkü ürün (kurs) güncellenmiş olabilir, eğitmen sorulara daha hızlı cevap veriyor olabilir.
# Güncel trendi yakalamak için yeni yorumlar daha değerlidir.

df.head()
df.info()

# Timestamp sütununu tarih formatına çeviriyoruz.
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Analiz yaptığımız günü sabitliyoruz (Veri setinin son tarihi gibi düşünebiliriz).
current_date = pd.to_datetime('2021-02-10 00:00:00')

# Yorumun yapıldığı günden bugüne kaç gün geçmiş? ('days' değişkeni)
df['days'] = (current_date - df['Timestamp']).dt.days

# --- ZAMAN DİLİMLERİNE GÖRE ORTALAMALAR ---
# Son 30 gün içinde gelen yorumların ortalaması (En güncel ve en değerli)
df.loc[df['days'] <= 30, 'Rating'].mean()

# 30-90 gün arası yorumların ortalaması
df.loc[(df['days'] > 30) & (df['days'] <= 90), 'Rating'].mean()

# 90-180 gün arası yorumların ortalaması
df.loc[(df['days'] > 90) & (df['days'] <= 180), 'Rating'].mean()

# 180 günden eski yorumların ortalaması (En az değerli)
df.loc[(df['days'] > 180), 'Rating'].mean()


# --- MANUEL AĞIRLIKLANDIRMA ---
# Yakın tarihe daha yüksek katsayı (ağırlık) veriyoruz.
# Örn: Son 30 güne %28, eskilere doğru giderek azalan ağırlıklar (%26, %24, %22).
df.loc[df['days'] <= 30, 'Rating'].mean() * 28 / 100 + \
df.loc[(df['days'] > 30) & (df['days'] <= 90), 'Rating'].mean() * 26 / 100 + \
df.loc[(df['days'] > 90) & (df['days'] <= 180), 'Rating'].mean() * 24 / 100 + \
df.loc[(df['days'] > 180), 'Rating'].mean() * 22 / 100


# --- FONKSİYONLAŞTIRMA ---
# Bu işlemi tekrar tekrar kullanabilmek için fonksiyon haline getiriyoruz.
# w1, w2, w3, w4: Zaman dilimlerine verilecek ağırlıklar.
def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[dataframe['days'] <= 30, 'Rating'].mean() * w1 / 100 + \
           dataframe.loc[(dataframe['days'] > 30) & (dataframe['days'] <= 90), 'Rating'].mean() * w2 / 100 + \
           dataframe.loc[(dataframe['days'] > 90) & (dataframe['days'] <= 180), 'Rating'].mean() * w3 / 100 + \
           dataframe.loc[(dataframe['days'] > 180), 'Rating'].mean() * w4 / 100


# Varsayılan ağırlıklarla hesaplama
time_based_weighted_average(df)

# Farklı ağırlık senaryosu (Yakın zamana daha da çok önem verelim: %30)
time_based_weighted_average(df, 30, 26, 22, 22)


################
# 3. User-Based Weighted Average (Kullanıcı Ağırlıklı Ortalama)
################
# AMAÇ: Ürünü/Hizmeti daha çok deneyimleyen kullanıcının puanına daha çok önem vermek.
# Senaryo: Kursun sadece %1'ini izleyip 1 puan veren ile %100'ünü izleyip 5 puan verenin
# değerlendirmesi aynı ağırlıkta olmamalıdır. Kursu bitiren kişi içeriğe daha hakimdir.

df.head()

# İlerleme durumuna (Progress) göre ortalama puanlara bakıyoruz.
df.groupby('Progress')['Rating'].mean()


# --- İLERLEME DURUMUNA GÖRE AĞIRLIKLANDIRMA ---
# Kursu az izleyenlerin (<=10) puanına daha az ağırlık (%22),
# Kursu çok izleyenlerin (>75) puanına daha çok ağırlık (%28) veriyoruz.

df.loc[df['Progress'] <= 10, 'Rating'].mean() * 22 / 100 + \
df.loc[(df['Progress'] > 10) & (df['Progress'] <= 45), 'Rating'].mean() * 24 / 100 + \
df.loc[(df['Progress'] > 45) & (df['Progress'] <= 75), 'Rating'].mean() * 26 / 100 + \
df.loc[(df['Progress'] > 75), 'Rating'].mean() * 28 / 100

# Fonksiyonlaştırılmış hali
def user_based_weighted_average(dataframe, w1=22, w2=24, w3=26, w4=28):
    return dataframe.loc[dataframe['Progress'] <= 10, 'Rating'].mean() * w1 / 100 + \
           dataframe.loc[(dataframe['Progress'] > 10) & (dataframe['Progress'] <= 45), 'Rating'].mean() * w2 / 100 + \
           dataframe.loc[(dataframe['Progress'] > 45) & (dataframe['Progress'] <= 75), 'Rating'].mean() * w3 / 100 + \
           dataframe.loc[(dataframe['Progress'] > 75), 'Rating'].mean() * w4 / 100


# Farklı bir senaryo ile hesaplama
user_based_weighted_average(df, 20, 24, 26, 30)


################
# 4. Weighted Rating (Karma Ağırlıklı Puanlama)
################
# FİNAL: Hem zamanı hem de kullanıcı kalitesini hesaba katan hibrid bir yapı.
# Zaman etkisi ve Kullanıcı etkisini birleştirip tek bir skor elde ediyoruz.

def course_weighted_rating(dataframe, time_w=50, user_w=50):
    """
    time_w: Zaman ağırlıklı hesaplamanın genel sonuca etki oranı.
    user_w: Kullanıcı ağırlıklı hesaplamanın genel sonuca etki oranı.
    """
    return time_based_weighted_average(dataframe) * time_w / 100 + user_based_weighted_average(dataframe) * user_w / 100

# Varsayılan değerlerle (%50 Zaman, %50 Kullanıcı) hesaplama
course_weighted_rating(df)

# Farklı bir senaryo: Kullanıcı deneyimine (%60) zamandan (%40) daha çok güveniyoruz.
course_weighted_rating(df, time_w=40, user_w=60)