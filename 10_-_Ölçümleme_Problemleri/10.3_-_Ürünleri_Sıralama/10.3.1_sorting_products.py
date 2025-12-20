################################
# Sorting Products
################################

################################
# Uygulama: Kurs Sıralama
################################

# Gerekli kütüphaneleri import ediyoruz. Veri işleme için pandas, matematiksel işlemler için math,
# istatistiksel hesaplamalar için scipy ve ölçeklendirme için sklearn kullanacağız.
import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

# Pandas görüntüleme ayarlarını yapıyoruz. Tüm sütunları ve satırları görmek istiyoruz,
# ayrıca float sayıların virgülden sonra 5 basamağını görmek istiyoruz.
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# Kurs incelemelerini içeren veri setimizi okuyoruz.
df = pd.read_csv("Datasets ( Genel )/course_reviews.csv")
print(df.shape)
df.head(10)


########################
# Sorting by Rating
########################

# Sadece satın alma sayısına göre sıralama yaparsak ne olur?
# Popüler olanlar öne çıkar ama kalitesi düşük olabilir.
df.sort_values('purchase_count', ascending=False).head(20)

# Sadece yorum sayısına göre sıralama yaparsak?
# Çok yorum alanlar öne çıkar ama yorumlar olumsuz olabilir.
df.sort_values('comment_count', ascending=False).head(20)


########################
# Sorting by Rating, Comment and Purchase
########################

# Derecelendirme, yorum sayısı ve satın alma sayısını bir arada değerlendirerek bir sıralama yapacağız.
# Ancak bu değişkenlerin ölçekleri farklı (biri 1-5 arası, diğeri binlerce).
# Bu yüzden önce MinMaxScaler ile hepsini aynı aralığa (1-5) çekiyoruz.

# Satın alma sayısını 1-5 arasına ölçekliyoruz.
df['purchase_count_sclaed'] = MinMaxScaler(feature_range=(1, 5)). \
fit(df[['purchase_count']]). \
transform(df[['purchase_count']])

df.describe().T

# Yorum sayısını 1-5 arasına ölçekliyoruz.
df['comment_count_scale'] = MinMaxScaler(feature_range=(1, 5)). \
fit(df[['comment_count']]). \
transform(df[['comment_count']])

# Şimdi ağırlıklı bir skor hesaplayabiliriz.
# Yorum sayısı %32, Satın alma %26, Puan %42 etkili olsun diyoruz.
(df['comment_count_scale'] * 32 / 100 +
 df['purchase_count_sclaed'] * 26 / 100 +
 df['Rating'] * 42 / 100)

# Bu işlemi bir fonksiyon haline getiriyoruz ki ağırlıkları kolayca değiştirebilelim.
def weighted_sorting_score(dataframe, w1=32, w2=26, w3=42):
    return (dataframe['comment_count_scale'] * w1 / 100 +
            dataframe['purchase_count_sclaed'] * w2 / 100 +
            dataframe['Rating'] * w3 / 100)

# Hesapladığımız skoru veri setine ekliyoruz.
df['weighted_sorting_score'] = weighted_sorting_score(df)

# Yeni skorumuza göre sıralama yapıyoruz.
df.sort_values('weighted_sorting_score', ascending=False).head(20)

# İçinde "Veri Bilimi" geçen kursları bu skora göre sıralayıp bakıyoruz.
df[df['course_name'].str.contains("Veri Bilimi")].sort_values('weighted_sorting_score', ascending=False).head(20)


########################
# Bayesian Average Rating Score
########################

# Sorting Products with 5 Star Rated
# Sorting Products According to Distribution of 5 Star Ratings

# Puanların ortalamasına bakmak yerine, puanların dağılımına bakarak bir skor üreteceğiz.
# Bu yöntem, az sayıda ama yüksek puan almış ürünlerle, çok sayıda ve yüksek puan almış ürünleri
# daha adil bir şekilde kıyaslamamızı sağlar. Olasılıksal bir yaklaşımdır.

def bayesian_average_rating(dataframe, n, confidence=0.95):
    if sum(n) == 0:
        return 0
    K = len(n)
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    N = sum(n)
    first_part = 0
    second_part = 0
    for k, n_k in enumerate(n):
        first_part += (k + 1) * (n[k] + 1) / (N + K)
        second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
    return score

df.head()

# Her bir kurs için 1, 2, 3, 4 ve 5 puan sayılarını kullanarak BAR skorunu hesaplıyoruz.
df['bar_score'] = df.apply(lambda x: bayesian_average_rating(x[['1_point', 
                                                                '2_point', 
                                                                '3_point', 
                                                                '4_point', 
                                                                '5_point']]), axis=1)

# Ağırlıklı sıralama skoru ile BAR skorunu karşılaştırıyoruz.
df.sort_values('weighted_sorting_score', ascending=False).head(20)
df.sort_values('bar_score', ascending=False).head(20)

# Belirli indekslerdeki kursları BAR skoruna göre inceliyoruz.
df[df['course_name'].index.isin([5, 1])].sort_values('bar_score', ascending=False)


########################
# Hybrid Sorting: BAR Score + Diğer Faktörler
########################

# Rating Products
# - Average
# - Time-Based Weighted Average
# - User-Based Weighted Average
# - Weighted Rating
# - Bayesian Average Rating Score (BAR Score)

# Sorting Products
# - Sorting by Rating
# - Sorting by comment and Purchase Count
# - Sorting by Rating, Comment and Purchase
# - Sorting by Bayesian Average Rating Score (Sorting Products with 5 Star Rated)
# - Hybrid Sorting: BAR Score + Diğer Faktörler

# Hem BAR skorunu (puan dağılımı kalitesi) hem de diğer faktörleri (yorum sayısı, satın alma vb.)
# birleştiren hibrit bir skor oluşturuyoruz.

def hybrid_sorting_score(dataframe, bar_w=60, wss_w=40):
    bar_score = dataframe.apply(lambda x: bayesian_average_rating(x[['1_point',
                                                                     '2_point',
                                                                     '3_point',
                                                                     '4_point',
                                                                     '5_point']]), axis=1)
    wss_score = weighted_sorting_score(dataframe)

    return bar_score * bar_w / 100 + wss_score * wss_w / 100

# Hibrit skoru hesaplayıp veri setine ekliyoruz.
df['hybrid_sorting_score'] = hybrid_sorting_score(df)

# Hibrit skora göre sıralama yapıyoruz.
df.sort_values('hybrid_sorting_score', ascending=False).head(20)

# "Veri Bilimi" kurslarını hibrit skora göre sıralıyoruz.
df[df['course_name'].str.contains("Veri Bilimi")].sort_values('hybrid_sorting_score', ascending=False).head(20)


########################
# Uygulama: IMDB Movie Scoring & Sorting
########################

# Şimdi aynı mantığı IMDB film veri seti üzerinde uygulayalım.
df = pd.read_csv("Datasets ( Genel )/movies_metadata.csv", low_memory=False)

# İlgilendiğimiz sütunları seçiyoruz: Başlık, oy sayısı, oy ortalaması.
df = df[['title', 'vote_count', 'vote_average']]

df.head()
df.shape


########################
# Vote Average'a Göre Sıralama
########################

# Sadece oy ortalamasına göre sıralarsak ne olur?
# Az oy almış ama ortalaması yüksek filmler en üstte çıkar. Bu yanıltıcıdır.
df.sort_values('vote_average', ascending=False).head(20)

# Oy sayılarının dağılımına bakıyoruz.
df['vote_count'].describe([0.10, 0.25, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99]).T

# Belirli bir oy sayısının üzerindeki filmleri (örneğin 400'den fazla) sıralarsak daha mantıklı sonuçlar alırız.
df[df['vote_count'] > 400].sort_values('vote_average', ascending=False).head(20)

# Oy sayısını 1-10 arasına ölçekliyoruz.
df['vote_count_score'] = MinMaxScaler(feature_range=(1, 10)). \
fit(df[['vote_count']]). \
transform(df[['vote_count']])


########################
# vote_average * vote_count
########################

# Basit bir yaklaşım: Oy ortalaması ile oy sayısını (ölçeklenmiş) çarpıyoruz.
# Hem çok izlenen hem de beğenilen filmleri öne çıkarmayı hedefliyoruz.
df['average_count_score'] = df['vote_average'] * df['vote_count_score']

df.sort_values('average_count_score', ascending=False).head(20)


########################
# IMDB Weighted Rating
########################

# weighted_rating = (v / (v + m) * r) + (m / (v + M) * C)

# r = vote average
# v = vote count
# m = minimum votes required to be listed in the Top 250
# C = the mean vote across the whole report (currently 7.0)

# IMDB'nin kullandığı ağırlıklı derecelendirme formülünü uygulayacağız.
# Bu formül, oy sayısı az olan filmlerin ortalamasını genel ortalamaya (C) doğru çeker (shrinkage).
# Oy sayısı arttıkça filmin kendi ortalaması (r) daha baskın hale gelir.

# images/Ölçümleme Problemleri/imdb_formül.png inceleyiniz.

# Film 1:
# r = 8
# m = 500
# v = 1000

# (1000 / (1000 + 500)) * 8 = 5.33


# Film 2:
# r = 8
# m = 500
# v = 3000

# (3000 / (3000 + 500)) * 8 = 6.85

# (1000 / (1000 + 500)) * 9.5


# Film 1:
# r = 8
# m = 500
# v = 1000

# Birinci bölüm:
# (1000 / (1000 + 500)) * 8 = 5.33

# İkinci bölüm:
# (500 / (1000 + 500)) * 7 = 2.33

# Toplam: 5.33 + 2.33 = 7.66


# Film 2:
# r = 8
# m = 500
# v = 3000

# Birinci bölüm:
# (3000 / (3000 + 500)) * 8 = 6.85

# İkinci bölüm:
# (500 / (3000 + 500)) * 7 = 1

# Toplam: 6.85 + 1 = 7.85

# M: Listeye girmek için gereken minimum oy sayısı (örneğin 2500)
# C: Tüm filmlerin ortalama puanı
M = 2500
C = df['vote_average'].mean()

def weighted_rating(r, v, M, C):
    return (v / (v + M) * r) + (M / (v + M) * C)


df.sort_values('average_count_score', ascending=False).head(20)

# Örnek hesaplamalar yapıyoruz.
weighted_rating(7.40000, 11444.00000, M, C)

weighted_rating(8.10000, 14075.00000, M, C)

weighted_rating(8.50000, 8358.00000, M, C)

# Tüm veri seti için ağırlıklı derecelendirmeyi hesaplıyoruz.
df['weighted_rating'] = weighted_rating(df['vote_average'], 
                                        df['vote_count'],  M, C)

df.sort_values('weighted_rating', ascending=False).head(10)


########################
# Bayesian Average Rating Score
########################

# 12481.   The Dark Knight
# 314      The Shawshank Redemption
# 2843.    Fight Club
# 15480.   Inception
# 292.     Pulp Fiction

# IMDB verileri için de BAR skorunu hesaplayabiliriz.
# Ancak burada puanlar 1-10 arasında olduğu için fonksiyonu buna göre uyarlamak veya
# veri setindeki puan dağılımlarını (1 yıldız, 2 yıldız... 10 yıldız) kullanmak gerekir.

def bayesian_average_rating(dataframe, n, confidence=0.95):
    if sum(n) == 0:
        return 0
    K = len(n)
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    N = sum(n)
    first_part = 0
    second_part = 0
    for k, n_k in enumerate(n):
        first_part += (k + 1) * (n[k] + 1) / (N + K)
        second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
    return score

# Örnek filmler için BAR skoru hesaplamaları.
bayesian_average_rating([34733, 4355, 4704, 6561, 13515, 26183, 87368, 273082, 600260, 1295351])

bayesian_average_rating([37128, 5879, 6268, 8419, 16603, 30016, 78538, 199430, 402518, 837905])

# IMDB puan dağılımlarını içeren veri setini okuyoruz.
df = pd.read_csv("Datasets ( Genel )/imdb_ratings.csv")

df = df.iloc[0:, 1:]


# Her film için 1'den 10'a kadar olan puanların dağılımını kullanarak BAR skorunu hesaplıyoruz.
df['bar_score'] = df.apply(lambda x: bayesian_average_rating(x[['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']]), axis=1)

df.sort_values('bar_score', ascending=False).head(20)

# Ağırlıklı Ortalama Derecelendirmeleri
# IMDb, ham veri ortalamaları yerine ağırlıklı oy ortalamalarını yayınlar.
# Bunu açıklamanın en basit yolu, tarafımızdan alınan tüm oyları kabul edip dikkate almamıza rağmen,
# tüm oyların nihai derecelendirme üzerinde aynı etkiye (veya 'ağırlığa') sahip olmamasıdır.

# Olağandışı oylama faaliyeti tespit edildiğinde,
# hizmetimizin güvenilirliğini korumak amacıyla alternatif bir ağırlıklandırma hesaplaması uygulanabilir.
# Derecelendirme mekanizmamızın etkin kalmasını sağlamak için,
# derecelendirmeyi oluşturmak için kullanılan kesin yöntemi açıklamıyoruz.

# Ayrıca IMDb derecelendirmeleri için tam SSS'ye bakın.
