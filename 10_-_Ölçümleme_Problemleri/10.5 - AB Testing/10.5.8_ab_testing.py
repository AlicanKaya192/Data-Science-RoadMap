################################
# Temel İstatistik Kavramları
################################

# İstatistiksel analizler ve AB testleri için gerekli kütüphaneleri import ediyoruz.
# numpy: Bilimsel hesaplamalar için.
# pandas: Veri manipülasyonu ve analizi için.
# matplotlib & seaborn: Veri görselleştirme için.
# statsmodels: İstatistiksel modeller ve testler için.
# scipy.stats: İstatistiksel fonksiyonlar ve testler (t-testi, shapiro, levene vb.) için.
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# !pip install statsmodels
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

# Pandas görüntüleme ayarlarını yapıyoruz.
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

################################
# Sampling (Örnekleme)
################################

# Örnekleme, büyük bir popülasyondan (anakütle) onu temsil edecek daha küçük bir alt küme seçme işlemidir.
# Amaç, tüm popülasyonu incelemek zor veya maliyetli olduğunda, örneklem üzerinden popülasyon hakkında çıkarımlarda bulunmaktır.

# 0 ile 80 arasında rastgele sayılardan oluşan 10.000 elemanlı bir popülasyon oluşturuyoruz.
populasyon = np.random.randint(0, 80, 10000)
# Popülasyonun ortalamasını hesaplıyoruz. Bu bizim "gerçek" değerimizdir.
populasyon.mean()

# Sonuçların tekrarlanabilir olması için rastgelelik tohumunu (seed) sabitliyoruz.
np.random.seed(115)

# Popülasyondan rastgele 100 elemanlı bir örneklem çekiyoruz.
orneklem = np.random.choice(a=populasyon, size=100)
# Örneklemin ortalamasını hesaplıyoruz. Popülasyon ortalamasına yakın olmasını bekleriz.
orneklem.mean()

# Merkezi Limit Teoremi (Central Limit Theorem) ve Büyük Sayılar Yasası'nı gözlemlemek için:
# Popülasyondan 10 farklı örneklem çekiyoruz.
np.random.seed(10)
orneklem1 = np.random.choice(a=populasyon, size=100)
orneklem2 = np.random.choice(a=populasyon, size=100)
orneklem3 = np.random.choice(a=populasyon, size=100)
orneklem4 = np.random.choice(a=populasyon, size=100)
orneklem5 = np.random.choice(a=populasyon, size=100)
orneklem6 = np.random.choice(a=populasyon, size=100)
orneklem7 = np.random.choice(a=populasyon, size=100)
orneklem8 = np.random.choice(a=populasyon, size=100)
orneklem9 = np.random.choice(a=populasyon, size=100)
orneklem10 = np.random.choice(a=populasyon, size=100)

# Bu 10 örneklemin ortalamalarının ortalamasını alıyoruz.
# Örneklem sayısı arttıkça, bu ortalamanın popülasyon ortalamasına daha da yaklaştığını görürüz.
(orneklem1.mean() + orneklem2.mean() + orneklem3.mean() + orneklem4.mean() +
 orneklem5.mean() + orneklem6.mean() + orneklem7.mean() + orneklem8.mean() +
 orneklem9.mean() + orneklem10.mean()) / 10


################################
# Descriptive Statistics (Betimsel İstatistikler)
################################

# Veri setini özetlemek ve temel özelliklerini anlamak için kullanılır.
# Ortalama, medyan, standart sapma, minimum, maksimum gibi değerleri içerir.

df = sns.load_dataset("tips")
# describe() fonksiyonu sayısal değişkenlerin temel istatistiklerini verir.
df.describe().T


################################
# Confidence Intervals (Güven Aralıkları)
################################

# Güven aralığı, bir popülasyon parametresinin (örneğin ortalama) belirli bir olasılıkla (genellikle %95)
# içinde bulunacağı değer aralığıdır.
# "Bu örneklemden yola çıkarak, popülasyon ortalamasının %95 ihtimalle bu iki değer arasında olduğunu söyleyebilirim" demektir.

# Tips Veri Setindeki Sayısal Değişkenler İçin Güven Aralıkları
df = sns.load_dataset("tips")
df.describe().T

df.head()

# total_bill değişkeni için %95 güven aralığını hesaplıyoruz.
# Yani tüm müşterilerin hesap ortalaması %95 ihtimalle bu aralıktadır.
sms.DescrStatsW(df["total_bill"]).tconfint_mean()

# tip (bahşiş) değişkeni için %95 güven aralığı.
sms.DescrStatsW(df["tip"]).tconfint_mean()

# Titanic Veri Setindeki Sayısal Değişkenler için Güven Aralıkları
df = pd.read_csv("Datasets ( Genel )/titanic.csv")
df.describe().T

# Yolcuların yaş ortalaması için güven aralığı (eksik değerleri çıkararak).
sms.DescrStatsW(df["age"].dropna()).tconfint_mean()

# Yolcuların ödediği bilet ücreti (fare) için güven aralığı.
sms.DescrStatsW(df["fare"].dropna()).tconfint_mean()


################################
# Correlation (Korelasyon)
################################

# İki değişken arasındaki ilişkinin yönünü ve şiddetini ölçer.
# -1 ile +1 arasında değer alır.
# +1: Mükemmel pozitif ilişki (biri artarken diğeri de artar).
# -1: Mükemmel negatif ilişki (biri artarken diğeri azalır).
# 0: İlişki yok.

# Bahşiş veri seti:
# total_bill: yemeğin toplam fiyatı (bahşiş ve vergi değil)
# tip: bahşiş
# sex: ücreti ödeyen kişinin cinsiyeti (0 = male, 1 = female)
# smoker: grupta sigara içen var mı? (0 = No, 1 = Yes)
# day: gün (3 = Thur, 4 = Fri, 5 = Sat, 6 = Sun)
# time: ne zaman ? (0 = Day, 1 = Night)
# size: grupta kaç kişi var ?

df = sns.load_dataset("tips")
df.head()

# total_bill değişkeni bahşişi de içeriyor olabilir, bu yüzden net hesap tutarını bulmak için bahşişi çıkarıyoruz.
# (Not: Veri setinin orijinalinde total_bill bahşişi içermiyor olabilir ama burada bir düzeltme yapılmış gibi görünüyor veya analiz tercihi)
df['total_bill'] = df['total_bill'] - df['tip']

# Hesap tutarı ile bahşiş arasındaki ilişkiyi görselleştiriyoruz (Scatter Plot).
df.plot.scatter('total_bill', 'tip')
plt.show()

# Hesap tutarı ile bahşiş arasındaki korelasyon katsayısını hesaplıyoruz.
# Pozitif ve güçlü bir ilişki çıkması beklenir (hesap arttıkça bahşiş artar).
df['tip'].corr(df['total_bill'])


################################
# AB Testing ( Bağımsız İki Örneklem Testi )
################################

# İki grup ortalaması arasında istatistiksel olarak anlamlı bir fark olup olmadığını test etmek için kullanılır.
# Örneğin: Yeni tasarım web sitesi ile eski tasarım web sitesinin dönüşüm oranları arasında fark var mı?

# Adımlar:
# 1. Hipotezleri Kur
# 2. Varsayım Kontrolü
#   - 1. Normallik Varsayımı (Shapiro-Wilk Testi)
#   - 2. Varyans Homojenliği (Levene Testi)
# 3. Hipotezin Uygulaması
#   - 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test - ttest_ind)
#   - 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test - mannwhitneyu)
# Not:
# - Normallik sağlanmıyorsa direk 2 numara (non-parametrik).
# - Varyans homojenliği sağlanmıyorsa 1 numaraya (ttest_ind) equal_var=False argümanı girilir.
# - Normallik incelemesi öncesi aykırı değer incelemesi ve düzeltmesi yapmak faydalı olabilir.


################################
# Uygulama 1: Sigara İçenler ile İçmeyenlerin Hesap Ortalamaları Arasında İst Ol An Fark var mı?
################################

df = sns.load_dataset("tips")
df.head()

# Sigara içen ve içmeyenlerin ortalama hesap tutarlarına bakıyoruz.
# Matematiksel (gözlemlenen) bir fark var ama bu şans eseri mi oluştu yoksa istatistiksel olarak anlamlı mı?
df.groupby("smoker").agg({"total_bill": "mean"})

################################
# 1. Hipotezi Kur
################################

# H0: M1 = M2 (Sigara içenler ve içmeyenlerin hesap ortalamaları arasında istatistiksel olarak anlamlı bir fark yoktur.)
# H1: M1 != M2 (Sigara içenler ve içmeyenlerin hesap ortalamaları arasında istatistiksel olarak anlamlı bir fark vardır.)

################################
# 2. Varsayım Kontrolü
################################

# Normallik Varsayımı
# Varyans Homojenliği

################################
# Normallik Varsayımı
################################

# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1: .. sağlanmamaktadır.

# Sigara içenler grubu için normallik testi (Shapiro-Wilk)
test_stat, pvalue = shapiro(df.loc[df["smoker"] == "Yes", "total_bill"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p-value < 0.05 ise H0 RED (Normal dağılmıyor)
# p-value > 0.05 ise H0 REDDEDİLEMEZ (Normal dağılıyor)

# Sigara içmeyenler grubu için normallik testi
test_stat, pvalue = shapiro(df.loc[df["smoker"] == "No", "total_bill"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))


################################
# Varyans Homojenliği Varsayımı
################################

# H0: Varyanslar homojendir.
# H1: Varyanslar homojen değildir.

# Levene testi ile varyansların homojenliğini kontrol ediyoruz.
test_stat, pvalue = levene(df.loc[df["smoker"] == "Yes", "total_bill"],
                           df.loc[df["smoker"] == "No", "total_bill"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p-value < 0.05 ise H0 RED (Varyanslar homojen değil)
# p-value > 0.05 ise H0 REDDEDİLEMEZ (Varyanslar homojen)


################################
# Hipotezin Uygulaması
################################

# 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
# 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)


################################
# 1.1 Varsayımlar Sağlanıyorsa Bağımsız İki Örneklem T Testi (Parametrik Test)
################################

# Eğer normallik ve varyans homojenliği sağlanıyorsa ttest_ind kullanılır.
# equal_var=True varyanslar homojen ise, False ise Welch testi yapılır.
test_stat, pvalue = ttest_ind(df.loc[df["smoker"] == "Yes", "total_bill"],
                              df.loc[df["smoker"] == "No", "total_bill"],
                              equal_var=True)
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p-value < 0.05 ise H0 RED (Anlamlı bir fark vardır)
# p-value > 0.05 ise H0 REDDEDİLEMEZ (Anlamlı bir fark yoktur)


################################
# 1.2 Varsayımlar Sağlanmıyorsa Mannwhitneyu Testi (Non-Parametrik Test)
################################

# Eğer normallik varsayımı sağlanmıyorsa Mann-Whitney U testi kullanılır.
# Bu test medyanları karşılaştırır ve dağılım varsayımı gerektirmez.
test_stat, pvalue = mannwhitneyu(df.loc[df["smoker"] == "Yes", "total_bill"],
                                 df.loc[df["smoker"] == "No", "total_bill"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))


################################
# Uygulama 2: Titanic Kadın ve Erkek Yolcuların Yaş Ortalaması Arasında İstatiksel Olarak Anlamlı Fark var mı?
################################

df = pd.read_csv("Datasets ( Genel )/titanic.csv")
df.head()

# Kadın ve erkeklerin yaş ortalamalarına bakıyoruz.
df.groupby("sex").agg({"age": "mean"})

# 1. Hipotezi Kur
# H0: M1 = M2 (Kadın ve Erkek Yolcuların Yaş Ortalamaları Arasında İstatistiksel Olarak Anlamlı Bir Fark Yoktur)
# H1: M1 != M2 (Kadın ve Erkek Yolcuların Yaş Ortalamaları Arasında İstatistiksel Olarak Anlamlı Bir Fark Vardır)

# 2. Varsayım Kontrolü

# Normallik Varsayımı
# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1: .. sağlanmamaktadır.

# Kadın yolcuların yaş dağılımı için normallik testi (eksik değerleri çıkararak).
test_stat, pvalue = shapiro(df.loc[df["sex"] == "female", "age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# Erkek yolcuların yaş dağılımı için normallik testi.
test_stat, pvalue = shapiro(df.loc[df["sex"] == "male", "age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# Varyans Homojenliği
# H0: Varyanslar homojendir.
# H1: Varyanslar homojen değildir.

# Varyans homojenliği testi.
test_stat, pvalue = levene(df.loc[df["sex"] == "female", "age"].dropna(),
                           df.loc[df["sex"] == "male", "age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# Varsayımlar sağlanmadığı için (özellikle normallik) non-parametrik test (Mann-Whitney U) uygulanacak.

test_stat, pvalue = mannwhitneyu(df.loc[df["sex"] == "female", "age"].dropna(),
                                 df.loc[df["sex"] == "male", "age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p-value < 0.05 olduğu için H0 RED.
# Yani kadın ve erkek yolcuların yaş ortalamaları arasında istatistiksel olarak anlamlı bir fark vardır.


################################
# Uygulama 3: Diyabet Hastalığı Olan ve Olmayanların Yaşları Arasında İstatiksel Olarak Anlamlı Fark var mı?
################################

# Diyabet veri setini yüklüyoruz.
df = pd.read_csv("Datasets ( Genel )/diabetes.csv")
df.head()

# Outcome: 1 (Diyabet hastası), 0 (Diyabet hastası değil)
# Age: Yaş

# 1. Hipotezi Kur
# H0: M1 = M2 (Diyabet hastası olan ve olmayanların yaş ortalamaları arasında fark yoktur)
# H1: M1 != M2 (Diyabet hastası olan ve olmayanların yaş ortalamaları arasında fark vardır)

# 2. Varsayım Kontrolü (Normallik ve Varyans Homojenliği)
# ... (Burada shapiro ve levene testleri yapılmalı)

# 3. Hipotez Testi
# Varsayımların sonucuna göre ttest_ind veya mannwhitneyu uygulanmalı.

# Örnek olarak ortalamalara bakalım:
# Diyabet durumu (Outcome) 1 olanlar ve 0 olanlar için yaş ortalamalarını inceliyoruz.
df.groupby("Outcome").agg({"Age": "mean"})

# 1. Hipotezi Kur
# H0: M1 = M2 (Diyabet hastası olan ve olmayanların yaş ortalamaları arasında istatistiksel olarak anlamlı bir fark yoktur.)
# H1: M1 != M2 (Diyabet hastası olan ve olmayanların yaş ortalamaları arasında istatistiksel olarak anlamlı bir fark vardır.)

# 2. Varsayımları İncele

# Normallik Varsayımı (H0: Normal dağılım varsayımı sağlanmaktadır.)
# Diyabet hastası olanların yaş dağılımının normalliğini test ediyoruz.
test_stat, pvalue = shapiro(df.loc[df["Outcome"] == 1, "Age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# Diyabet hastası olmayanların yaş dağılımının normalliğini test ediyoruz.
test_stat, pvalue = shapiro(df.loc[df["Outcome"] == 0, "Age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# Normallik varsayımı sağlanmadığı için (p-value < 0.05) non-parametrik test (Mann-Whitney U) uygulanacak.
# Eğer varsayım sağlansaydı parametrik test (T-Test) uygulanacaktı.

# Hipotez (H0: M1 = M2)
# Mann-Whitney U testi ile iki grubun medyanları arasında anlamlı bir fark olup olmadığını test ediyoruz.
test_stat, pvalue = mannwhitneyu(df.loc[df["Outcome"] == 1, "Age"].dropna(),
                                 df.loc[df["Outcome"] == 0, "Age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))


###################################################
# İş Problemi: Kursun Büyük Çoğunluğunu İzleyenler ile İzlemeyenlerin Puanları Birbirinden Farklı mı?
###################################################

# H0: M1 = M2 (Kursun %75'inden fazlasını izleyenler ile %25'inden azını izleyenlerin puan ortalamaları arasında istatistiksel olarak anlamlı bir fark yoktur.)
# H1: M1 != M2 (.... vardır.)

# Veri setini okuyoruz.
df = pd.read_csv("Datasets ( Genel )/course_reviews.csv")
df.head()

# İlerleme durumu %75'ten büyük olanların ortalama puanı:
df[(df["Progress"] > 75)]["Rating"].mean()

# İlerleme durumu %25'ten küçük olanların ortalama puanı:
df[(df["Progress"] < 25)]["Rating"].mean()

# Normallik Varsayımı Kontrolü
# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1: Normal dağılım varsayımı sağlanmamaktadır.

# %75'ten fazla izleyenlerin puan dağılımı normal mi?
test_stat, pvalue = shapiro(df[(df["Progress"] > 75)]["Rating"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# %25'ten az izleyenlerin puan dağılımı normal mi?
test_stat, pvalue = shapiro(df[(df["Progress"] < 25)]["Rating"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# Normallik varsayımı sağlanmadığı için (p-value < 0.05) Non-Parametrik Test (Mann-Whitney U) uyguluyoruz.
test_stat, pvalue = mannwhitneyu(df[(df["Progress"] > 75)]["Rating"],
                                 df[(df["Progress"] < 25)]["Rating"])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))


######################################################
# AB Testing (İki Örneklem Oran Testi)
######################################################

# İki farklı grubun oranları arasında anlamlı bir fark olup olmadığını test etmek için kullanılır.
# Örneğin: Yeni tasarımın dönüşüm oranı ile eski tasarımın dönüşüm oranı arasında fark var mı?

# H0: p1 = p2
# Yeni Tasarımın Dönüşüm Oranı ile Eski Tasarımın Dönüşüm Oranı Arasında İstatistiksel Olarak Anlamlı Bir Farklılık Yoktur.
# H1: p1 != p2
# Yeni Tasarımın Dönüşüm Oranı ile Eski Tasarımın Dönüşüm Oranı Arasında İstatistiksel Olarak Anlamlı Bir Farklılık Vardır.

# Başarı sayıları (dönüşüm sayıları) ve gözlem sayıları (toplam ziyaretçi sayıları) tanımlanır.
basari_sayisi = np.array([300, 250])
gozlem_sayilari = np.array([1000, 1100])

# Proportions Z-Testi uygulanır.
proportions_ztest(count=basari_sayisi, nobs=gozlem_sayilari)

# Oranları hesaplayarak karşılaştırma yapalım:
basari_sayisi / gozlem_sayilari


############################
# Uygulama: Kadın ve Erkeklerin Hayatta Kalma Oranları Arasında İstatistiksel Olarak Anlamlı Farklılık var mıdır?
############################

# H0: p1 = p2
# Kadın ve Erkeklerin Hayatta Kalma Oranları Arasında İstatistiksel Olarak Anlamlı Bir Fark Yoktur.

# H1: p1 != p2
# Kadın ve Erkeklerin Hayatta Kalma Oranları Arasında İstatistiksel Olarak Anlamlı Bir Fark Vardır.

# Veri setini yüklüyoruz.
df = pd.read_csv("Datasets ( Genel )/titanic.csv")
df.head()

# Kadınların hayatta kalma oranı:
df.loc[df["sex"] == "female", "survived"].mean()

# Erkeklerin hayatta kalma oranı:
df.loc[df["sex"] == "male", "survived"].mean()

# Başarı sayılarını (hayatta kalan sayısı) hesaplıyoruz.
female_succ_count = df.loc[df["sex"] == "female", "survived"].sum()
male_succ_count = df.loc[df["sex"] == "male", "survived"].sum()

# Toplam gözlem sayılarını (toplam yolcu sayısı) ve başarı sayılarını kullanarak testi uyguluyoruz.
test_stat, pvalue = proportions_ztest(count=[female_succ_count, male_succ_count],
                                      nobs=[df.loc[df["sex"] == "female", "survived"].shape[0],
                                            df.loc[df["sex"] == "male", "survived"].shape[0]])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))


######################################################
# ANOVA (Analysis of Variance)
######################################################

# İkiden fazla grup ortalamasını karşılaştırmak için kullanılır.
# Örneğin: Haftanın günlerine göre ödenen hesap ortalamaları arasında fark var mı?

# Veri setini yüklüyoruz. (Bu veri seti seaborn kütüphanesi içinden çekilmektedir.)
df = sns.load_dataset("tips")
df.head()

# Günlere göre toplam hesap ortalamalarına bakıyoruz.
df.groupby("day")["total_bill"].mean()

# 1. Hipotezleri kur

# HO: m1 = m2 = m3 = m4
# Grup ortalamaları arasında istatistiksel olarak anlamlı bir fark yoktur.

# H1: En az bir grup ortalaması diğerlerinden farklıdır.

# 2. Varsayım kontrolü

# Normallik varsayımı
# Varyans homojenliği varsayımı

# Varsayım sağlanıyorsa One Way ANOVA (Parametrik)
# Varsayım sağlanmıyorsa Kruskal-Wallis (Non-Parametrik)

# H0: Normal dağılım varsayımı sağlanmaktadır.

for group in list(df["day"].unique()):
    pvalue = shapiro(df.loc[df["day"] == group, "total_bill"])[1]
    print(group, 'p-value: %.4f' % pvalue)


# H0: Varyans homojenliği varsayımı sağlanmaktadır.

test_stat, pvalue = levene(df.loc[df["day"] == "Sun", "total_bill"],
                           df.loc[df["day"] == "Sat", "total_bill"],
                           df.loc[df["day"] == "Thur", "total_bill"],
                           df.loc[df["day"] == "Fri", "total_bill"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))


# 3. Hipotez testi ve p-value yorumu

# Hiç biri sağlamıyor.
df.groupby("day").agg({"total_bill": ["mean", "median"]})


# HO: Grup ortalamaları arasında ist ol anl fark yoktur

# parametrik anova testi:
f_oneway(df.loc[df["day"] == "Thur", "total_bill"],
         df.loc[df["day"] == "Fri", "total_bill"],
         df.loc[df["day"] == "Sat", "total_bill"],
         df.loc[df["day"] == "Sun", "total_bill"])

# Nonparametrik anova testi:
kruskal(df.loc[df["day"] == "Thur", "total_bill"],
        df.loc[df["day"] == "Fri", "total_bill"],
        df.loc[df["day"] == "Sat", "total_bill"],
        df.loc[df["day"] == "Sun", "total_bill"])

from statsmodels.stats.multicomp import MultiComparison
comparison = MultiComparison(df['total_bill'], df['day'])
tukey = comparison.tukeyhsd(0.05)
print(tukey.summary())