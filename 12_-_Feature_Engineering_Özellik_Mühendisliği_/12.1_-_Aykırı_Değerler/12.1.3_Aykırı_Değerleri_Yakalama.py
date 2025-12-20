import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date

from pandas.io.pytables import dropna_doc
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: "%.3f" % x)
pd.set_option('display.width', 500)

def load_application_train():
    data = pd.read_csv("Datasets ( Genel )/application_train.csv")
    return data

df = load_application_train()
df.head()

def load():
    data = pd.read_csv("Datasets ( Genel )/titanic.csv")
    return data

df = load()
df.head()

def load():
    data = pd.read_csv("Datasets ( Genel )/course_reviews.csv")
    return data

df = load()
df.head()

###############################
# 1. Outliers (Aykırı Değerler)
###############################

###############################
# Aykırı Değerleri Yakalama
###############################

###############################
# Grafik Teknikle Aykırı Değerler
###############################

# Grafik teknikle aykırı değerleri görmek istersek bu durumda kutu grafik kullanılır.

sns.boxplot(x=df["Age"])
plt.show()

# boxplot, kutu grafik, bir sayısal değişkenin dağılım bilgisini verir.
# Elimizde bir sayısal değişken varsa, bu sayısal değişkeni gösterebileceğimiz en yaygın kutu grafikten sonra Histogram
# grafiği kullanılır.

###############################
# Aykırı Değerler Nasıl Yakalanır
###############################

# Önce yapmamız gereken teorik bölümde gördüğümüz eşik değerlere erişmeliyiz. Bir değişkenin çeyrek değerlerini hesaplamalıyız.

q1 = df["Age"].quantile(0.25) # 1. Çeyrek değeri

q3 = df["Age"].quantile(0.75) # 3. Çeyrek değeri

iqr = q3 - q1 # IQR Değeri

up = q3 + 1.5 * iqr # Üst Sınır
low = q1 - 1.5 * iqr # Alt Sınır

# UYARI!!! Burada Alt sınır değerimiz eksi geliyor. Eksik yaş değeri olamayacağı ve yaş değişkenimizde eksik değer
# olmadığı için birazdan yapılacak işlemlerde bunu görmezden gelecek. Bir eylemde bulunmayacak.

df[((df["Age"] < low) | (df["Age"] > up))] # Low değerden küçük ve Up değerden yüksek olanları getirir. Bunlar aykırı değerlerdir.
# Low değerimiz eksilerde olduğu için herhangi küçük yaş grubu gelmeyecek çünkü - yaş yok.

# Diyelim ki gelen sonuçlara bir şey yapmak istiyoruz şuan veya daha sonra bir şey yapmak istersek bize indexleri lazım.
# Peki bunun için ne yapabiliriz. "Nasıl bu index'leri seçebilirim ?"
# Az önce yaptığımız seçim işleminin sonuna .index der isem;

df[((df["Age"] < low) | (df["Age"] > up))].index # Bu şekilde aykırı değerlerin index'lerini eğer istersek tutabiliriz.

###############################
# Aykırı Değer Var Mı, Yok Mu ?
###############################

# Peki diyelim ki hızlı bir şekilde Aykırı değer var mı yok mu öğrenmek istersek ne yapmalıyız peki ?

# Öyle bir şey yapmalıyız ki, az önce yaptığımız sorguda birçok satır gelmesin de sadece orada satır olup olmadığı
# bilgisi dönsün

df[(df["Age"] < low) | (df["Age"] > up)].any(axis=None)

# Burada herhangi bir değer var mı ? any() ifadesini kullanırsak bu durumda içerisinde bir şey olup olmama durumunu sorgular.
# Satır ya da sütuna göre değil de hepsine bakmak istediğimizden axis'i None yaptık.

# Bu işlem True dönecektir. Diğer sorguya göre bize veri döndürmek yerine aykırı değer var mı yok mu bunu söylemiş olacak.
# Kod akışımızı bool tipte True veya False göre sürdürmek istersek bunu yapabiliriz.


###############################
# İşlemleri Fonksiyonlaştırmak
###############################

# Belirtilen değişken için alt ve üst aykırı değer eşiklerini hesaplayan fonksiyon.
# q1 ve q3 varsayılan olarak %25 ve %75 çeyreklik değerlerdir.
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    q1 = dataframe[col_name].quantile(q1)  # 1. çeyreklik değer
    q3 = dataframe[col_name].quantile(q3)  # 3. çeyreklik değer
    iqr = q3 - q1  # Interquartile range (çeyrekler arası genişlik)
    up = q3 + 1.5 * iqr  # Üst sınır
    low = q1 - 1.5 * iqr  # Alt sınır
    return low, up

# Fonksiyon çağırılıyor. q1 ve q3 değerleri default olarak alınıyor.
outlier_thresholds(df, "Age")
outlier_thresholds(df, "Fare")

# Alt ve üst limit değerlerini değişken olarak alma
low, up = outlier_thresholds(df, "Age")

# Age değişkenindeki alt veya üst eşik dışına çıkan aykırı gözlemleri gösterme
df[((df["Age"] < low) | (df["Age"] > up))].head()

# Bir değişkende aykırı değer olup olmadığını kontrol eden fonksiyon
def outlier_checker(dataframe, col_name):
    low, up = outlier_thresholds(dataframe, col_name)  # Eşik değerleri al

    # Eğer alt veya üst sınırı aşan değer varsa True döner
    if dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)].any(axis=None):
        return True
    else:
        return False

    # Dipnot:
    # Eğer outlier_checker fonksiyonunda outlier_thresholds fonksiyonunun çeyreklik değerlerini (q1, q3)
    # değiştirebilmek istersen, outlier_checker fonksiyonunun da parametrelerine q1 ve q3 eklemen gerekir.
    # Aksi hâlde outlier_checker her zaman outlier_thresholds'in varsayılan q1=0.25 ve q3=0.75 değerlerini kullanır.


###############################
# grap_col_names Fonksiyonu
###############################

dff = load_application_train()
dff.head()

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini döndürür.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Args
    -----
        dataframe : dataframe
            Değişken isimleri alınmak istenilen dataframe.
        cat_th : int, optional
            Numerik fakat kategorik olan değişkenler için sınıf eşik değeri.
        car_th : int, optional
            Kategorik fakat kardinal değişkenler için sınıf eşik değeri.

    Returns
    -------
        cat_cols : list
            Kategorik değişken listesi.
        num_cols : list
            Numerik değişken listesi.
        cat_but_car : list
            Kategorik görünümlü kardinal değişken listesi.
    """

    # 1️⃣ Kategorik değişkenleri seç: object, category ve bool tipindeki sütunlar
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    # 2️⃣ Numerik görünümlü fakat az sayıda farklı değere sahip sütunları seç (numerik ama kategorik)
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]

    # 3️⃣ Kategorik ama çok fazla farklı değere sahip sütunları seç (kardinal)
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]

    # 4️⃣ Asıl kategorik değişken listesine numerik ama kategorik olanları ekle
    cat_cols = cat_cols + num_but_cat

    # 5️⃣ Kategorik listeden kardinal değişkenleri çıkar
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # 6️⃣ Numerik değişkenleri seç (int ve float tipinde olanlar)
    num_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["int64", "float64"]]

    # 7️⃣ Numerik listeden kategorik olanları çıkar
    num_cols = [col for col in num_cols if col not in cat_cols]

    # 8️⃣ Özet bilgileri yazdır
    print(f"Observations: {dataframe.shape[0]}")  # Satır sayısı
    print(f"Variables: {dataframe.shape[1]}")     # Sütun sayısı
    print(f"cat_cols: {len(cat_cols)}")           # Kategorik değişken sayısı
    print(f"num_cols: {len(num_cols)}")           # Numerik değişken sayısı
    print(f"cat_but_car: {len(cat_but_car)}")     # Kardinal değişken sayısı
    print(f"num_but_cat: {len(num_but_cat)}")     # Numerik ama kategorik değişken sayısı

    # Değişken listelerini döndür
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in "PassengerId"]
# PassengerId bir istisna olduğu için ondan kurtuluyoruz.

for col in num_cols:
    print(col, outlier_checker(dff, col))

cat_cols, num_cols, cat_but_car = grab_col_names(dff)

num_cols = [col for col in num_cols if col not in "SK_ID_CURR"]

for col in num_cols:
    print(col, outlier_checker(dff, col))


###############################
# Aykırı Değerlerin Kendilerine Erişmek
###############################

# Bu kısımda aykırı değerlere erişmek istediğimiz de ne yapmamız gerekiyor onu göreceğiz.
# Yukarıda manuel şekilde index değerlerine erişmiştik ve değerleri de görmüştük. Şimdi bu işlemi fonksiyonlaştıracağız.

def grap_outliers(dataframe, col_name, index=False):
    # Belirtilen değişken için alt ve üst eşikleri hesapla
    low, up = outlier_checker(dataframe, col_name)

    # Aykırı değerleri filtrele
    outliers = dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)]

    # Eğer aykırı değer sayısı 10'dan büyükse ilk 5 tanesini yazdır
    if outliers.shape[0] > 10:
        print(outliers.head())
    # Değilse tamamını yazdır
    else:
        print(outliers)

    # Kullanıcı index=True derse aykırı değerlerin index'lerini döndür
    if index:
        return outliers.index

grap_outliers(df, "Age")

age_index = grap_outliers(df, "Age", True)


###############################
# Aykırı Değer Problemini Çözme
###############################

############
# Silme
############

# Aykırı değerleri tanımlamak için öncelikle alt ve üst değer gerekiyor.

low, up = outlier_checker(df, "Fare")
df.shape

df[~((df["Fare"] < low) | (df["Fare"] > up))].shape

def remove_outliers(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = df[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers


cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in "PassengerId"]
df.shape

for col in num_cols:
    new_df = remove_outliers(df, col)

df.shape[0] - new_df.shape[0]

# DİKKAT!!!! - 1 tane hücredeki 1 tane aykırılıktan dolayı bir silme işlemi yaptığımızda diğer tam olan gözlemlerdeki
# verileri de silmiş oluyoruz. Bundan dolayı bazı senaryolarda silmek yerine bu değerleri baskılama yöntemiyle baskılamayı da
# tercih edebiliriz.

############
# Baskılama Yöntemi (re-assignment with thresholds)
############

# Baskılama yöntemi de şu şekilde çalışır. Kabul edilebilir eşik değerinin üzerinde kalan değerler eşik değerleriyle
# değiştirilir.

low, up = outlier_thresholds(df, "Fare")

df[((df["Fare"] < low) | (df["Fare"] > up))]["Fare"]

df.loc[((df["Fare"] < low) & (df["Fare"] > up)), "Fare"]

df.loc[(df["Fare"] > up), "Fare"] = up

df.loc[(df["Fare"] < up), "Fare"] = low

def replace_with_threshold(dataframe,variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

df = load()

cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in "PassengerId"]

df.shape

for col in num_cols:
    print(col, outlier_checker(df, col))

for col in num_cols:
    replace_with_threshold(df,col)

for col in num_cols:
    print(col, outlier_checker(df, col))


############
# Recap
############

df = load()

outlier_thresholds(df, "Age")
outlier_checker(df, "Age")
grap_outliers(df, "Age", index=True)

remove_outliers(df, "Age").shape
replace_with_threshold(df, "Age")
outlier_checker(df, "Age")


#############################################################
# Çok Değişkenli Aykırı Değer Analizi: Local Outlier Factor
#############################################################

# Bu bölüme başlamadan önce "Çok Değişkenli Aykırı Değer Ne Demek Ki ?" sorusuna cevap verelim.
# Örnek; Diyelim ki elimizde 2 tane değişken var. Değişkenin birincisi yaş değişkeni olsun, ikincisi ise evlilik
# sayısı değişkeni olsun. Evlilik sayısı değişkenine odaklanacak olursak. "Örneğin, 3 sayısı yani 3 kere evlenmiş olmak
# aykırı bir değer midir?". Hayır, aykırı değer olmayabilir, olabilir, 3 olur, 5 olur.Çok anormal olmayabilir. Sayı
# yükseldikçe evlilik sayısı için aykırı değer olma ihtimali ortaya çıkacaktır. Ama değeri 3 çok normal değil gibi
# gözüküyor. Şimdi yaş değişkenini ele alalım. Örneğin: 18 olsun. 18 sayısı da yaş için anormal bir değer mi? Değil gibi
# gözüküyor ama soru şu; "18 yaşında olup 3 defa evlenmiş olmak durumu anormal midir?". Anormaldir yani, bir aykırı değerdir.
# Anlaşılacağı üzere tek başına aykırı olamayacak bazı değerler birlikte ele alındığında bu durum aykırılık yaratıyor olabilir.
# İşte bundan dolayı aykırı değerlere çok değişkenli olarak da bir bakmak faydalı olacaktır.

# Peki Local Outlier Factor yöntemi nedir ?

# LOF yöntemi çok değişkenli bir aykırı değer belirleme yöntemidir.
# images/Feature Engineering içerisinde ki görselleri inceleyebilirsiniz.

# LOF yöntemi ne yapar ?

# Gözlemleri bulundukları konumda yoğunluk tabanlı skorlayarak, buna göre aykırı değer olabilecek değerleri tanıma
# imkanı sağlar. "Peki bu ne demek ?"

# Bir noktanın lokal yoğunluğu demek, ilgili noktanın etrafında ki komşuluklar demektir. Eğer bir nokta komşularının
# yoğunluğundan anlamlı bir şekilde düşük ise bu durumda bu nokta daha seyrek bir bölgededir. Yani demek ki bu aykırı değer
# olabilir yorumu yapılır.


df = sns.load_dataset("diamonds")
df = df.select_dtypes(include=["float", "int64"])
df = df.dropna()
df.head()
df.shape

for col in df.columns:
    print(col, outlier_checker(df, col))

low, up = outlier_thresholds(df, "carat")

df[((df["carat"] < low) | (df["carat"] > up))].shape
df[((df["depth"] < low) | (df["depth"] > up))].shape

clf = LocalOutlierFactor(n_neighbors=20) # Burada ki girilecek değer sizlere kalmış fakat ön tanım değeri olan 20 önerilir.
clf.fit_predict(df) # LocalOutlierFactor skorlarını getirecek.

df_scores = clf.negative_outlier_factor_ # Takip edilebilirlik için skorları tutuyoruz.
df_scores[0:5] # Kullandığımız metotdan dolayı skorları bize - olarak verecek. Bizde buna göre değerlendirme yapacağız.

# Eğer - olarak değil + olarak değerlendirmek istersek
# df_scores = -df_scores

# Bunu, fonksiyonun bize verdiği orjinal hali yani negatif değerlerle kullanmayı tercih edeceğiz.
# Bunun sebebi eşik değere karar vermek için kullanıcı olarak bir bakış gerçekleştirmek istediğimizde oluşturacak olduğumuz
# elbow yöntemi yani, dirsek yöntemi grafik tekniğinde daha rahat okunabilirlik açısından eksi olarak bırakacağız.

# Buradaki değerlerin 1'e yakın olması inlier olması durumunu gösteriyordu fakat şuan -1'e yakın olması inlier
# olması durumunu gösteriyor gibi değerlendireceğiz. Diyelim ki -1'den -10'a doğru gidiyoruz. -10'a doğru gittikçe
# değerlerin daha aykırı olma eğiliminde olduğunu yorumlayacağız.

np.sort(df_scores)[0:5]

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 20], style='.-')
plt.show()

th = np.sort(df_scores)[3]

df[df_scores < th]
df[df_scores < th].shape

# Aykırı değerler neden aykırı bunu anlamamız lazım.

df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T

df[df_scores < th].drop(axis=0, labels=df[df_scores < th].index)

# Gözlem sayısı çok olduğunda baskılama yöntemini burada kullanak çok mantıklı olmayacaktır. Gözlem sayısı az olduğunda
# çok değişkenli baktıkdan sonra o aykırılık çıkarılmalı veri setinden. Burada ki azlık çokluk çalışmaya göre değişir.