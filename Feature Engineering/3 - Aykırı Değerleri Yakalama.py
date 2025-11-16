import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: "%.3f" % x)
pd.set_option('display.width', 500)

def load_application_train():
    data = pd.read_csv("Datasets/Feature Engineering/Outliers/application_train.csv")
    return data

df = load_application_train()
df.head()

def load():
    data = pd.read_csv("Datasets/Feature Engineering/Outliers/titanic.csv")
    return data

df = load()
df.head()

def load():
    data = pd.read_csv("Datasets/Feature Engineering/Outliers/course_reviews.csv")
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

df[(df["Age"] < low) | (df["Age"] > up)] # Low değerden küçük ve Up değerden yüksek olanları getirir. Bunlar aykırı değerlerdir.
# Low değerimiz eksilerde olduğu için herhangi küçük yaş grubu gelmeyecek çünkü - yaş yok.

# Diyelim ki gelen sonuçlara bir şey yapmak istiyoruz şuan veya daha sonra bir şey yapmak istersek bize indexleri lazım.
# Peki bunun için ne yapabiliriz. "Nasıl bu index'leri seçebilirim ?"
# Az önce yaptığımız seçim işleminin sonuna .index der isem;

df[(df["Age"] < low) | (df["Age"] > up)].index # Bu şekilde aykırı değerlerin index'lerini eğer istersek tutabiliriz.

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
df[(df["Age"] < low) | (df["Age"] > up)].head()

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

num_cols = [col for col in num_cols if col not in "PassengerId"] # PassengerId bir istisna olduğu için ondan kurtuluyoruz.

for col in num_cols:
    print(col, outlier_checker(dff, col))

cat_cols, num_cols, cat_but_car = grab_col_names(dff)

num_cols = [col for col in num_cols if col not in "SK_ID_CURR"]

for col in num_cols:
    print(col, outlier_checker(dff, col))

