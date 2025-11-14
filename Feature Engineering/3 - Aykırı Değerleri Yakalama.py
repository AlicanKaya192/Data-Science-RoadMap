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
pd.set_option('max_width', 500)

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










