import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date

from numpy.distutils.conv_template import header
from pandas.io.pytables import dropna_doc
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: "%.3f" % x)
pd.set_option('display.width', 500)


def load():
    data = pd.read_csv("Datasets ( Genel )/titanic.csv")
    return data


def load_application_train():
    data = pd.read_csv("Datasets ( Genel )/application_train.csv")
    return data


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


####################################
# Feature Scaling (Özellik Ölçeklendirme)
####################################
# Değişkenlerin değer aralıklarını birbirine yaklaştırmak ve modellerin daha hızlı/doğru çalışmasını sağlamak için kullanılır.
# Özellikle uzaklık temelli (KNN, K-Means) ve gradient descent kullanan yöntemlerde önemlidir.

####################################
# StandartScaler: Klasik standartlaştırma. Ortalamayı çıkar, standart sapmaya böl. z = (x - u) / s
####################################
# Değişkenin ortalamasını 0, standart sapmasını 1 yapacak şekilde dönüştürür.
# Aykırı değerlerden etkilenir çünkü ortalama ve standart sapma aykırı değerlerden etkilenen metriklerdir.

df = load()
ss = StandardScaler() # StandardScaler nesnesi oluşturulur.
df['Age_standard_scaler'] = ss.fit_transform(df[['Age']]) # Age değişkeni standartlaştırılır ve yeni bir sütun olarak eklenir.
df.head() # İlk 5 satır gözlemlenir.


####################################
# RobustScaler: Medyanı çıkar, IQR'a böl.
####################################
# Aykırı değerlere karşı dayanıklı (robust) bir ölçeklendirme yöntemidir.
# Ortalamayı değil medyanı çıkarır ve standart sapmaya değil IQR'a (Interquartile Range) böler.
# Böylece aykırı değerlerin etkisi minimize edilmiş olur.

rs = RobustScaler() # RobustScaler nesnesi oluşturulur.
df['Age_robust_scaler'] = rs.fit_transform(df[['Age']]) # Age değişkeni Robust Scaler ile dönüştürülür.
df.describe().T # İstatistiksel özet görüntülenir.


####################################
# MinMaxScaler: Verilen 2 değer arasında değişken dönüşümü
####################################
# Değişkenleri genellikle 0 ile 1 arasında (veya belirlenen min-max aralığında) ölçeklendirir.
# Veri setindeki dağılımı korur ancak aykırı değerlere karşı duyarlıdır.

# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# X_scaled = X_std * (max - min) + min

mms = MinMaxScaler() # MinMaxScaler nesnesi oluşturulur.
df['Age_min_max_scaler'] = mms.fit_transform(df[['Age']]) # Age değişkeni 0-1 arasına (varsayılan) dönüştürülür.
df.describe().T # İstatistiksel özet görüntülenir.

df.head() # Değişiklikleri görmek için ilk 5 satıra bakılır.

# Oluşturduğumuz yeni değişkenleri kıyaslamak için içerisinde "Age" geçenleri seçelim.
age_cols = [col for col in df.columns if "Age" in col]

def num_summary(dataframe, numerical_col, plot=False):
    """
    Numerik değişkenlerin özet istatistiklerini yazdırır ve isteğe bağlı olarak histogram grafiğini oluşturur.

    Args:
        dataframe (pd.DataFrame): İşlem yapılacak veri seti.
        numerical_col (str): Özetlenmek istenen numerik değişkenin ismi.
        plot (bool, optional): Grafik çizdirilmek istenirse True, aksi halde False. Varsayılan False.
    """
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99] # İstenen yüzdelik dilimler belirlenir.
    print(dataframe[numerical_col].describe(quantiles).T) # İstatistiksel özet yazdırılır.

    if plot: # Eğer grafik çizdirilmek isteniyorsa:
        dataframe[numerical_col].hist(bins=20) # Histogram grafiği oluşturulur.
        plt.xlabel(numerical_col) # X ekseni isimlendirilir.
        plt.title(numerical_col) # Grafik başlığı eklenir.
        plt.show() # Grafik gösterilir.


# Tüm yaş değişkenleri için özet istatistikleri ve grafikleri görelim.
for col in age_cols:
    num_summary(df, col, plot=True)


####################################
# Numeric to Categorical: Sayısal Değişkenleri Kategorik Değişkenlere Çevirme 
# Binning
####################################
# Sayısal değişkenleri belirli aralıklara bölerek kategorik hale getirme işlemidir.
# qcut fonksiyonu, değişkeni değerlere göre değil, yüzdelik dilimlere (quantile) göre eşit parçalara böler.

df["Age_qcut"] = pd.qcut(df["Age"], 5) # Age değişkeni, değerlerin dağılımına göre 5 eşit parçaya bölünür.