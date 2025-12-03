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
    data = pd.read_csv("Datasets/Feature Engineering/Outliers/titanic.csv")
    return data


def load_application_train():
    data = pd.read_csv("Datasets/Feature Engineering/Outliers/application_train.csv")
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


###################################
# Rare Encoding
###################################

# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi.
# 2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi.
# 3. Rare encoder'ın yazılması.

###################################
# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi.
###################################

# Veri setini yüklüyoruz.
df = load_application_train()

# Örnek: NAME_EDUCATION_TYPE değişkeninin sınıf sayıları ve oranları
df["NAME_EDUCATION_TYPE"].value_counts()

# Bu fonksiyon bize kategorik, numerik ve kategorik görünümlü kardinal değişkenleri getirir.
cat_cols, num_cols, cat_but_car = grab_col_names(df)


# Kategorik değişken özeti çıkaran fonksiyon
def cat_summary(dataframe, col_name, plot=False):
    """
    Bu fonksiyon bir kategorik değişkenin:
    - Frekanslarını (kaç kez geçtiğini)
    - Oranlarını (toplam içindeki yüzdesini)
    - (opsiyonel) countplot grafiğini verir.
    """

    # value_counts() ile kategori sayıları
    # oran için toplam gözlem sayısına bölüyoruz
    print(pd.DataFrame({
        col_name: dataframe[col_name].value_counts(),
        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)
    }))

    print("############################################")

    # Eğer plot=True ise değişken grafiğini çizdiriyoruz
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


# Tüm kategorik kolonlar için özet tabloyu yazıyoruz
for col in cat_cols:
    cat_summary(df, col)

###################################
# 2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi.
###################################

# NAME_INCOME_TYPE değişkenindeki sınıfların dağılımı
df["NAME_INCOME_TYPE"].value_counts()

# Her kategori için TARGET ortalamasına bakıyoruz.
# Böylece bazı kategorilerin hedef değişkenle ilişkisi anlaşılır.
df.groupby("NAME_INCOME_TYPE")["TARGET"].mean()


# Rare kategori analiz fonksiyonu
def rare_analyser(dataframe, target, cat_cols):
    """
    Amaç:
    - Kategorik değişkenlerde kaç sınıf var?
    - Hangi sınıf dataset içinde ne kadar yer kaplıyor? (COUNT, RATIO)
    - Rare sınıfların TARGET üzerindeki etkisi nedir? (TARGET_MEAN)

    Rare encoding yapmadan önce mutlaka bu analiz yapılmalıdır.
    """

    for col in cat_cols:
        # Kategori sayısını yazdır
        print(col, ":", len(dataframe[col].value_counts()))

        # COUNT: kategori frekansları
        # RATIO: frekans / toplam gözlem
        # TARGET_MEAN: o kategori için hedef ortalaması (risk, davranış vs.)
        print(pd.DataFrame({
            "COUNT": dataframe[col].value_counts(),
            "RATIO": dataframe[col].value_counts() / len(dataframe),
            "TARGET_MEAN": dataframe.groupby(col)[target].mean()
        }), end="\n\n\n")


# Rare analizini tüm kategorik değişkenlerde çalıştırıyoruz
rare_analyser(df, "TARGET", cat_cols)


###################################
# 3. Rare encoder'ın yazılması
###################################





















