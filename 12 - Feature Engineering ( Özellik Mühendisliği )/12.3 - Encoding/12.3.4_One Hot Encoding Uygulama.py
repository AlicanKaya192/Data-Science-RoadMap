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


###################################
# One-Hot Encoding
###################################

df = load()
df.head()

# Embarked değişkeninin kategorik sınıflarının frekansını gösterir.
# Kategorik değişkenleri dummies'e çevirmeden önce dağılımı görmek önemlidir.
df["Embarked"].value_counts()

# pd.get_dummies:
# Kategorik değişkenleri modelin anlayabileceği şekilde 0-1 formatında sütunlara ayırır.
# Buna "one-hot encoding" veya "dummy değişken oluşturma" denir.
pd.get_dummies(df, columns=["Embarked"]).head()

# drop_first=True:
# Dummy değişken tuzağına (dummy variable trap) düşmemek için ilk kategoriyi düşer.
# İzahı:
# Eğer Embarked_C, Embarked_Q, Embarked_S şeklinde üç sütun oluşturursak
# bu üç sütun birbirleri üzerinden türetilebilir hale gelir (toplamları her zaman 1 olur).
# Yani değişkenler arasında tam doğrusal bağlantı oluşur (multicollinearity).
# Bu nedenle drop_first=True yaparak ilk kategoriyi atarız ve bu sorunu önleriz.
pd.get_dummies(df, columns=["Embarked"], drop_first=True).head()

# dummy_na=True:
# Eksik değerleri de ayrı bir kategori (örn. Embarked_nan) gibi gösterir.
# Eğer bir değişkende eksik değerlerin model tarafından özel bir sınıf gibi ele alınmasını istiyorsak
# dummy_na=True kullanırız.
# Ancak dikkat:
# Eksik değerleri kategori haline getirirsek, model bunları "ayrı bir kategori" gibi yorumlar.
# Bu her durumda istenmez; problem yapısına göre düşünülmelidir.
pd.get_dummies(df, columns=["Embarked"], dummy_na=True).head()

# Birden fazla kategorik değişkene aynı anda dummies uygulama.
# Sex ve Embarked değişkenleri için tek seferde 0-1 sütunlar üretilir.
pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True).head()


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    # One-hot encoding fonksiyonu:
    # categorical_cols listesindeki kategorik değişkenleri dummies'e çevirir.
    # drop_first argümanı True olursa dummy tuzağına düşmemek için ilk kategori atılır.
    dataframe = pd.get_dummies(dataframe,
                               columns=categorical_cols,
                               drop_first=drop_first)
    return dataframe


df = load()

# Burada normalde cat_cols, num_cols, cat_but_car gibi değişkenler alınabilirdi.
# cat_but_car: Cardinal kategorik değişkenleri (çok fazla sınıf) ayırmak için kullanılır.
# Biz bu satırı sadece gösterim amaçlı yoruma aldık.
# cat_cols, num_cols, cat_but_car = grab_col_names(df)

# ohe_cols seçimi:
# Kategorik ve kardinal olmayan değişkenleri seçmek için:
# uniq sayısı 2’den büyük ve 10’dan küçük/sınırlı olan değişkenler seçilir.
# Yani çok fazla sınıfı olmayan kategorik değişkenler one-hot encoding için uygundur.
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

# Seçtiğimiz değişkenlere one-hot encoding uyguluyoruz.
# drop_first=False: İlk kategoriyi düşmüyoruz, tüm kategorileri görüyoruz.
# Eğer regresyon modeli kullanılacaksa drop_first=True yapılabilir.
one_hot_encoder(df, ohe_cols).head()


# ------------------------------------------------------------------------------
# GENEL ÖZET
# ------------------------------------------------------------------------------

# One-hot encoding (dummies), kategorik değişkenleri makine öğrenmesi modellerinin
# anlayabileceği 0-1 formatındaki sütunlara çevirme işlemidir.
#
# Örneğin Embarked değişkeni: ["C", "Q", "S"] ise
# bunlar ayrı sütunlar olarak Embarked_C, Embarked_Q, Embarked_S şeklinde dönüştürülür.
#
# Dummy değişken tuzağı (dummy variable trap):
# Tüm kategorileri sütun olarak eklediğimizde sütunlar tamamen birbirinden
# üretilebilir hale gelir. (Örn: üç sütun toplamı = 1). Bu durum çoklu doğrusal
# bağlantı (multicollinearity) oluşturur ve özellikle regresyon modellerini bozar.
#
# Bunu engellemek için drop_first=True kullanılır:
# İlk kategoriyi silerek referans sınıf oluşturur ve değişkenler arasındaki tam
# doğrusal ilişkiyi ortadan kaldırır.
#
# Eksik değerleri kategori olarak göstermek için dummy_na=True:
# Eğer değişkende eksik değerler varsa ve bu eksik değerlerin de model tarafından
# ayrı bir grup olarak öğrenilmesi isteniyorsa dummy_na=True kullanılır.
# Bu durumda örneğin Embarked_nan isimli ek bir sütun oluşur.
#
# ohe_cols listesi seçilirken:
# Çok fazla sınıfı olan değişkenler one-hot yapılmaz (kardinal değişkenler).
# Bunun nedeni veri boyutunu aşırı arttırmasıdır.
# Bu yüzden uniq değeri düşük olan kategorikler tercih edilir.
#
# Özetle:
# - get_dummies → kategorikleri 0-1'e çevirir
# - drop_first=True → dummy tuzağını engeller
# - dummy_na=True → eksikleri kategori yapar
# - one_hot_encoder fonksiyonu → tek yerde otomatik OHE işlemi yapar
#
# Bu teknik, makine öğrenmesi modellerinin daha iyi öğrenebilmesi için
# kategorik veriyi sayısal forma dönüştürmenin en yaygın ve güvenli yoludur.
# ------------------------------------------------------------------------------
