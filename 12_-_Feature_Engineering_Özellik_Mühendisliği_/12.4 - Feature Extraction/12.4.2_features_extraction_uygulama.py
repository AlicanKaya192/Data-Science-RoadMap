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


##############################
# Feature Extraction ( Özellik Çıkarımı )
##############################

##############################
# Binary Features: Flag, Bool, True-False
##############################

import pandas as pd
from statsmodels.stats.proportion import proportions_ztest

df = load()
df.head()

# =============================================================================
# 1. ADIM: KABİN BİLGİSİ (NEW_CABIN_BOOL)
# =============================================================================

# 'Cabin' sütunundaki karmaşıklığı giderip, 'Kabini Var mı?' (1/0) bilgisine indirgiyoruz.
df['NEW_CABIN_BOOL'] = df["Cabin"].notnull().astype('int')

# Oluşturduğumuz bu yeni grupların hayatta kalma ortalamalarına bakıyoruz.
print("Kabin Durumuna Göre Hayatta Kalma Ortalamaları:")
print(df.groupby("NEW_CABIN_BOOL").agg({"Survived": "mean"}))

# --- İSTATİSTİKSEL TEST: PROPORTION Z-TEST ---
# NEDEN BU TESTİ YAPIYORUZ?
# İki farklı grubun (Kabini olanlar vs Olmayanlar) başarı oranlarını (Hayatta kalma)
# karşılaştırdığımız için 'İki Oran Z-Testi' (Two-Proportion Z-Test) kullanıyoruz.
# Amacımız: Aradaki farkın şans eseri olup olmadığını kanıtlamak.

test_stat, p_value = proportions_ztest(
    count=[
        df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].sum(), # 1. Grubun başarı sayısı
        df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].sum()  # 2. Grubun başarı sayısı
    ],
    nobs=[
        df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].shape[0], # 1. Grubun toplam sayısı
        df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].shape[0]  # 2. Grubun toplam sayısı
    ]
)

# SONUCUN YORUMLANMASI:
# Eğer p-value < 0.05 ise; aradaki fark istatistiksel olarak anlamlıdır.
# Yani bu yeni özellik (Feature), model için ayırt edici bir bilgidir.
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, p_value))


# =============================================================================
# 2. ADIM: YALNIZLIK DURUMU (NEW_IS_ALONE)
# =============================================================================

# Aile üyeleri (SibSp + Parch) toplamı 0 ise kişi yalnızdır.
df.loc[((df['SibSp'] + df['Parch']) > 0), 'NEW_IS_ALONE'] = 'NO'
df.loc[((df['SibSp'] + df['Parch']) == 0), 'NEW_IS_ALONE'] = 'YES'

print("\nYalnız Olma Durumuna Göre Hayatta Kalma Ortalamaları:")
print(df.groupby("NEW_IS_ALONE").agg({"Survived": "mean"}))

# Aynı testi bu yeni özellik için de uyguluyoruz.
test_stat, p_value = proportions_ztest(
    count=[
        df.loc[df["NEW_IS_ALONE"] == 'YES', "Survived"].sum(),
        df.loc[df["NEW_IS_ALONE"] == 'NO', "Survived"].sum()
    ],
    nobs=[
        df.loc[df["NEW_IS_ALONE"] == 'YES', "Survived"].shape[0],
        df.loc[df["NEW_IS_ALONE"] == 'NO', "Survived"].shape[0]
    ]
)

# p-value kontrolü: < 0.05 ise özellik anlamlıdır.
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, p_value))


##############################
# Text'ler Üzerinden Özellik Türetme
##############################
# Veri setimizdeki metin (string) ifadeleri, makine öğrenmesi modelinin anlayabileceği
# sayısal veya kategorik verilere dönüştürmek için özellik türetiyoruz.

df.head()


###############
# Letter Count
###############

# İsimlerin karakter uzunluğunu hesaplıyoruz.
# Hipotez: Daha uzun isme sahip olanlar daha soylu veya zengin olabilir mi?
df['NEW_NAME_COUNT'] = df['Name'].str.len()


###############
# Word Count
###############

# İsimde kaç kelime geçtiğini sayıyoruz.
# (Unvanlar, takma adlar vs. kelime sayısını artırır, bu da statü belirtisi olabilir.)
df['NEW_NAME_WORD_COUNT'] = df['Name'].apply(lambda x: len(str(x).split(" ")))


###############
# Özel Yapıları Yakalamak
###############

# İsimlerin içinde "Dr" (Doktor) ifadesi geçenleri yakalayıp sayıyoruz.
# Doktorların kaza anındaki davranışları (yardım etme vs.) hayatta kalmalarını etkilemiş olabilir.
df['NEW_NAME_DR'] = df['Name'].apply(lambda x: len([x for x in x.split() if x.startswith('Dr')]))

# Doktor olanların (1) ve olmayanların (0) hayatta kalma oranlarına bakıyoruz.
df.groupby('NEW_NAME_DR').agg({'Survived': 'mean'})


##############################
# Regex ile Değişken Türetme
##############################

df.head()

# İsimlerin içindeki Mr, Mrs, Miss gibi unvanları Regex (Düzenli İfade) ile çekip alıyoruz.
# Mantık: Boşlukla başla, harflerle devam et, noktayla bitir.
df['NEW_TITLE'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

# Çıkardığımız unvanlara göre yaş ve hayatta kalma istatistiklerini inceliyoruz.
# Bu sayede hangi unvanın (yani hangi sosyal grubun) daha şanslı olduğunu görüyoruz.
df[['NEW_TITLE', 'Survived', 'Age']].groupby(['NEW_TITLE']).agg({'Survived': 'mean', 'Age': ['count', 'mean']})


##############################
# Date Değişkenleri Üretmek
##############################
# Zaman damgalarından (tarih) anlamlı yeni bilgiler (mevsim, gün, geçen süre vb.) çıkarıyoruz.

dff = pd.read_csv("Datasets ( Genel )/course_reviews.csv")
dff.head()
dff.info()

# String formatındaki tarihi, Python'un anlayacağı datetime formatına çeviriyoruz.
dff['Timestamp'] = pd.to_datetime(dff['Timestamp'], format='%Y-%m-%d')

# Year: Yıllık trendleri görmek için yılı ayırıyoruz.
dff['year'] = dff['Timestamp'].dt.year

# Month: Mevsimselliği (Yaz/Kış farkı gibi) yakalamak için ayı ayırıyoruz.
dff['month'] = dff['Timestamp'].dt.month

# year diff: Yorumun yapıldığı tarihten bugüne kaç yıl geçmiş? (Eskilik/Yenilik hesabı)
dff['year_diff'] = date.today().year - dff['Timestamp'].dt.year

# month diff (iki tarih arasındaki ay farkı): yıl farkı + ay farkı
# Sadece yılları çıkarmak yetmez, toplam kaç ay geçtiğini bulmak için yılları 12 ile çarpıp ayları ekliyoruz.
dff['month_diff'] = (date.today().year - dff['Timestamp'].dt.year) * 12 + (date.today().month - dff['Timestamp'].dt.month)

# day name: Gün ismini alıyoruz (Hafta sonu mu, hafta içi mi ayrımı için değerli olabilir).
dff['day_name'] = dff['Timestamp'].dt.day_name()


##############################
# Feature Interactions ( Özellik Etkileşimleri )
##############################
# Burası en kritik yer. Değişkenleri tek başına kullanmak yerine, birbirleriyle etkileşime sokarak
# modelin yakalamakta zorlanacağı karmaşık ilişkileri ona hazır olarak sunuyoruz.

df = load()
df.head()

# Yaş ile Sınıfı (Pclass) çarpıyoruz.
# Mantık: Pclass değeri arttıkça (3. sınıf) refah düşer. Age arttıkça yaşlılık artar.
# Bu çarpım bize refah ve yaşın birleşik etkisini tek bir sayıda verir.
df['NEW_AGE_PCLASS'] = df['Age'] * df['Pclass']

# Ailedeki toplam kişi sayısını buluyoruz (Kardeş/Eş + Ebeveyn/Çocuk + Kendisi).
df['NEW_FAMILY_SIZE'] = df['SibSp'] + df['Parch'] + 1

# AŞAĞIDAKİ BÖLÜMÜN MANTIĞI (Neden tek tek ayırdık?):
# Titanic'te kural "Önce Kadınlar ve Çocuklar" idi.
# Bu yüzden 'Yaş' değişkeni erkekler için hayati önem taşırken (çocuksa kurtulur, yetişkinse ölür),
# kadınlar için etkisi daha farklıdır.
# Eğer sadece 'Sex' ve 'Age' sütunlarını ayrı ayrı verirsek, model bu "Erkek VE Çocuk" ilişkisini
# yakalamakta zorlanabilir.
# Biz burada veriyi manuel olarak segmente ederek (parçalara ayırarak) modele kopya veriyoruz:
# "Bak, bu grup 'genç erkek', bu grup 'yetişkin kadın'. Bunların davranışları birbirinden tamamen farklı."

df.loc[(df['Sex'] == 'male') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'

df.loc[(df['Sex'] == 'male') & (df['Age'] > 21) & (df['Age'] <= 50), 'NEW_SEX_CAT'] = 'maturemale'

df.loc[(df['Sex'] == 'male') & (df['Age'] > 50), 'NEW_SEX_CAT'] = 'seniormale'

df.loc[(df['Sex'] == 'female') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'

df.loc[(df['Sex'] == 'female') & (df['Age'] > 21) & (df['Age'] <= 50), 'NEW_SEX_CAT'] = 'maturefemale'

df.loc[(df['Sex'] == 'female') & (df['Age'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'

df.head()

# Oluşturduğumuz bu yeni kategorilerin (segmentlerin) hayatta kalma oranlarına bakıyoruz.
# Muhtemelen 'maturemale' (yetişkin erkek) en düşük, kadın ve çocuk grupları en yüksek çıkacaktır.
df.groupby('NEW_SEX_CAT').agg({'Survived': 'mean'})