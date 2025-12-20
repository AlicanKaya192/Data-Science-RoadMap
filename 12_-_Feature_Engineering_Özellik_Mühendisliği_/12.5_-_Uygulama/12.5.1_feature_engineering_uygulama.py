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

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    q1 = dataframe[col_name].quantile(q1)  # 1. çeyreklik değer
    q3 = dataframe[col_name].quantile(q3)  # 3. çeyreklik değer
    iqr = q3 - q1  # Interquartile range (çeyrekler arası genişlik)
    up = q3 + 1.5 * iqr  # Üst sınır
    low = q1 - 1.5 * iqr  # Alt sınır
    return low, up

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

def replace_with_threshold(dataframe,variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def missing_values_table(dataframe, na_name=False):
    # 1) Eksik değeri olan kolonları bul
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    # 2) Her kolon için toplam eksik değer sayısı
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)

    # 3) Eksik değer oranı (%)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)

    # 4) Sayı ve oranı tek bir tablo halinde birleştir
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])

    # 5) Sonucu ekrana yazdır
    print(missing_df, end='\n')

    # 6) Eğer kullanıcı kolon isimlerini de görmek isterse onları döndür
    if na_name:
        return na_columns

def label_encoder(dataframe, binary_col):
    # LabelEncoder sınıfını içe aktar (kategorik verileri sayısal değerlere dönüştürmek için)
    labelencoder = LabelEncoder()

    # Belirtilen kolondaki kategorik verileri 0 ve 1 gibi sayısal değerlere dönüştür
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])

    # Güncellenmiş dataframe'i geri döndür
    return dataframe

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

def rare_encoder(dataframe, rare_perc):
    """
    Belirtilen oran (rare_perc) altında frekansa sahip kategorik değişkenlerdeki
    nadir (rare) kategorileri tespit ederek bunları 'Rare' etiketi ile birleştirir.

    Parametreler
    ------------
    dataframe : pandas.DataFrame
        Üzerinde işlem yapılacak veri seti.
    rare_perc : float
        Bir kategorinin nadir olarak kabul edilmesi için eşik oran.
        Örn: 0.01 -> %1'in altında görülen kategoriler 'Rare' yapılır.

    Dönüş
    ------
    pandas.DataFrame
        Nadir kategorileri birleştirilmiş yeni DataFrame.
    """

    temp_df = dataframe.copy()  # Orijinal veri setini bozmamak için kopya oluşturulur.

    # Kategorik değişkenlerden içinde rare kategorisi bulunan sütunları tespit et.
    rare_columns = [
        col for col in temp_df.columns
        if temp_df[col].dtypes == "O"  # Sadece kategorik/object değişkenler
        and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)
    ]
    
    # Rare kolonlar için: rare oranının altında kalan kategorileri 'Rare' ile değiştir.
    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)  # Kategori frekans oranları
        rare_labels = tmp[tmp < rare_perc].index         # Rare olarak işaretlenecek etiketler
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    # One-hot encoding fonksiyonu:
    # categorical_cols listesindeki kategorik değişkenleri dummies'e çevirir.
    # drop_first argümanı True olursa dummy tuzağına düşmemek için ilk kategori atılır.
    dataframe = pd.get_dummies(dataframe,
                               columns=categorical_cols,
                               drop_first=drop_first)
    return dataframe

##############################
# Titanic Uçtan Uca Feature Engineering & Data Preprocessing
##############################

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Not: Aşağıdaki yardımcı fonksiyonların (load, grab_col_names, outlier_checker vb.)
# daha önce tanımlandığını varsayıyoruz.

df = load()
df.shape
df.head()

# Sütun isimlerini standartlaştırmak için hepsini BÜYÜK HARFE çeviriyoruz.
df.columns = [col.upper() for col in df.columns]


##############################
# 1. Feature Engineering ( Değişken Mühendisliği )
##############################
# Modelin veri setindeki kalıpları daha iyi yakalayabilmesi için
# ham veriden yeni, anlamlı değişkenler türetiyoruz.

# Cabin bool: Kabin numarası verisinin kendisinden ziyade, dolu olup olmaması (1/0) önemlidir.
df['NEW_CABIN_BOOL'] = df["CABIN"].notnull().astype('int')

# Name count: İsmin uzunluğu (harf sayısı). Uzun isimler sosyal statü belirtisi olabilir.
df['NEW_NAME_COUNT'] = df["NAME"].str.len()

# Name word count: İsimdeki kelime sayısı (Unvanlar, takma adlar vs.).
df['NEW_NAME_WORD_COUNT'] = df["NAME"].apply(lambda x: len(x.split(" ")))

# Name dr: İsimde 'Dr' (Doktor) ifadesi geçiyor mu? Doktorların hayatta kalma davranışı farklı olabilir.
df['NEW_NAME_DR'] = df["NAME"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))

# Name title: Regex ile isimlerden Mr, Mrs, Miss, Master gibi unvanları çekiyoruz.
# Bu değişken; yaş, cinsiyet ve statü hakkında en temiz bilgiyi veren değişkendir.
df['NEW_TITLE'] = df.NAME.str.extract(' ([A-Za-z]+)\.', expand=False)

# Family size: Gemideki toplam aile birey sayısı (Kardeş/Eş + Ebeveyn/Çocuk + 1 Kendisi).
df['NEW_FAMILY_SIZE'] = df['SIBSP'] + df['PARCH'] + 1

# Age_pclass: Yaş ve Sınıf etkileşimi. Refah düzeyi (Pclass) ve Yaşın (Age) birleşik etkisi.
df['NEW_AGE_PCLASS'] = df['AGE'] * df['PCLASS']

# is alone: Kişi yalnız mı seyahat ediyor? (Aile toplamı 0 ise yalnızdır).
df.loc[((df['SIBSP'] + df['PARCH']) > 0), 'NEW_IS_ALONE'] = 'NO'
df.loc[((df['SIBSP'] + df['PARCH']) == 0), 'NEW_IS_ALONE'] = 'YES'

# Age level: Yaş değişkenini sürekli sayısal halden, kategorik hale (Genç, Yetişkin, Yaşlı) getiriyoruz.
df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'

# sex x age: Cinsiyet ve Yaş kırılımını manuel olarak oluşturuyoruz.
# Amaç: "Önce kadınlar ve çocuklar" kuralını modelin tek bir değişkende görebilmesini sağlamak.
# Örneğin: 'maturemale' (Yetişkin Erkek) muhtemelen en düşük hayatta kalma oranına sahip olacaktır.

df.loc[(df['Sex'] == 'male') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['Sex'] == 'male') & (df['Age'] > 21) & (df['Age'] <= 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['Sex'] == 'male') & (df['Age'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['Sex'] == 'female') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] > 21) & (df['Age'] <= 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'

df.head()
df.shape

# Değişken türlerini (Kategorik, Sayısal, Kardinal) ayrıştırıyoruz.
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# PassengerId sayısal görünse de aslında bir ID olduğu için analiz dışı bırakıyoruz.
num_cols = [col for col in num_cols if "PASSENGERID" not in col]


##############################
# 2. Outliers ( Aykırı Değerler )
##############################
# Sayısal değişkenlerdeki (Age, Fare vb.) aşırı uç değerleri tespit edip baskılıyoruz.
# Bu işlem, özellikle lineer modellerin ve mesafe temelli algoritmaların sapmasını engeller.

for col in num_cols:
    print(col, outlier_checker(df, col)) # Var mı yok mu kontrolü

for col in num_cols:
    replace_with_threshold(df, col) # Eşik değerlerle baskılama (Thresholding)

for col in num_cols:
    print(col, outlier_checker(df, col)) # Tekrar kontrol (Hepsi temizlenmiş olmalı)


##############################
# 3. Missing Values ( Eksik Değerler )
##############################
# Eksik verileri doldurma veya silme stratejilerini uyguluyoruz.

missing_values_table(df)

# 'CABIN' değişkeninde çok fazla eksik olduğu için siliyoruz (zaten Bool türetmiştik).
df.drop('CABIN', axis=1, inplace=True)

# 'TICKET' ve 'NAME' değişkenlerinden gerekli bilgileri aldık, ham hallerini siliyoruz.
remove_cols = ['TICKET', 'NAME']
df.drop(remove_cols, axis=1, inplace=True)

# EKSİK YAŞ DOLDURMA STRATEJİSİ:
# Yaşı eksik olanları direkt ortalama ile doldurmak yerine, kişinin 'Unvanına' (Title) göre dolduruyoruz.
# Yani 'Master' unvanı olan bir çocuğun yaşını, diğer 'Master'ların ortalamasıyla;
# 'Mr' unvanı olan birinin yaşını diğer 'Mr'ların ortalamasıyla dolduruyoruz. Bu çok daha hassas bir yöntemdir.
df['AGE'] = df['AGE'].fillna(df.groupby('NEW_TITLE')['AGE'].transform('median'))


# KRİTİK NOKTA:
# Yukarıda 'AGE' değişkenindeki eksikleri doldurduk. Yani 'AGE' verisi değişti/güncellendi.
# Ancak 1. Bölümde 'NEW_AGE_PCLASS', 'NEW_AGE_CAT' ve 'NEW_SEX_CAT' değişkenlerini
# eksik 'AGE' verisiyle oluşturmuştuk.
# Şimdi bu türetilmiş değişkenleri DOLU 'AGE' verisiyle TEKRAR OLUŞTURMALIYIZ.

df['NEW_AGE_PCLASS'] = df['AGE'] * df['PCLASS']

df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'

df.loc[(df['Sex'] == 'male') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['Sex'] == 'male') & (df['Age'] > 21) & (df['Age'] <= 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['Sex'] == 'male') & (df['Age'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['Sex'] == 'female') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] > 21) & (df['Age'] <= 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'

# Kategorik olup (Object type) 10'dan az sınıfı olan değişkenlerdeki eksikleri 'Mod' (en çok tekrar eden) ile dolduruyoruz.
# (Örn: Embarked sütunu)
df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == 'O' and len(x.unique()) <= 10) else x, axis=0)


##############################
# 4. Label Encoding
##############################
# İki sınıflı kategorik değişkenleri (Binary) 0 ve 1'e çeviriyoruz.
# Örn: Sex (Male/Female) -> 1/0

binary_cols = [col for col in df.columns if df[col].dtypes not in [int, float] and  df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)


##############################
# 5. Rare Encoding
##############################
# Veri setinde çok az frekansa sahip (Örn: %1'den az görülen) kategorileri birleştiriyoruz.
# "Rare" (Nadir) olarak etiketleyerek gürültüyü azaltıyoruz.

rare_analyser(df, "SURVIVED", cat_cols)

# %1'in altında kalan sınıfları 'Rare' çatısı altında topluyoruz.
df = rare_encoder(df, 0.01)

df['NEW_TITLE'].value_counts()


##############################
# 6. One-Hot Encoding
##############################
# Sınıflar arası büyüklük küçüklük ilişkisi olmayan kategorik değişkenleri
# (Nominal) 0 ve 1'lerden oluşan sütunlara çeviriyoruz.
# Örn: Embarked_S, Embarked_C, Embarked_Q

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

df = one_hot_encoder(df, ohe_cols)

df.head()
df.shape

# Encoding sonrası sütunlar değiştiği için değişkenleri tekrar yakalıyoruz.
cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if "PASSENGERID" not in col]

rare_analyser(df, "SURVIVED", cat_cols)

# Gereksiz Değişken Temizliği:
# Bir değişkenin %99'u aynı değerse, o değişkenin modele bir katkısı yoktur.
# Bu tarz "işe yaramaz" (useless) sütunları tespit edip siliyoruz.
useless_cols = [col for col in df.columns if df[col].nunique() == 2 and (df[col].value_counts() / len(df) < 0.01).any(axis=None)]

# df.drop(useless_cols, axis=1, inplace=True) # İsteğe bağlı, şu an yorumda.


##############################
# 7. Standard Scaler
##############################
# Sayısal değişkenlerin ölçeklerini (scale) birbirine yaklaştırıyoruz.
# Age (0-80) ile Fare (0-500) değişkenlerini standartlaştırarak (Mean=0, Std=1)
# modelin bir değişkene diğerinden daha fazla ağırlık vermesini engelliyoruz.

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df[num_cols].head()

df.head()
df.shape


##############################
# 8. Model
##############################
# Verimiz artık temiz, türetilmiş ve makine öğrenmesine hazır.

y = df["SURVIVED"]
x = df.drop(["SURVIVED", "PASSENGERID"], axis=1)

# Veriyi Eğitim (%70) ve Test (%30) olarak bölüyoruz.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=17)

from sklearn.ensemble import RandomForestClassifier

# RandomForest modelini kuruyoruz.
rf_model = RandomForestClassifier(random_state=46).fit(x_train, y_train)
y_pred = rf_model.predict(x_test)

# Modelin başarısını (Accuracy) ölçüyoruz.
accuracy_score(y_test, y_pred)

##############################
# Hiç bir işlem yapılmadan elde edilecek skor ?
##############################
# Feature Engineering yapmasaydık sonuç ne olurdu? (Base Model)
# Bu kısım, yukarıda harcadığımız eforun karşılığını görüp görmediğimizi test etmek içindir.

dff = load()
dff.dropna(inplace=True) # Eksikleri direkt siliyoruz (Basit yaklaşım)
dff = pd.get_dummies(dff, columns=['Sex', 'Embarked'], drop_first=True) # Basit encoding
y = dff["Survived"]
X = dff.drop(["Survived", "PassengerId", "Name", 'Ticket', 'Cabin'], axis=1) # Türetme yok, silme var.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_test, y_pred) # Muhtemelen yukarıdaki skordan daha düşük çıkacaktır.

# Yeni ürettiğimiz değişkenler ne alemde ?
# Hangi değişken model için daha önemli olmuş?

def plot_importance(model, features, num=len(x), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

# Modelin özellik önem düzeylerini görselleştiriyoruz.
# Burada 'NEW_TITLE', 'NEW_SEX_CAT' gibi bizim ürettiğimiz değişkenlerin üst sıralarda olması,
# Feature Engineering işleminin başarısını gösterir.
plot_importance(rf_model, X_train)
