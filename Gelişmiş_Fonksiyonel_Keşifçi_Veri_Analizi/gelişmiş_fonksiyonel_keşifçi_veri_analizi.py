###############################################
# GELİŞMİŞ FONKSİYONEL KEŞİFÇİ VERİ ANALİZİ (ADVANCED FUNCTIONAL EDA)
###############################################
# 1. Genel Resim
# 2. Kategorik Değişken Analizi (Analysis of Categorical Variables)
# 3. Sayısal Değişken Analizi (Analysis of Numerical Variables)
# 4. Hedef Değişken Analizi (Analysis of Target Variable)
# 5. Korelasyon Analizi (Analysis of Correlation)


###############################################
# 1. Genel Resim
###############################################
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

# Tüm sütunların çıktıda görünmesini sağlar
pd.set_option('display.max_columns', None)

# Ekranda veri çerçevesinin geniş görünmesi için satır genişliğini artırır
pd.set_option('display.width', 500)

# Titanic veri setini seaborn kütüphanesinden yükle
df = sns.load_dataset("titanic")

# Veri setinin ilk 5 gözlemini gösterir (veri yapısına genel bakış)
df.head()

# Veri setinin son 5 gözlemini gösterir (alt kısımdaki veri yapısını da görmeye yarar)
df.tail()

# Veri setinin boyutlarını döndürür → (satır sayısı, sütun sayısı)
df.shape

# Veri setine ait genel bilgileri gösterir: sütun adları, veri tipleri, eksik değer sayısı
df.info()

# Tüm sütun isimlerini listeler
df.columns

# Veri çerçevesinin indeks bilgilerini gösterir
df.index

# Sayısal değişkenlere ilişkin temel istatistikleri verir (ortalama, std, min, max, vs.)
# .T ile çıktıyı transpoze ederek daha okunabilir hale getirir (sütunlar satır olur)
df.describe().T

# Veri setinde eksik değer olup olmadığını kontrol eder (True/False döner)
df.isnull().values.any()  # <- Bu satırda küçük bir hata var; doğrusu aşağıda

# Her sütunda kaç adet eksik değer olduğunu gösterir
df.isnull().sum()



def check_df(dataframe, head=5):
    print("############# SHAPE #############")
    print(dataframe.shape)
    print("############# Types #############")
    print(df.dtypes)
    print("############# Head #############")
    print(df.head(head))
    print("############# Tail #############")
    print(df.tail(head))
    print("############# NA #############")
    print(df.isnull().sum())
    print("############# Quantiles #############")
    print(df.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

df = sns.load_dataset("tips")
check_df(df)


###############################################
# 2. Kategorik Değişken Analizi (Analysis of Categorical Variables)
###############################################
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# Görseldeki sütunların tam görünmesi için ayar
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

# Titanic veri setini yükle
df = sns.load_dataset("titanic")

# İlk 5 gözlemi görüntüle
df.head()

# 'embarked' (binilen liman) değişkeninin frekanslarını göster
df["embarked"].value_counts()

# 'sex' (cinsiyet) değişkeninin benzersiz değerlerini göster
df["sex"].unique()

# 'class' (bilet sınıfı) değişkeninin kaç farklı kategoriye sahip olduğunu göster
df["class"].nunique()

# Kategorik değişkenleri seç:
# Türü object, category veya bool olan sütunları listele
cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["object", "category", "bool"]]

# Sayısal görünümlü ama kategorik olan değişkenleri seç:
# (örnek: 0-1 veya 1-2-3 gibi az sayıda farklı değer alan sayısal sütunlar)
num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int", "float"]]

# Kategorik görünümlü ama kardinal (çok fazla sınıfa sahip) değişkenleri seç:
# (örnek: isim, ID, benzersiz kod gibi değişkenler)
cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["object", "category"]]

# Kategorik değişken listesine (cat_cols),
# sayısal görünümlü ama aslında kategorik olan değişkenleri (num_but_cat) de ekle.
# Böylece tüm kategorik özellikler tek listede toplanmış olur.
cat_cols = cat_cols + num_but_cat

# Kategorik değişken listesinden (cat_cols),
# 'kardinal' yani çok fazla sınıfa sahip olan değişkenleri (cat_but_car) çıkar.
# Böylece cat_cols listesi yalnızca analizde anlamlı olan "gerçek" kategorik değişkenleri içerir.
cat_cols = [col for col in cat_cols if col not in cat_but_car]

# Veri setinden sadece kategorik değişkenleri (cat_cols listesindekileri) seçer.
# Yani df veri çerçevesinin sadece kategorik sütunlarını içeren bir alt veri çerçevesi döner.
df[cat_cols]

# Kategorik değişkenlerin (cat_cols listesindeki sütunların)
# her birinde kaç farklı (benzersiz) kategori bulunduğunu gösterir.
df[cat_cols].nunique()

# Veri setindeki (df.columns) tüm sütunlar arasından,
# kategorik değişkenler listesinde (cat_cols) yer almayanları seçer.
# Yani bu işlem sonucunda sayısal (numeric) değişkenlerin listesini elde ederiz.
[col for col in df.columns if col not in cat_cols]

# 'survived' değişkeninin (hayatta kalma durumu) frekans dağılımını gösterir.
# Kaç kişinin hayatta kaldığını (1) ve kaç kişinin kalmadığını (0) sayar.
df["survived"].value_counts()

# 'survived' değişkenindeki değerlerin yüzde dağılımını hesaplar.
# Her sınıfın (0 veya 1) toplam içindeki oranını yüzdelik olarak gösterir.
100 * df["survived"].value_counts() / len(df)

# Kategorik değişkenler için özet bilgi (frekans ve oran) oluşturan fonksiyon
def cat_summary(dataframe, col_name):
    # Belirtilen sütundaki (col_name) her bir kategorinin sayısını (frekansını) hesaplar.
    # Ardından, her kategorinin toplam gözlem içindeki oranını (yüzdesini) bulur.
    # Bu iki bilgiyi tek bir DataFrame içinde birleştirip ekrana yazdırır.
    print(pd.DataFrame({
        col_name: dataframe[col_name].value_counts(),                                # Kategori frekansları
        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)           # Kategori yüzdeleri
    }))
    print("################################")

# 'sex' sütunu için cat_summary fonksiyonunu çalıştırır.
# Yani cinsiyet değişkenindeki her bir kategori (male, female)
# kaç kez geçtiğini ve toplam içindeki yüzdelik oranını gösterir.
cat_summary(df, "sex")

# Kategorik değişkenler listesinde (cat_cols) yer alan her bir sütun için
# cat_summary fonksiyonunu çalıştır.
# Böylece tüm kategorik sütunların frekans ve yüzdelik dağılımlarını topluca görebiliriz.
for col in cat_cols:
    cat_summary(df, col)


# Kategorik değişkenler için özet ve isteğe bağlı görselleştirme yapan fonksiyon
def cat_summary(dataframe, col_name, plot=False):
    # 1️⃣ Tablo: Kategori frekansları ve yüzdelerini hesapla ve yazdır
    print(pd.DataFrame({
        col_name: dataframe[col_name].value_counts(),  # Kategori frekansları
        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)  # Kategori yüzdeleri
    }))

    # Ayırıcı çizgi ile çıktı daha okunabilir olur
    print("################################")

    # 2️⃣ Eğer plot=True verilmişse, kategori dağılımını görselleştir
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)  # Grafiğin hemen gösterilmesini sağlar


# 1️⃣ 'sex' sütunu için kategorik özet ve görselleştirme
cat_summary(df, "sex", plot=True)

# 2️⃣ Tüm kategorik değişkenler için döngü
for col in cat_cols:
    # Eğer sütun veri tipi boolean ise, sadece "bool" yazdır
    if df[col].dtypes == "bool":
        print("bool")
    # Boolean değilse, cat_summary fonksiyonunu çalıştır ve grafiği göster
    else:
        cat_summary(df, col, plot=True)


# 'adult_male' sütununu boolean (True/False) tipinden integer (0/1) tipine çevirir
# Bu, görselleştirme veya modelleme öncesi sayısal işlem yapılabilmesi için gerekebilir
df["adult_male"].astype(int)

# Tüm kategorik değişkenler üzerinde döngü
for col in cat_cols:
    # Eğer sütun boolean ise
    if df[col].dtypes == "bool":
        # Boolean değerleri integer'a dönüştür (True->1, False->0)
        df[col] = df[col].astype(int)
        # Dönüştürülen sütun için kategorik özet ve grafiği göster
        cat_summary(df, col, plot=True)
    else:
        # Boolean olmayan kategorik sütunlar için sadece özet ve grafiği göster
        cat_summary(df, col, plot=True)


def cat_summary(dataframe, col_name, plot=False):
    # 1️⃣ Eğer sütun boolean tipindeyse
    if dataframe[col_name].dtypes == "bool":
        # Boolean -> int (True=1, False=0) dönüşümü
        dataframe[col_name] = dataframe[col_name].astype(int)

        # Kategori frekans tablosu ve yüzdeleri
        print(pd.DataFrame({
            col_name: dataframe[col_name].value_counts(),
            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)
        }))

        print("################################")

        # Eğer plot=True verilmişse, countplot grafiği oluştur
        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)

    # 2️⃣ Boolean olmayan sütunlar için
    else:
        # Burada da sütunu int yapıyorsun; bu genellikle sayısal kodlama amaçlı
        dataframe[col_name] = dataframe[col_name].astype(int)

        # Frekans tablosu ve yüzdeleri
        print(pd.DataFrame({
            col_name: dataframe[col_name].value_counts(),
            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)
        }))

        print("################################")

        # Grafiği göster
        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)

cat_summary(df, "adult_male", plot=True)


###############################################
# 3. Sayısal Değişken Analizi (Analysis of Numerical Variables)
###############################################
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()


# Kategorik değişkenleri seç:
# Türü object, category veya bool olan sütunları listele
cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["object", "category", "bool"]]

# Sayısal görünümlü ama kategorik olan değişkenleri seç:
# (örnek: 0-1 veya 1-2-3 gibi az sayıda farklı değer alan sayısal sütunlar)
num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int", "float"]]

# Kategorik görünümlü ama kardinal (çok fazla sınıfa sahip) değişkenleri seç:
# (örnek: isim, ID, benzersiz kod gibi değişkenler)
cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["object", "category"]]

# Kategorik değişken listesine (cat_cols),
# sayısal görünümlü ama aslında kategorik olan değişkenleri (num_but_cat) de ekle.
# Böylece tüm kategorik özellikler tek listede toplanmış olur.
cat_cols = cat_cols + num_but_cat

# Kategorik değişken listesinden (cat_cols),
# 'kardinal' yani çok fazla sınıfa sahip olan değişkenleri (cat_but_car) çıkar.
# Böylece cat_cols listesi yalnızca analizde anlamlı olan "gerçek" kategorik değişkenleri içerir.
cat_cols = [col for col in cat_cols if col not in cat_but_car]

# Sayısal değişkenler 'age' ve 'fare' için temel istatistikleri hesapla
# .describe() → count, mean, std, min, 25%, 50%, 75%, max değerlerini verir
# .T → transpoze ederek sütunları satır, istatistikleri sütun yapar (daha okunabilir)
df[["age", "fare"]].describe().T

# Veri setindeki sayısal değişkenlerin listesini oluştur
# int veya float veri tipinde olan sütunlar seçilir
num_cols = [col for col in df.columns if str(df[col].dtypes) in ["int", "float"]]

# Sayısal değişkenleri belirleme
# cat_cols listesinde yer almayan tüm sütunlar sayısal değişken olarak kabul edilir
num_cols = [col for col in df.columns if col not in cat_cols]


# Sayısal değişkenler için detaylı özet fonksiyonu
def num_summary(dataframe, numerical_col):
    # Belirlenen yüzdelik değerler (quantiles)
    quantiles = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]

    # describe() fonksiyonuyla temel istatistikleri ve belirlenen yüzdelikleri hesapla
    print(dataframe[numerical_col].describe(quantiles).T)


# Örnek kullanım: 'age' sütunu için sayısal özet
num_summary(df, "age")

# Sayısal sütunlar listesindeki (num_cols) her bir sütun için
# num_summary fonksiyonunu çalıştır.
# Böylece tüm sayısal sütunların istatistiksel özetlerini hızlıca görebiliriz.
for col in num_cols:
    num_summary(df, col)


def num_summary(dataframe, numerical_col, plot=False):
    # 1️⃣ İncelenecek özel yüzdelikler
    quantiles = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40,
                 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]

    # 2️⃣ Temel istatistikler ve belirlenen yüzdelikleri hesapla ve yazdır
    print(dataframe[numerical_col].describe(quantiles).T)

    # 3️⃣ Eğer plot=True ise, histogram grafiği çiz
    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)  # x ekseni etiketi
        plt.title(numerical_col)  # grafiğe başlık
        plt.show(block=True)  # grafiğin hemen görünmesini sağlar


num_summary(df, "age", plot=True)


# Sayısal sütunlar listesindeki (num_cols) her bir sütun için
# num_summary fonksiyonunu çalıştır, hem tablo hem histogram grafiği göster
for col in num_cols:
    num_summary(df, col, plot=True)



###############################################
# Değişkenlerin Yakalanması ve İşlemlerin Genelleştirilmesi
###############################################
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()
df.info()


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
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["object", "category", "bool"]]

    # 2️⃣ Numerik görünümlü fakat az sayıda farklı değere sahip sütunları seç (numerik ama kategorik)
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and str(dataframe[col].dtypes) in ["int64", "float64"]]

    # 3️⃣ Kategorik ama çok fazla farklı değere sahip sütunları seç (kardinal)
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and str(dataframe[col].dtypes) in ["object", "category"]]

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


help(grab_col_names)

# Fonksiyon: Veri setindeki kategorik, numerik ve kardinal değişken isimlerini döndürür
cat_cols, num_cols, cat_but_car = grab_col_names(df)  # Fonksiyonu çağır ve listeleri al

for col in cat_cols:
    cat_summary(df, col)


def num_summary(dataframe, numerical_col, plot=False):
    # 1️⃣ İncelenecek özel yüzdelikler
    quantiles = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40,
                 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]

    # 2️⃣ Temel istatistikler ve belirlenen yüzdelikleri hesapla ve yazdır
    print(dataframe[numerical_col].describe(quantiles).T)

    # 3️⃣ Eğer plot=True ise, histogram grafiği çiz
    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)  # x ekseni etiketi
        plt.title(numerical_col)  # grafiğe başlık
        plt.show(block=True)  # grafiğin hemen görünmesini sağlar


for col in num_cols:
    num_summary(df, col, plot=True)



# BONUS
df = sns.load_dataset("titanic")

for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

def cat_summary(dataframe, col_name, plot=False):
    # 1️⃣ Eğer sütun boolean tipindeyse
    if dataframe[col_name].dtypes == "bool":
        # Boolean -> int (True=1, False=0) dönüşümü
        dataframe[col_name] = dataframe[col_name].astype(int)

        # Kategori frekans tablosu ve yüzdeleri
        print(pd.DataFrame({
            col_name: dataframe[col_name].value_counts(),
            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)
        }))

        print("################################")

        # Eğer plot=True verilmişse, countplot grafiği oluştur
        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)

    # 2️⃣ Boolean olmayan sütunlar için
    else:
        # Burada da sütunu int yapıyorsun; bu genellikle sayısal kodlama amaçlı
        dataframe[col_name] = dataframe[col_name].astype(int)

        # Frekans tablosu ve yüzdeleri
        print(pd.DataFrame({
            col_name: dataframe[col_name].value_counts(),
            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)
        }))

        print("################################")

        # Grafiği göster
        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)


for col in cat_cols:
    cat_summary(df, col, plot=True)


###############################################
# 4. Hedef Değişken Analizi (Analysis of Target Variable)
###############################################
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")


for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)


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
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["object", "category", "bool"]]

    # 2️⃣ Numerik görünümlü fakat az sayıda farklı değere sahip sütunları seç (numerik ama kategorik)
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and str(dataframe[col].dtypes) in ["int64", "float64"]]

    # 3️⃣ Kategorik ama çok fazla farklı değere sahip sütunları seç (kardinal)
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and str(dataframe[col].dtypes) in ["object", "category"]]

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

df["survived"].value_counts()
cat_summary(df, "survived")

###############################################
# Hedef Değişkenin Kategorik Değişkenler ile Analizi
###############################################


df.groupby("sex")["survived"].mean()


def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}))


target_summary_with_cat(df, "survived", "sex")


for col in cat_cols:
    target_summary_with_cat(df, "survived", col)



###############################################
# Hedef Değişkenin Sayısal Değişkenler ile Analizi
###############################################

df.groupby("survived")["age"].mean()

df.groupby("survived").agg({"age": "mean"})

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


target_summary_with_num(df, "survived", "age")

for col in num_cols:
    target_summary_with_num(df, "survived", col)



###############################################
# 5. Korelasyon Analizi (Analysis of Correlation)
###############################################
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")


