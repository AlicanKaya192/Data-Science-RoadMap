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
df.isnull().value.any()  # <- Bu satırda küçük bir hata var; doğrusu aşağıda

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











