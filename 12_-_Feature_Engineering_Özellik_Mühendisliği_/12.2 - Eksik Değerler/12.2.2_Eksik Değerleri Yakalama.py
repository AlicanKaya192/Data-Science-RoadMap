####################################
# Missing Values (Eksik Değerler)
####################################

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

####################################
# Eksik Değerlerin Yakalanması
####################################

df = load()
df.head()

# eksik gözlem var mı yok mu sorgusu
df.isnull().values.any()

# değişkenlerdeki eksik değer sayısı
df.isnull().sum()

# değişkenlerdeki tam değer sayısı
df.notnull().sum()

# veri setindeki toplam eksik değer sayısı
df.isnull().sum().sum()
# 1 satır, 1 hücerede bile eksiklik var ise onu da sayacaktır.
# Bundan dolayı, kendisinde en az 1 tane eksik hücre olan satır sayısıdır bu.

# en az 1 tane eksik değere sahip olan gözlem birimleri
df[df.isnull().any(axis=1)]

# tam olan gözlem birimleri
df[df.notnull().all(axis=1)]

# Azalan şekilde sıralamak
df.isnull().sum().sort_values(ascending=False)

# Güzel ama ben bu eksikliğin bütün veri setindeki oranını bilmiyorum. Bu orana göre nasıl hareket edebilirim ? diyebiliriz.
(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)
# Her bir değişkendeki eksikliği ifade eden frekansları veri setinin toplam gözlem sayısına böleceğiz. Daha sonra yüzdelik
# olarak görebilmek için 100 ile çarpacağız.

# Sadece eksik değere sahip değişkenlerin isimlerini acaba yakalayabilir miyim ?
na_cols = [col for col in df.columns if df[col].isnull().sum() > 0]

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

missing_values_table(df)


####################################
# Eksik Değer Problemini Çözme
####################################

missing_values_table(df)

# Eğer, ağaca dayalı yöntemler kullanılıyorsa bu durumda eksik değerler tıpkı aykırı değerlerdeki gibi etkisi göz ardı
# eilebilir durumlardır.

# Bir istisna var. Eğer ilgilendiğimiz problem regresyon problemi ise ve bağımlı değişken de dolayısıyla sayısal bir
# değişken ise bu durumda orada aykırılık olması durumunda sonuca gitme süresi biraz uzayabilir.

# Toparlayacak olursak özetle eksik değer ve aykırı değer problemlerine genel hatları itibariyle çok iyi hakim olmak lazım.
# Ama nerede ne kadar etkili olduğunu da iyi bilmek lazım.

# Doğrusal yöntemlerde ve gradient descent temelli yöntemlerde bizim için teknikler çok daha hassasken ağaca dayalı yöntemlerde
# etkisi çok daha düşüktür.


####################################
# 1.Çözüm: Hızlıca Silmek
####################################

df.dropna().shape
# dropna ile hızlıca eksik değerleri silebiliriz ama bu durumda değişken sayısı daha doğrusu gözlem sayısı azalacaktır.
# Çünkü bir satırda en az 1 tane bile eksik değer varsa dropna() onları siliyor olacaktır. Dolayısıyla dropna() kullanırken
# buna dikkat etmek lazım.

# Elimizdeki veri setinde eğer eksik değer problemimiz varsa ve eğer gözlem sayımız çok fazlaysa mesela; 500.000, 1 milyon,
# 3 milyon, 8 milyon vesaire gibi. Bu durumda direkt silmek tercih edilebilir veri boyutu yeterli olduğundan dolayı.


####################################
# 2.Çözüm: Basit Atama Yöntemleri ile Doldurmak
####################################

# Örneğin; yaştaki eksiklikleri ortalamasıyla ya da medyanıyla doldurabileceğimiz gibi herhangi sabit bir değer ile de
# doldurabiliriz.

# Mesela yaş değişkenini onun ortalamasıyla doldurmak istediğimizi düşünelim.
df["Age"].fillna(df["Age"].mean()).isnull().sum()
df["Age"].fillna(df["Age"].median()).isnull().sum()
df["Age"].fillna(0).isnull().sum()

df.apply(lambda x: x.fillna(x.mean()), axis=0) # Bu hatalı olacaktır. Çünkü kategorik değişkenlere denk gelecektir.
df.apply(lambda x: x.fillna(x.median()) if x.dtype != "0" else x, axis=0).head()
# İlgili değişkenin tipi objectden farklı ise NA leri median a göre doldurmasını istedik.

dff = df.apply(lambda x: x.fillna(x.median()) if x.dtype != "0" else x, axis=0)
dff.isnull().sum().sort_values(ascending=False)

# Kategorik değişkenlerdeki eksikliklere ne yapacağız ?
# DİKKAT!!! - Kategorik değişkenler için en mantıklı görülebilecek doldurma yöntemlerinden birisi mod almaktır.

df["Embarked"].fillna(df["Embarked"].mode()[0]).isnull().sum()
# mode()'un string karşılığını yani value suna erişmek için 0.index'i getir diyoruz.

# Eğer dilersek herhangi özel bir ifadeyi de kullanabiliriz. Mesela;
df["Embarked"].fillna("missing")

# Peki bunu otomatik olarak nasıl yapacağız ?

df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis = 0).isnull().sum()


####################################
# Kategorik Değişken Kırılımda Değer Atama
####################################

# Şimdi, veri setinde var olan bazı kategorik değişkenleri kırılım olarak ele almak
# ve bu kırılımlar sonucunda eksik değerleri doğru şekilde doldurmak istiyoruz.
# "Bu ne demek?"
# Örneğin: Yaş değişkeninde eksik değerler var. Bu eksikleri herkes için tek bir ortalamayla
# doldurmak yerine, kadınların yaşını kadın yaş ortalamasına, erkeklerin yaşını erkek yaş ortalamasına göre dolduracağız.
# Böylece veri seti daha gerçekçi ve anlamlı hale gelir.

# Cinsiyete göre yaş ortalamalarını hesapla (kadın ve erkek ayrı ayrı).
df.groupby("Sex")["Age"].mean()

# Veri setindeki genel yaş ortalamasını hesapla.
df["Age"].mean()

# Eksik yaşları, kişilerin cinsiyetine göre ilgili yaş ortalamasıyla doldur.
# Ardından kalan eksik değer sayısına bak.
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()

# Kadınların yaş ortalamasını getir.
df.groupby("Sex")["Age"].mean()["female"]

# Yaşı eksik olan kadın yolcuların yaşlarını kadın yaş ortalamasıyla doldur.
df.loc[(df["Age"].isnull()) & (df["Sex"] == "female"), "Age"] = df.groupby("Sex")["Age"].mean()["female"]

# Yaşı eksik olan erkek yolcuların yaşlarını erkek yaş ortalamasıyla doldur.
df.loc[(df["Age"].isnull()) & (df["Sex"] == "male"), "Age"] = df.groupby("Sex")["Age"].mean()["male"]


####################################
# Çözüm 3: Tahmine Dayalı Atama İle Doldurma
####################################

# DİKKAT!!! - Bu kısım makine öğrenmesi ile doldurma konusunu ele alır. Eğer makine öğrenmesi konuları hakkında fikriniz
# yok ise buraya daha sonra bakabilir ve ya şimdi bakarak fikir edinebilirsiniz.

# Şimdi bir makine öğrenmesi yöntemiyle tahmine dayalı bir şekilde modelleme işlemi gerçekleştireceğiz. "Nasıl ?"

# Eksikliğe sahip olan değişkeni bağımlı değişken, diğer değişkenleri bağımsız değişkenler gibi kabul edip bir modelleme
# işlemi gerçekleştireceğiz ve modelleme işlemine göre eksik değerlere sahip olan noktaları tahmin etmeye çalışacağız.
# Fakat burada birkaç kritik konu olacak.

# 1. Kategorik değişkenleri one hot encoder a sokmamız lazım, yani bir modelleme tekniği kullanacak olduğumuzdan dolayı
# bu modelin bizden değişkenleri beklediği bir standart var. Bundan dolayı bu standarda uymamız gerekmekte.

# 2. KNN uzaklık temelli bir algoritma olduğundan dolayı değişkenleri standartlaştırmamız lazım ve şimdi dataframe i tekrar
# okuyalım.

df = load()

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


# Bu fonksiyon bize lazım. NEDEN ?. Kategorik değişkenleri yakaladık, numerik değişkenleri yakaladık, PassengerId istemiyoruz.
# Kardinal değişkenleri yakaladık. Bu numerik değişkenleri içerisinden PassengerId'yi çıkaralım.

cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in "PassengerId"]

# Şimdi buradaki cat_cols 'lara bir dönüşüm işlemi yapmamız lazım.
# Bunları encoder dan geçirmemiz lazım, yani label encoding işlemi ya da one hot encoding işlemi yapmamız lazım.

# Label encoding işlemini ve one hot encoding işlemini aynı anda yapabilmek için one hot encoder'ı uygulayacağız.
# get_dummies() metotunu kullanabiliriz.

dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)

# Bunun drop_first = True argümanını bu şekilde ayarlarsak bu durumda iki sınıfa sahip olan kategorik değişkenlerin
# ilk sınıfını atacak, ikinci sınıfını tutacak. Böylece elimizde örneğin; cinsiyet gibi; male, female bir kategorik
# değişken olduğunda bu kategorik değişkeni de binary bir şekilde temsil edebiliyor olacağız.

# Burada özetle yaptığımız işlem kategorik değişkenleri iki sınıflı ya da daha fazla sayıda sınıfa sahip olan kategorik
# değişkenleri numerik bir şekilde ifade etmek.

# get_dummies() metot'u bütün değişkenleri birlikte versek dahi sadece kategorik değişkenlere bir dönüşüm uygulamaktadır.
# Dolayısıyla kullanacak olduğumuz değişkenleri bir araya getirmeyi tercih ediyoruz.

dff.head()

# Şu anda, kullanacak olduğumuz makine öğrenmesi yönteminin anlayacağı dilden veriyi dönüştürdük. "Cinsiyet erkek mi,
# kadın mı ?", "Embarked_Q mu ?, Embarked_S mi ?" şeklinde kategorik değişkenleri dönüştürdük. İki sınıflı olanlar hala
# iki sınıflı bunlarda bir değişiklik yok.

# Bir diğer ihtiyacımız değişkenlerin standartlaştırılması gerekmesi ihtiyacı.

scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
dff.head()

# KNN'in uygulanması.
from sklearn.impute import KNNImputer

# Bu, bize makine öğrenmesi yöntemiyle tahmine dayalı bir şekilde eksik değerleri doldurma imkanı sağlayacaktır. Model
# nesnesini oluşturuyoruz ve komşuluk sayısını 5 yapıyoruz.

imputer = KNNImputer(n_neighbors=5)

# KNN Yöntemi Nasıl Çalışır ?
# "Bana arkadaşını söyle, sana kim olduğunu söyleyeyim" der.

dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()
# fit_transform() dediğimizde ilgili dataframe 'e 5 komşuluk özellikli bu imputer nesnesi çalıştırılmış, uygulanmış olacak.
# Uygulandıkdan sonra elde edilecek olan format hoşumuza gitmeyecek bir format olduğundan dolayı bunu dataframe 'e geri
# çevirip bu şekilde gösterebiliyoruz.

dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)
# inverse_transform() diyerek scaler nesnesinin daha önce MinMaxScaler 'ı tutan nesnenin içerisinde ilgili dönüştürme
# bilgisi olduğundan dolayı geriye dönüştürüyoruz.

df["age_imputed_knn"] = dff["Age"]

df.loc[df["Age"].isnull(), ["Age", "age_imputed_knn"]]
df.loc[df["Age"].isnull()]


####################################
# Gelişmiş Analizler
####################################

####################
# Eksik Veri Yapısının İncelenmesi
####################

# Veri Setindeki tam olan gözlemlerin sayılarını verir.
msno.bar(df)
plt.show()

# Değişkenlerdeki eksikliklerin bir arada çıkıp çıkmama durumunu incelemek için görsel bir araçtır.
msno.matrix(df)
plt.show()

# Eksiklikler üzerine kurulu ısı haritası. Bize nullity correlation değerlerini verir.
msno.heatmap(df)
plt.show()


####################################
# Eksik Değerlerin Bağımlı Değişken İle İlişkisinin İncelenmesi
####################################

# Eksik değer içeren kolonların listesini alıyorum.
# (missing_values_table zaten eksik sayıları ve oranlarını hesaplıyor.)
na_cols = missing_values_table(df, True)


def missing_vs_target(dataframe, target, na_columns):
    # Orijinal df’i değiştirmemek için bir kopya üzerinden ilerliyorum.
    temp_df = dataframe.copy()

    # Eksik değerin hedef değişkene etkisini incelemek için,
    # her eksik kolon için bir flag oluşturuyorum.
    # Kolonda eksik varsa 1, yoksa 0 şeklinde.
    for col in na_columns:
        temp_df[col + "_NA_FLAG"] = np.where(temp_df[col].isnull(), 1, 0)

    # Az önce eklediğim tüm flag kolonlarını seçiyorum.
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    # Her flag için target ortalamasına bakacağım.
    # Böylece “bu kolon eksik olduğunda hedef değişken nasıl değişiyor?” sorusuna cevap bulacağım.
    for col in na_flags:
        print(
            pd.DataFrame({
                "TARGET_MEAN": temp_df.groupby(col)[target].mean(),  # eksik-eksik değil için hedef ort.
                "Count": temp_df.groupby(col)[target].count()        # her grubun kaç gözlem içerdiği
            }),
            end="\n\n\n"
        )

# Fonksiyonu çalıştırıyorum.
# Burada Survived ile eksiklik durumlarının ilişkisini incelemiş oluyorum.
missing_values_table(df, "Survived", na_cols)