import numpy as np
import seaborn as sns
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

#########################################
# Görev 1: Seaborn kütüphanesi içerisinden Titanic veri setini tanımlayınız.
#########################################

df = pd.read_csv("Datasets ( Genel )/titanic.csv")
df.head()
df.shape

#########################################
# Görev 2: Yukarıda tanımlanan Titanic veri setindeki kadın ve erkek yolcuların sayısını bulunuz.
#########################################

df["sex"].value_counts()

#########################################
# Görev 3: Her bir sutuna ait unique değerlerin sayısını bulunuz.
#########################################

df.nunique()

#########################################
# Görev 4: pclass değişkeninin unique değerleri bulunuz.
#########################################

df.loc[:, "pclass"].unique()

df["pclass"].unique()

#########################################
# Görev 5:  pclass ve parch değişkenlerinin unique değerlerinin sayısını bulunuz.
#########################################

df.loc[:, ["pclass", "parch"]].nunique()

#########################################
# Görev 6: embarked değişkeninin tipini kontrol ediniz. Tipini category olarak değiştiriniz. Tekrar tipini kontrol ediniz.
#########################################

df["embarked"].dtype
str(df["embarked"].dtype)
df["embarked"] = df["embarked"].astype("category")
str(df["embarked"].dtype)

# OBJECT
# Out[16]: array(['S', 'C', 'Q', nan], dtype=object)

# CATEGORY
# ['S', 'C', 'Q', NaN]
# Categories (3, object): ['C', 'Q', 'S']
# df.loc[df["embarked"] == "S", "embarked"] = "A"  # değiştirmeye çalışmak hata verecektir

#########################################
# Görev 7: embarked değeri C olanların tüm bilgilerini gösteriniz.
#########################################

# df[df["embarked"] == "C"]
df.loc[df["embarked"] == "C"]

########
# BONUS
########
df.loc[df["embarked"].str.contains("c", na=False, case=False)]

df.loc[(df["embarked"] == "C") | (df["embarked"] == "S")].head()
df.loc[df["embarked"].isin(["C", "S"])].head()

#########################################
# Görev 8: embarked değeri S olmayanların tüm bilgilerini gösteriniz.
#########################################

# df[df["embarked"] != "S"]
df.loc[df["embarked"] != "S"]

# 2. YOL
# df[~(df["embarked"] == "S")]
df.loc[~(df["embarked"] == "S")]

# df.loc[ ~(df["embarked"].str.contains("S", na=False)) ]

#########################################
# Görev 9: Yaşı 30 dan küçük ve kadın olan yolcuların tüm bilgilerini gösteriniz.
#########################################

df.loc[(df["age"] < 30) & (df["sex"] == "female")].head()

#########################################
# Görev 10: Fare'i 500'den büyük veya yaşı 70 den büyük yolcuların bilgilerini gösteriniz.
#########################################

df.loc[(df["fare"] > 500) | (df["age"] > 70)].head()

#########################################
# Görev 11: Her bir değişkendeki boş değerlerin toplamını bulunuz.
#########################################

df.isnull().sum()

# BONUS
df.loc[df.isnull().any(axis=1)]  # herhangi bir satırda eksik değer varsa gelir

# notnull => 1 tane de olsa dolu değer varsa satırı getirir
# (satır nasıl geliyor => axis=1) => axis=0 deseydik sütun olurdu
# df.loc[df.notnull().all(axis=1)]  # => tüm satırlar doluysa getir
# df.loc[df.notnull().any(axis=1)]  # => bir değer bile doluysa getir

#########################################
# Görev 12: who değişkenini dataframe'den düşürün.
#########################################

df.drop("who", axis=1, inplace=True)
# df = df.drop("who", axis=1)  # inplace=True'nun yaptığı

#########################################
# Görev 13: deck değikenindeki boş değerleri deck değişkenin en çok tekrar eden değeri (mode) ile doldurunuz.
#########################################

df["deck"].mode()
type(df["deck"].mode())
df["deck"].mode()[0]  # çıktısı pd.series
type(df["deck"].mode()[0])  # str

df["deck"].fillna(df["deck"].mode()[0], inplace=True)
# df["deck"] = df["deck"].fillna(df["deck"].mode()[0])  # => inplace=True'nun yaptığı
df["deck"].isnull().sum()
df.deck.tail()

#########################################
# Görev 14: age değikenindeki boş değerleri age değişkenin medyanı ile doldurun.
#########################################

df["age"].fillna(df["age"].median(), inplace=True)
df["age"].isnull().sum()

#########################################
# Görev 15: survived değişkeninin Pclass ve Cinsiyet değişkenleri kırılımınında sum, count, mean değerlerini bulunuz.
#########################################

df.groupby(["Pclass", "sex"]).agg({"survived": ["sum","count","mean"]})

df.pivot_table("survived", ["Pcclass", "sex"], aggfunc=["sum", "count", "mean"])

#########################################
# Görev 16:  30 yaşın altında olanlar 1, 30'a eşit ve üstünde olanlara 0 verecek bir fonksiyon yazınız.
# Yazdığınız fonksiyonu kullanarak titanik veri setinde age_flag adında bir değişken oluşturunuz. (apply ve lambda yapılarını kullanınız)
#########################################

def age_30(age):
    if age < 30:
        return 1
    else:
        return 0

df["age_flag"] = df["age"].apply(lambda x: age_30(x))

# 2. YOL
# fonksiyon tanımlamadan
df["age_flag"] = df["age"].apply(lambda x: 1 if x < 30 else 0)

# # 3. YOL - apply/lambda kullanmadan
# df["age_flag_2"] = np.where(df["age"] < 30, 1, 0)

#########################################
# Görev 17: Seaborn kütüphanesi içerisinden Tips veri setini tanımlayınız.
#########################################
df = sns.load_dataset("tips")
df.head()
df.shape

#########################################
# Görev 18: Time değişkeninin kategorilerine (Dinner, Lunch) göre total_bill  değerlerinin toplamını, min, max ve ortalamasını bulunuz.
#########################################

df.groupby("time").agg({"total_bill": ["sum","min","mean","max"]})

df.pivot_table("total_bill", "time", aggfunc=["min", "max", "mean"])

#########################################
# Görev 19: Günlere ve time göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.
#########################################

df.groupby(["day", "time"]).agg({"total_bill": ["sum","min","mean","max"]})

df.pivot_table(index=["day", "time"], values="total_bill", aggfunc=["sum","min","mean","max"])

# df.loc[(df["day"] == "Sat") & (df["time"] == "Lunch")]  # Empty DataFrame
# df.loc[(df["day"] == "Sun") & (df["time"] == "Lunch")]  # Empty DataFrame

#########################################
# Görev 20:Lunch zamanına ve kadın müşterilere ait total_bill ve tip  değerlerinin day'e göre toplamını, min, max ve ortalamasını bulunuz.
#########################################

df.loc[(df["time"] == "Lunch") & (df["sex"] == "Female")].groupby("day").agg({"total_bill": ["sum","min","max","mean"],
                                                                           "tip": ["sum","min","max","mean"]})

#########################################
# Görev 21: size'i 3'ten küçük, total_bill'i 10'dan büyük olan siparişlerin ortalaması nedir?
#########################################

df.loc[(df["size"] < 3) & (df["total_bill"] > 10), "total_bill"].mean()  # 17.184965034965035
# df[(df["size"] < 3) & (df["total_bill"] > 10)]["total_bill"].mean()

#########################################
# Görev 22: total_bill_tip_sum adında yeni bir değişken oluşturun. Her bir müşterinin ödediği totalbill ve tip in toplamını versin.
#########################################

df["total_bill_tip_sum"] = df["total_bill"] + df["tip"]
df.head()

#########################################
# Görev 23: total_bill_tip_sum değişkenine göre büyükten küçüğe sıralayınız ve ilk 30 kişiyi yeni bir dataframe'e atayınız.
#########################################

new_df = df.sort_values("total_bill_tip_sum", ascending=False)[:30]
new_df.shape
new_df.head()

# 2. YOL
new_df = df.sort_values("total_bill_tip_sum", ascending=False).head(30)
new_df.shape



#########################################
# BONUS 1

# Total_bill değişkeninin kadın ve erkek için ayrı ayrı ortalamasını bulun.
# Bulduğunuz ortalamaların altında olanlara 0, üstünde ve eşit olanlara bir verildiği yeni bir total_bill_flag değişkeni oluşturun.
# Dikkat !! Female olanlar için kadınlar için bulunan ortalama dikkate alıncak, male için ise erkekler için bulunan ortalama.
# parametre olarak cinsiyet ve total_bill alan bir fonksiyon yazarak başlayın. (If-else koşulları içerecek)
#########################################

# total bill'in kadınlar ve erkekler için ortalaması

f_avg = df[df["sex"]=="Female"]["total_bill"].mean() # 18.056896551724133
m_avg = df[df["sex"]=="Male"]["total_bill"].mean() # 20.744076433121016

def func(sex,total_bill):
    if sex=="Female":
        if total_bill < f_avg:
            return 0
        else:
            return 1
    else:
        if total_bill < m_avg:
            return 0
        else:
            return 1

df["total_bill_flag"] = df[["sex","total_bill"]].apply(lambda x: func(x["sex"],x["total_bill"]),axis=1)


#########################################
# BONUS 2

# total_bill_flag değişkenini kullanarak cinsiyetlere göre ortalamanın altında ve üstünde olanların sayısını gözlemleyin.
#########################################
df.groupby(["sex","total_bill_flag"]).agg({"total_bill_flag":"count"})





#################
# Agg. örnekler
################
import numpy as np
import pandas as pd
df = pd.DataFrame({'Animal': ['Falcon', 'Falcon', "Falcon","Falcon",
                               'Parrot', 'Parrot'],
                   'Max Speed': [380., np.nan, np.nan, 45,24., 26.]})

# agg_df = df.groupby("Animal")["Max Speed"].mean().reset_index()
# agg_df.columns = ["animal", "avg_max_speed"]
# agg_df = agg_df.rename({"Max Speed":"avg_max_speed"}, axis=1)
# df.groupby("Animal").agg(avg_max_speed=("Max Speed", "mean")).reset_index()



# dropna argümanının farkı
import pandas as pd
import numpy as np

# Örnek veri oluşturma
data = {
    'kategori': ['A', 'B', 'A', np.nan, 'B', 'C', np.nan],
    'deger': [10, 20, 30, 40, np.nan, 60, 100]
}

df = pd.DataFrame(data)

# dropna=True örneği
grouped_dropna_true = df.groupby('kategori', dropna=True).agg(
    toplam_deger=('deger', 'sum'),
    ortalama_deger=('deger', 'mean')
)

# dropna=False örneği
grouped_dropna_false = df.groupby('kategori', dropna=False).agg(
    toplam_deger=('deger', 'sum'),
    ortalama_deger=('deger', 'mean')
)

print("dropna=True ile gruplandırma:")
print(grouped_dropna_true)
print("\ndropna=False ile gruplandırma:")
print(grouped_dropna_false)
