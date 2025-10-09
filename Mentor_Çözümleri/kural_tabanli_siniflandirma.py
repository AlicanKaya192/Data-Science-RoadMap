
#############################################
# Kural Tabanlı Sınıflandırma ile Potansiyel Müşteri Getirisi Hesaplama
#############################################

#############################################
# İş Problemi
#############################################
# Bir oyun şirketi müşterilerinin bazı özelliklerini kullanarak seviye tabanlı (level based) yeni müşteri tanımları (persona)
# oluşturmak ve bu yeni müşteri tanımlarına göre segmentler oluşturup bu segmentlere göre yeni gelebilecek müşterilerin şirkete
# ortalama ne kadar kazandırabileceğini tahmin etmek istemektedir.

# Örneğin: Türkiye’den IOS kullanıcısı olan 25 yaşındaki bir erkek kullanıcının ortalama ne kadar kazandırabileceği belirlenmek isteniyor.


#############################################
# Veri Seti Hikayesi
#############################################
# Persona.csv veri seti uluslararası bir oyun şirketinin sattığı ürünlerin fiyatlarını ve bu ürünleri satın alan kullanıcıların bazı
# demografik bilgilerini barındırmaktadır. Veri seti her satış işleminde oluşan kayıtlardan meydana gelmektedir. Bunun anlamı tablo
# tekilleştirilmemiştir. Diğer bir ifade ile belirli demografik özelliklere sahip bir kullanıcı birden fazla alışveriş yapmış olabilir.

# Price: Müşterinin harcama tutarı
# Source: Müşterinin bağlandığı cihaz türü
# Sex: Müşterinin cinsiyeti
# Country: Müşterinin ülkesi
# Age: Müşterinin yaşı

################# Uygulama Öncesi #####################

#    PRICE   SOURCE   SEX COUNTRY  AGE
# 0     39  android  male     bra   17
# 1     39  android  male     bra   17
# 2     49  android  male     bra   17
# 3     29  android  male     tur   17
# 4     49  android  male     tur   17

################# Uygulama Sonrası #####################

#       customers_level_based        PRICE SEGMENT
# 0   BRA_ANDROID_FEMALE_0_18  1139.800000       A
# 1  BRA_ANDROID_FEMALE_19_23  1070.600000       A
# 2  BRA_ANDROID_FEMALE_24_30   508.142857       A
# 3  BRA_ANDROID_FEMALE_31_40   233.166667       C
# 4  BRA_ANDROID_FEMALE_41_66   236.666667       C


#############################################
# PROJE GÖREVLERİ
#############################################

#############################################
# GÖREV 1: Aşağıdaki soruları yanıtlayınız.
#############################################

# Soru 1: persona.csv dosyasını okutunuz ve veri seti ile ilgili genel bilgileri gösteriniz.
import numpy as np
import pandas as pd
pd.set_option("display.max_rows", None)

df = pd.read_csv('Kural_Tabanli_Siniflandirma/persona.csv')
df.shape
df.head()
df.info()

# Soru 2: Kaç unique SOURCE vardır? Frekansları nedir?
df["SOURCE"].nunique()
df["SOURCE"].value_counts()

# Soru 3: Kaç unique PRICE vardır?
df["PRICE"].nunique()


# Soru 4: Hangi PRICE'dan kaçar tane satış gerçekleşmiş?
df["PRICE"].value_counts()

# Soru 5: Hangi ülkeden kaçar tane satış olmuş?
df["COUNTRY"].value_counts()


# Soru 6: Ülkelere göre satışlardan toplam ne kadar kazanılmış?
df.groupby("COUNTRY")["PRICE"].sum()
# df.groupby("COUNTRY").agg({"PRICE": "sum"})
# df.pivot_table(index="COUNTRY", values="PRICE", aggfunc="sum")

# Soru 7: SOURCE türlerine göre satış sayıları nedir?
df["SOURCE"].value_counts()

# Soru 8: Ülkelere göre PRICE ortalamaları nedir?
df.groupby("COUNTRY")["PRICE"].mean()
# df.groupby(by=['COUNTRY']).agg({"PRICE": "mean"})
# df.pivot_table(index="COUNTRY", values="PRICE", aggfunc="mean")
# df.pivot_table(index="COUNTRY", values="PRICE")  # aynı değeri verir çünkü aggfunc'ın defaultu mean'dir

# Soru 9: SOURCE'lara göre PRICE ortalamaları nedir?
df.groupby(by=['SOURCE']).agg({"PRICE": "mean"})
# df.groupby('SOURCE').agg({"PRICE": "mean"})

# Soru 10: COUNTRY-SOURCE kırılımında PRICE ortalamaları nedir?
df.groupby(by=["COUNTRY", 'SOURCE']).agg({"PRICE": "mean"})
# df.groupby(["COUNTRY", 'SOURCE']).agg({"PRICE": "mean"})


#############################################
# GÖREV 2: COUNTRY, SOURCE, SEX, AGE kırılımında ortalama kazançlar nedir?
#############################################
df.groupby(["COUNTRY", 'SOURCE', "SEX", "AGE"]).agg({"PRICE": "mean"}).head()  # head sadece küçük bir kısmı görmek için
# df.groupby(["COUNTRY", 'SOURCE', "SEX", "AGE"]).agg({"PRICE": "mean"})

# Bonus örnek
# yaşı 25'ten büyük olanların COUNTRY, SOURCE, SEX, AGE kırılımında ortalama kazançları
# df.loc[df["AGE"] > 25].groupby(["COUNTRY", 'SOURCE', "SEX", "AGE"]).agg({"PRICE": "mean"}).head()

#############################################
# GÖREV 3: Çıktıyı PRICE'a göre sıralayınız.
#############################################
# Önceki sorudaki çıktıyı daha iyi görebilmek için sort_values metodunu azalan olacak şekilde PRICE'a uygulayınız.
# Çıktıyı agg_df olarak kaydediniz.
agg_df = df.groupby(by=["COUNTRY", 'SOURCE', "SEX", "AGE"]).agg({"PRICE": "mean"}).sort_values("PRICE", ascending=False)
agg_df.head()


#############################################
# GÖREV 4: Indekste yer alan isimleri değişken ismine çeviriniz.
#############################################
# Üçüncü sorunun çıktısında yer alan PRICE dışındaki tüm değişkenler index isimleridir.
# Bu isimleri değişken isimlerine çeviriniz.
# İpucu: reset_index()

# agg_df.reset_index(inplace=True)
agg_df = agg_df.reset_index()
agg_df.head()


#############################################
# GÖREV 5: AGE değişkenini kategorik değişkene çeviriniz ve agg_df'e ekleyiniz.
#############################################
# Age sayısal değişkenini kategorik değişkene çeviriniz.
# Aralıkları ikna edici olacağını düşündüğünüz şekilde oluşturunuz.
# Örneğin: '0_18', '19_23', '24_30', '31_40', '41_70'


# AGE değişkeninin nerelerden bölüneceğini belirtelim:
bins = [0, 18, 23, 30, 40, agg_df["AGE"].max()]

# Bölünen noktalara karşılık isimlendirmelerin ne olacağını ifade edelim:
mylabels = ['0_18', '19_23', '24_30', '31_40', '41_' + str(agg_df["AGE"].max())]

# age'i bölelim:
agg_df["age_cat"] = pd.cut(agg_df["AGE"], bins, labels=mylabels)
agg_df.head()


#############################################
# GÖREV 6: Yeni seviye tabanlı müşterileri (persona) tanımlayınız ve veri setine değişken olarak ekleyiniz.
#############################################
# customers_level_based adında bir değişken tanımlayınız ve veri setine bu değişkeni ekleyiniz.
# Dikkat!
# list comp ile customers_level_based değerleri oluşturulduktan sonra bu değerlerin tekilleştirilmesi gerekmektedir.
# Örneğin birden fazla şu ifadeden olabilir: USA_ANDROID_MALE_0_18
# Bunları groupby'a alıp price ortalamalarını almak gerekmektedir.

# YÖNTEM 1
# değişken isimleri:
agg_df.columns

# gözlem değerlerine nasıl erişiriz?
agg_df.values
for row in agg_df.values:
    print(row)

# COUNTRY, SOURCE, SEX ve age_cat değişkenlerinin DEĞERLERİNİ yan yana koymak ve alt tireyle birleştirmek istiyoruz.
# Bunu list comprehension ile yapabiliriz.
# Yukarıdaki döngüdeki gözlem değerlerinin bize lazım olanlarını seçecek şekilde işlemi gerçekletirelim:
[row[0].upper() + "_" + row[1].upper() + "_" + row[2].upper() + "_" + row[5].upper() for row in agg_df.values]

# Veri setine eklemek istersek
# agg_df["customers_level_based"] = [row[0].upper() + "_" + row[1].upper() + "_" + row[2].upper() + "_" + row[5].upper() for row in agg_df.values]
# agg_df.head()


l = ["a", "b", "c"]
"_".join(l).upper()
# YÖNTEM 2
agg_df["customers_level_based"] = ['_'.join(i).upper() for i in agg_df.drop(["AGE", "PRICE"], axis=1).values]
# ['_'.join(i).upper() for i in agg_df[['COUNTRY', 'SOURCE', 'SEX', 'age_cat']].values]


# YÖNTEM 3 - Yöntem 1'de değişiklik yapılırsa için yukarıdaki işlemleri tekrar çalıştır!
# APPLY ÖZET
agg_df[["COUNTRY", "SOURCE", "SEX", "age_cat"]].head().apply(lambda x: print(x), axis=1)  # apply ile satırları alabiliyoruz
agg_df[["COUNTRY", "SOURCE", "SEX", "age_cat"]].head().apply(lambda x: print(type(x)), axis=1)  # her birini series olarak verir

agg_df[["COUNTRY", "SOURCE", "SEX", "age_cat"]].apply(lambda x: x["COUNTRY"].upper()+"_"+
                                                                x["SOURCE"].upper()+"_"+x["SEX"].upper()+"_"+
                                                                x["age_cat"].upper(), axis=1)

# YÖNTEM 4
agg_df.apply(lambda x: f"{x['COUNTRY']}_{x['SOURCE']}_{x['SEX']}_{x['age_cat']}".upper(), axis=1)


# YÖNTEM 5
# l = ["a", "b", "c"]
# "_".join(l)

agg_df['customers_level_based'] = agg_df[['COUNTRY', 'SOURCE', 'SEX', 'age_cat']].apply(lambda x: '_'.join(x).upper(), axis=1)


# YÖNTEM 6
agg_df['customers_level_based'] = agg_df[['COUNTRY', 'SOURCE', 'SEX', 'age_cat']].agg(lambda x: '_'.join(x).upper(), axis=1)  # axis=1 satır bazında işlem

# YÖNTEM 7
agg_df['customers_level_based'] = [f'{COUNTRY}_{SOURCE}_{SEX}_{AGE_CAT}'.upper()
                                   for COUNTRY, SOURCE, SEX, AGE_CAT in
                                   zip(agg_df['COUNTRY'], agg_df['SOURCE'], agg_df['SEX'], agg_df['age_cat'])]


# YÖNTEM 8
agg_df['customers_level_based'] = (agg_df['COUNTRY'] + "_" + agg_df['SOURCE'] + "_" + agg_df['SEX'] + "_" + agg_df['age_cat'].astype(object)).str.upper()
agg_df.head()




# Gereksiz değişkenleri çıkaralım:
agg_df = agg_df[["customers_level_based", "PRICE"]]
agg_df.head()


# Amacımıza bir adım daha yaklaştık.
# Burada ufak bir problem var. Birçok aynı segment olacak.
# örneğin USA_ANDROID_MALE_0_18 segmentinden birçok sayıda olabilir.
# kontrol edelim:
agg_df.shape
agg_df["customers_level_based"].nunique()
agg_df["customers_level_based"].value_counts()

# Bu sebeple segmentlere göre groupby yaptıktan sonra price ortalamalarını almalı ve segmentleri tekilleştirmeliyiz.
agg_df = agg_df.groupby("customers_level_based").agg({"PRICE": "mean"})
agg_df


# customers_level_based index'te yer almaktadır. Bunu değişkene çevirelim.
agg_df = agg_df.reset_index()
agg_df.head()
# agg_df.groupby("customers_level_based", as_index=False).agg({"PRICE": "mean"})  # reset_index yapmadan sütuna çekmek için


# kontrol edelim. her bir persona'nın 1 tane olmasını bekleriz:
agg_df.shape
agg_df["customers_level_based"].nunique()
agg_df["customers_level_based"].value_counts()
agg_df.head()


#############################################
# GÖREV 7: Yeni müşterileri (USA_ANDROID_MALE_0_18) segmentlere ayırınız.
#############################################
# PRICE'a göre segmentlere ayırınız,
# segmentleri "SEGMENT" isimlendirmesi ile agg_df'e ekleyiniz,
# segmentleri betimleyiniz,
agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 4, labels=["D", "C", "B", "A"])
agg_df.head()
agg_df.groupby("SEGMENT").agg({"PRICE": "mean"})




#############################################
# GÖREV 8: Yeni gelen müşterileri sınıflandırınız ne kadar gelir getirebileceğini tahmin ediniz.
#############################################
# 33 yaşında ANDROID kullanan bir Türk kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?
new_user = "TUR_ANDROID_FEMALE_31_40"
agg_df.loc[agg_df["customers_level_based"] == new_user]

# 35 yaşında IOS kullanan bir Fransız kadını hangi segmente ve ortalama ne kadar gelir kazandırması beklenir?
new_user = "FRA_IOS_FEMALE_31_40"
agg_df.loc[agg_df["customers_level_based"] == new_user]


# BONUS (Fonksiyonlaştırma)
def get_customer_average_income(customer_id):
    return agg_df.loc[agg_df['customers_level_based'] == customer_id]

new_user = "FRA_IOS_FEMALE_31_40"
get_customer_average_income(new_user)

get_customer_average_income("TUR_ANDROID_FEMALE_31_40")
get_customer_average_income("CAN_ANDROID_FEMALE_41_66")










