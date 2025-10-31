
#########################
# İş Problemi
#########################

# Türkiye’nin en büyük online hizmet platformu olan Armut, hizmet verenler ile hizmet almak isteyenleri buluşturmaktadır.
# Bilgisayarın veya akıllı telefonunun üzerinden birkaç dokunuşla temizlik, tadilat, nakliyat gibi hizmetlere kolayca
# ulaşılmasını sağlamaktadır.
# Hizmet alan kullanıcıları ve bu kullanıcıların almış oldukları servis ve kategorileri içeren veri setini kullanarak
# Association Rule Learning ile ürün tavsiye sistemi oluşturulmak istenmektedir.

#########################
# Veri Seti
#########################
#Veri seti müşterilerin aldıkları servislerden ve bu servislerin kategorilerinden oluşmaktadır.
# Alınan her hizmetin tarih ve saat bilgisini içermektedir.

# UserId: Müşteri numarası
# ServiceId: Her kategoriye ait anonimleştirilmiş servislerdir. (Örnek : Temizlik kategorisi altında koltuk yıkama servisi)
# Bir ServiceId farklı kategoriler altında bulanabilir ve farklı kategoriler altında farklı servisleri ifade eder.
# (Örnek: CategoryId’si 7 ServiceId’si 4 olan hizmet petek temizliği iken CategoryId’si 2 ServiceId’si 4 olan hizmet mobilya montaj)
# CategoryId: Anonimleştirilmiş kategorilerdir. (Örnek : Temizlik, nakliyat, tadilat kategorisi)
# CreateDate: Hizmetin satın alındığı tarih


import pandas as pd
pd.set_option('display.max_columns', None)
from mlxtend.frequent_patterns import apriori, association_rules

import warnings
warnings.filterwarnings("ignore")

#########################
# GÖREV 1: Veriyi Hazırlama
#########################

# Adım 1: armut_data.csv dosyasınız okutunuz.
df_ = pd.read_csv("Armut ARL/armut_data.csv")
df = df_.copy()
df.head()


# Adım 2: ServisID her bir CategoryID özelinde farklı bir hizmeti temsil etmektedir.
# ServiceID ve CategoryID'yi "_" ile birleştirerek hizmetleri temsil edecek yeni bir değişken oluşturunuz.
df["Hizmet"] = df["ServiceId"].astype(str) + "_" + df["CategoryId"].astype(str)

# df["Hizmet"] = df[["ServiceId", "CategoryId"]].apply(lambda x: "_".join(x.astype(str)), axis=1)
# df["Hizmet"] = df.apply(lambda x: str(x["ServiceId"]) + "_" + str(x["CategoryId"]), axis=1)
# df["Hizmet"] = [str(row[1]) + "_" + str(row[2]) for row in df.values]


# Adım 3: Veri seti hizmetlerin alındığı tarih ve saatten oluşmaktadır, herhangi bir sepet tanımı (fatura vb. ) bulunmamaktadır.
# Association Rule Learning uygulayabilmek için bir sepet (fatura vb.) tanımı oluşturulması gerekmektedir.
# Burada sepet tanımı her bir müşterinin aylık aldığı hizmetlerdir. Örneğin; 7256 id'li müşteri 2017'in 8.ayında aldığı 9_4, 46_4 hizmetleri bir sepeti;
# 2017’in 10.ayında aldığı  9_4, 38_4  hizmetleri başka bir sepeti ifade etmektedir. Sepetleri unique bir ID ile tanımlanması gerekmektedir.
# Bunun için öncelikle sadece yıl ve ay içeren yeni bir date değişkeni oluşturunuz. UserID ve yeni oluşturduğunuz date değişkenini "_"
# ile birleştirirek ID adında yeni bir değişkene atayınız.

df["CreateDate"] = pd.to_datetime(df["CreateDate"])
df.head()
df.info()

# strftime() => tarih ve saati string ifadelerine dönüştürür
# https://www.programiz.com/python-programming/datetime/strftime
# https://www.geeksforgeeks.org/python-strftime-function/
df["NEW_DATE"] = df["CreateDate"].dt.strftime("%Y-%m")
# df["CreateDate"].head().dt.strftime("%Y/%m") # ÖRNEK
# df["CreateDate"].dt.strftime("%Y-%m-%d")  # ÖRNEK


df["SepetID"] = df["UserId"].astype(str) + "_" + df["NEW_DATE"]
df.head()
# df["SepetID"] = [str(row[0]) + "_" + str(row[5]) for row in df.values]
df[df["UserId"] == 7256]



#########################
# GÖREV 2: Birliktelik Kuralları Üretiniz
#########################

# Adım 1: Aşağıdaki gibi sepet hizmet pivot table’i oluşturunuz.

# Hizmet         0_8  10_9  11_11  12_7  13_11  14_7  15_1  16_8  17_5  18_4..
# SepetID
# 0_2017-08        0     0      0     0      0     0     0     0     0     0..
# 0_2017-09        0     0      0     0      0     0     0     0     0     0..
# 0_2018-01        0     0      0     0      0     0     0     0     0     0..
# 0_2018-04        0     0      0     0      0     1     0     0     0     0..
# 10000_2017-08    0     0      0     0      0     0     0     0     0     0..

# Unstack dokümantasyon => https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.unstack.html
# Unstack örneği => https://www.datasciencemadesimple.com/reshape-using-stack-unstack-function-pandas-python/
invoice_product_df = df.groupby(['SepetID', 'Hizmet'])['Hizmet'].count().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)
# invoice_product_df = df.groupby(['SepetID', 'Hizmet'])['Hizmet'].count().unstack(fill_value=0).applymap(lambda x: 1 if x > 0 else 0)
# df.pivot_table(index='SepetID', columns="Hizmet", aggfunc="count", fill_value=0).applymap(lambda x: 1 if x > 0 else 0)
invoice_product_df.head()




# Adım 2: Birliktelik kurallarını oluşturunuz.
frequent_itemsets = apriori(invoice_product_df, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules.head()





def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    # kuralları lifte göre büyükten kücüğe sıralar. (en uyumlu ilk ürünü yakalayabilmek için)
    # confidence'e göre de sıralanabilir insiyatife baglıdır.
    recommendation_list = [] # tavsiye edilecek ürünler için bos bir liste olusturuyoruz.
    # antecedents: X
    #items denildigi için frozenset olarak getirir. index ve hizmeti birleştirir.
    # i: index
    # product: X yani öneri isteyen hizmet
    for i, product in sorted_rules["antecedents"].items():
        for j in list(product): # hizmetlerde(product) gez:
            if j == product_id:# eger tavsiye istenen ürün yakalanırsa:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))
                # index bilgisini i ile tutuyordun bu index bilgisindeki consequents(Y) değerini recommendation_list'e ekle.

    # tavsiye listesinde tekrarlamayı önlemek için:
    # mesela 2'li 3'lü kombinasyonlarda aynı ürün tekrar düşmüş olabilir listeye gibi;
    # sözlük yapısının unique özelliginden yararlanıyoruz.
    recommendation_list = list({item for item_list in recommendation_list for item in item_list})
    return recommendation_list[:rec_count] # :rec_count istenen sayıya kadar tavsiye ürün getir.


#Adım 3: arl_recommender fonksiyonunu kullanarak en son 2_0 hizmetini alan bir kullanıcıya hizmet önerisinde bulununuz.
arl_recommender(rules, "2_0", 4)
