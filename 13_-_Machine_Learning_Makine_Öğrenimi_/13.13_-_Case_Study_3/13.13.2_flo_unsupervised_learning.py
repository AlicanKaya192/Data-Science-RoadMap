
################################################################################################
# FLO Gözetimsiz Öğrenme ile Müşteri Segmentasyonu (Unsupervised Learning Customer Segmentation)
################################################################################################

# İş Problemi (Business Problem):
# FLO müşterilerini segmentlere ayırıp bu segmentlere göre pazarlama stratejileri belirlemek istiyor.
# Buna yönelik olarak müşterilerin davranışları tanımlanacak ve bu davranış öbeklenmelerine göre gruplar oluşturulacak.

# Veri Seti Hikayesi:
# Veri seti son alışverişlerini 2020 - 2021 yıllarında OmniChannel(hem online hem offline alışveriş yapan) olarak yapan
# müşterilerin geçmiş alışveriş davranışlarından elde edilen bilgilerden oluşmaktadır.

# master_id: Eşsiz müşteri numarası
# order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : En son alışverişin yapıldığı kanal
# first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date : Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online : Muşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline : Muşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online : Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline : Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline : Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online : Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 : Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
import datetime as dt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
import warnings
warnings.filterwarnings("ignore")

################################################################
# Adım 1: Veriyi Hazırlama (Data Preparation)
################################################################

# Veri setinin okunması
df = pd.read_csv("/Users/tuce/Desktop/Data-Science-RoadMap/Datasets ( Genel )/flo_data_20k.csv")

# Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir.
# Herbir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturulması
df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

# Tarih ifade eden değişkenlerin tipinin date'e çevrilmesi
date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)

# Analiz tarihinin belirlenmesi (Son alışveriş tarihinden 2 gün sonrası)
analysis_date = df["last_order_date"].max() + dt.timedelta(days=2)

# Recency ve Tenure değişkenlerinin oluşturulması
df["recency"] = (analysis_date - df["last_order_date"]).dt.days
df["tenure"] = (df["last_order_date"] - df["first_order_date"]).dt.days

# Modelde kullanılacak değişkenlerin seçilmesi (Recency, Frequency, Monetary, Tenure)
model_df = df[["order_num_total", "customer_value_total", "recency", "tenure"]]
print(model_df.head())

################################################################
# Adım 2: K-Means ile Müşteri Segmentasyonu
################################################################

# Verinin ölçeklendirilmesi (Skewness'ı azaltmak için Log Transformation da yapılabilir ama burada MinMax kullanacağız)
# K-Means uzaklık temelli olduğu için standartlaştırma önemlidir.
sc = MinMaxScaler((0, 1))
model_scaling = sc.fit_transform(model_df)
model_df = pd.DataFrame(model_scaling, columns=model_df.columns)
print(model_df.head())

# Optimum Küme Sayısının Belirlenmesi (Elbow Method)
kmeans = KMeans()
ssd = []
K = range(1, 30)

for k in K:
    kmeans = KMeans(n_clusters=k).fit(model_df)
    ssd.append(kmeans.inertia_)

plt.plot(K, ssd, "bx-")
plt.xlabel("Farklı K Değerlerine Karşılık SSE/SSR/SSD")
plt.title("Optimum Küme sayısı için Elbow Yöntemi")
# plt.show() # Grafiği görmek için yorumu kaldırabilirsiniz.

# Elbow yöntemine göre optimum küme sayısının belirlenmesi (Örneğin grafikten 4 veya 5 seçilebilir)
# Biz burada örnek olarak k=4 seçelim.
kmeans = KMeans(n_clusters=4, random_state=17).fit(model_df)

# Kümelerin belirlenmesi
clusters_kmeans = kmeans.labels_

# Orijinal veri setine segmentlerin eklenmesi
df["segment_kmeans"] = clusters_kmeans + 1 # 1'den başlaması için

print("\nK-Means Segment Analizi:")
print(df.groupby("segment_kmeans").agg({"order_num_total": ["mean", "min", "max"],
                                        "customer_value_total": ["mean", "min", "max"],
                                        "recency": ["mean", "min", "max"],
                                        "tenure": ["mean", "min", "max", "count"]}))

################################################################
# Adım 3: Hierarchical Clustering ile Müşteri Segmentasyonu
################################################################

# Dendrogram oluşturulması
hc_complete = linkage(model_df, 'complete')

plt.figure(figsize=(15, 10))
plt.title("Hiyerarşik Kümeleme - Dendrogram")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_complete,
           leaf_font_size=10)
# plt.show() # Grafiği görmek için yorumu kaldırabilirsiniz.

# Modelin oluşturulması ve kümelerin belirlenmesi (K-Means ile aynı sayıda küme seçelim: 4)
hc = AgglomerativeClustering(n_clusters=4)
segments_hc = hc.fit_predict(model_df)

df["segment_hc"] = segments_hc + 1

print("\nHierarchical Clustering Segment Analizi:")
print(df.groupby("segment_hc").agg({"order_num_total": ["mean", "min", "max"],
                                    "customer_value_total": ["mean", "min", "max"],
                                    "recency": ["mean", "min", "max"],
                                    "tenure": ["mean", "min", "max", "count"]}))

################################################################
# Adım 4: Sonuçların Değerlendirilmesi
################################################################

# K-Means ve Hiyerarşik Kümeleme sonuçlarının karşılaştırılması veya
# seçilen yönteme göre aksiyon planlarının oluşturulması.

# Örneğin K-Means segmentlerine göre:
# Segment 1: Sadık Müşteriler (Yüksek Tenure, Düşük Recency)
# Segment 2: Yeni Müşteriler (Düşük Tenure, Düşük Recency)
# Segment 3: Kayıp Riski Olanlar (Yüksek Recency)
# vb. yorumlar yapılabilir.

print("\nSegment Dağılımları:")
print(df["segment_kmeans"].value_counts())
print(df["segment_hc"].value_counts())
