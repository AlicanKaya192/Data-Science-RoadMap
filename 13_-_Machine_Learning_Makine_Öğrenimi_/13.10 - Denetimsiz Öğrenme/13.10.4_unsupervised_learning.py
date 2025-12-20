################################
# Unsupervised Learning (Gözetimsiz Öğrenme)
################################

# Gerekli kütüphanelerin import edilmesi
# Veri işleme, görselleştirme ve makine öğrenmesi için gerekli araçları yüklüyoruz.
# pip install yellowbrick komutu ile yellowbrick kütüphanesini kurabilirsiniz.
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder

################################
# K-Means
################################

# Veri setinin okunması
# USArrests veri seti, ABD eyaletlerindeki suç istatistiklerini içerir.
# index_col=0 diyerek eyalet isimlerinin index olarak kullanılmasını sağlıyoruz.
df = pd.read_csv("datasets/USArrests.csv", index_col=0)

# Veri setine ilk bakış
# İlk 5 satırı getirerek verinin genel yapısını görüyoruz.
df.head()
# Veri setinde eksik gözlem (NaN) var mı kontrol ediyoruz.
df.isnull().sum()
# Değişken tiplerini ve bellek kullanımını inceliyoruz.
df.info()
# Sayısal değişkenlerin özet istatistiklerini (ortalama, standart sapma vb.) transpoze ederek inceliyoruz.
df.describe().T

# Verilerin ölçeklendirilmesi (Scaling)
# Uzaklık temelli yöntemlerde (K-Means gibi) değişkenlerin aynı ölçekte olması model başarısı için kritiktir.
# MinMaxScaler kullanarak tüm değerleri 0 ile 1 arasına sıkıştırıyoruz.
sc = MinMaxScaler((0, 1))
df = sc.fit_transform(df)
# Ölçeklendirme sonrası ilk 5 gözleme bakıyoruz.
df[0:5]

# K-Means modelinin kurulması
# Başlangıç olarak rastgele bir küme sayısı (n_clusters=4) belirliyoruz.
# random_state=17 parametresi ile kod her çalıştığında aynı sonuçları üretmesini sağlıyoruz.
kmeans = KMeans(n_clusters=4, random_state=17).fit(df)
# Modelin kullandığı parametreleri görüntülüyoruz.
kmeans.get_params()

# Model parametrelerinin incelenmesi
# Belirlediğimiz küme sayısı.
kmeans.n_clusters
# Oluşturulan kümelerin merkez noktaları.
kmeans.cluster_centers_
# Her bir gözlemin (eyaletin) hangi kümeye atandığı bilgisi (0, 1, 2, 3).
kmeans.labels_
# SSD (Sum of Squared Distances) değeri. Küme içi hata kareler toplamı.
# Bu değerin düşük olması kümelerin daha sıkı (homojen) olduğunu gösterir.
kmeans.inertia_

################################
# Optimum Küme Sayısının Belirlenmesi
################################

# Elbow (Dirsek) Yöntemi ile optimum küme sayısını bulmaya çalışıyoruz.
# Boş bir KMeans nesnesi oluşturuyoruz.
kmeans = KMeans()
# Hata değerlerini (SSD) tutmak için boş bir liste oluşturuyoruz.
ssd = []
# 1'den 30'a kadar farklı küme sayılarını deneyeceğiz.
K = range(1, 30)

# Her bir K değeri için modeli kurup, hata değerini (inertia) listeye ekliyoruz.
for k in K:
    kmeans = KMeans(n_clusters=k).fit(df)
    ssd.append(kmeans.inertia_)

# Elde ettiğimiz hata değerlerini görselleştiriyoruz.
# Grafikteki kırılma noktası (dirsek), optimum küme sayısı olarak kabul edilir.
plt.plot(K, ssd, "bx-")
plt.xlabel("Farklı K Değerlerine Karşılık SSE/SSR/SSD")
plt.title("Optimum Küme sayısı için Elbow Yöntemi")
plt.show()

# Yellowbrick kütüphanesi ile Elbow yöntemini otomatikleştirme
# Bu araç, optimum küme sayısını hem görselleştirir hem de matematiksel olarak önerir.
kmeans = KMeans()
# k=(2, 20) diyerek 2 ile 20 arasındaki küme sayılarını denemesini istiyoruz.
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(df)
elbow.show()

# Yellowbrick tarafından önerilen optimum küme sayısını alıyoruz.
elbow.elbow_value_

################################
# Final Cluster'ların Oluşturulması
################################

# Belirlenen optimum küme sayısı ile final modelimizi kuruyoruz.
kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(df)

# Modelin özelliklerini kontrol ediyoruz.
kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_
# Ölçeklendirilmiş verinin ilk 5 satırı.
df[0:5]

# Elde edilen küme etiketlerini bir değişkene atıyoruz.
clusters_kmeans = kmeans.labels_

# Veri setini tekrar okuyoruz.
# Amacımız, ölçeklendirilmemiş (orijinal) veriler üzerinde analiz yapabilmek.
df = pd.read_csv("datasets/USArrests.csv", index_col=0)

# Küme etiketlerini orijinal veri setine yeni bir sütun olarak ekliyoruz.
df["cluster"] = clusters_kmeans

df.head()

# Küme numaraları 0'dan başladığı için, daha anlaşılır olması adına 1 ekliyoruz (1, 2, 3, 4...).
df["cluster"] = df["cluster"] + 1

# Örneğin 5. kümeye ait olan eyaletleri filtreleyip görüyoruz.
df[df["cluster"]==5]

# Kümelerin istatistiksel özelliklerini inceliyoruz.
# Hangi küme suç oranları bakımından nasıl bir profile sahip?
# groupby ile kümelere göre gruplayıp, count (sayı), mean (ortalama) ve median (ortanca) değerlerine bakıyoruz.
df.groupby("cluster").agg(["count","mean","median"])

# Elde ettiğimiz sonuçları daha sonra kullanmak üzere CSV dosyasına kaydediyoruz.
df.to_csv("clusters.csv")


################################
# Hierarchical Clustering (Hiyerarşik Kümeleme)
################################

# Veri setini temiz bir başlangıç için tekrar okuyoruz.
df = pd.read_csv("datasets/USArrests.csv", index_col=0)

# Hiyerarşik kümeleme de uzaklık temelli olduğu için verileri tekrar ölçeklendiriyoruz (0-1 arası).
sc = MinMaxScaler((0, 1))
df = sc.fit_transform(df)

# Linkage matrisinin oluşturulması.
# Agglomerative (birleştirici) yöntem kullanıyoruz.
# "average" metodu, iki küme arasındaki uzaklığı, kümelerdeki tüm noktaların ortalaması olarak alır.
hc_average = linkage(df, "average")

# Dendrogram (Ağaç Yapısı) çizimi.
# Veri setindeki hiyerarşik yapıyı görselleştirerek kümelerin nasıl birleştiğini görüyoruz.
plt.figure(figsize=(10, 5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
# leaf_font_size ile yaprakların (gözlemlerin) yazı boyutunu ayarlıyoruz.
dendrogram(hc_average,
           leaf_font_size=10)
plt.show()

# Dendrogramı sadeleştirmek (truncate) için.
# Gözlem sayısı çok fazla olduğunda dendrogram karmaşıklaşır.
# truncate_mode="lastp" ve p=10 diyerek sadece son 10 birleşimi gösteriyoruz.
plt.figure(figsize=(7, 5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
           truncate_mode="lastp",
           p=10,
           show_contracted=True,
           leaf_font_size=10)
plt.show()

################################
# Küme Sayısını Belirlemek
################################

# Dendrogram üzerinde yatay çizgiler çekerek küme sayısına karar veriyoruz.
# Çizginin kestiği dikey hat sayısı bize küme sayısını verir.
plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
dend = dendrogram(hc_average)
# Kırmızı çizgi (y=0.5) çekilirse kaç küme oluşur?
plt.axhline(y=0.5, color='r', linestyle='--')
# Mavi çizgi (y=0.6) çekilirse kaç küme oluşur?
plt.axhline(y=0.6, color='b', linestyle='--')
plt.show()

################################
# Final Modeli Oluşturmak
################################

from sklearn.cluster import AgglomerativeClustering

# Dendrogram analizine göre 5 küme oluşturmaya karar veriyoruz.
# linkage="average" parametresi ile ortalama bağlantı yöntemini kullanıyoruz.
cluster = AgglomerativeClustering(n_clusters=5, linkage="average")

# Modeli veriye fit edip tahmin edilen küme etiketlerini alıyoruz.
clusters = cluster.fit_predict(df)

# Orijinal veri setine küme etiketlerini eklemek için veriyi tekrar okuyoruz.
df = pd.read_csv("datasets/USArrests.csv", index_col=0)
# Hiyerarşik kümeleme sonuçlarını "hi_cluster_no" sütununa ekliyoruz.
df["hi_cluster_no"] = clusters

# Küme numaralarını 1'den başlatıyoruz.
df["hi_cluster_no"] = df["hi_cluster_no"] + 1

# Daha önce yaptığımız K-Means sonuçlarını da bu dataframe'e ekleyip karşılaştırma yapabiliriz.
# (Not: clusters_kmeans değişkeni yukarıdaki K-Means bölümünden hafızada kalmıştı)
df["kmeans_cluster_no"] = clusters_kmeans
df["kmeans_cluster_no"] = df["kmeans_cluster_no"] + 1

# Artık her eyaletin hem K-Means hem de Hiyerarşik Kümeleme sonucunu görebiliriz.
# df.head() diyerek kontrol edebilirsiniz.

################################
# Principal Component Analysis (PCA)
################################

# Hitters veri setini okuyoruz (Beyzbol oyuncu verileri).
df = pd.read_csv("datasets/Hitters.csv")
df.head()

# PCA sadece sayısal değişkenlere uygulanır.
# Kategorik değişkenleri ve bağımlı değişkenimiz olan "Salary"i çıkarıyoruz.
num_cols = [col for col in df.columns if df[col].dtypes != "O" and "Salary" not in col]

# Seçilen sayısal sütunlara bakıyoruz.
df[num_cols].head()

# Veri setini sadece sayısal değişkenlerden oluşacak şekilde güncelliyoruz.
df = df[num_cols]
# Eksik değerler PCA'de hataya neden olur, bu yüzden siliyoruz.
df.dropna(inplace=True)
# Veri setinin boyutunu kontrol ediyoruz.
df.shape

# PCA öncesi standartlaştırma işlemi şarttır.
# Değişkenlerin varyanslarının ölçekten etkilenmemesi için hepsini standartlaştırıyoruz (mean=0, std=1).
df = StandardScaler().fit_transform(df)

# PCA modelini kuruyoruz ve veriye uyguluyoruz.
pca = PCA()
pca_fit = pca.fit_transform(df)

# Her bir bileşenin (component) açıkladığı varyans oranlarını görüyoruz.
# İlk bileşen genellikle en yüksek varyansı açıklar.
pca.explained_variance_ratio_

# Kümülatif varyans oranlarına bakıyoruz.
# Bileşenler toplandıkça toplam bilginin ne kadarı açıklanıyor?
np.cumsum(pca.explained_variance_ratio_)


################################
# Optimum Bileşen Sayısı
################################

# Kümülatif varyans grafiğini çizerek kaç bileşen seçeceğimize karar veriyoruz.
pca = PCA().fit(df)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Bileşen Sayısını")
plt.ylabel("Kümülatif Varyans Oranı")
plt.show()

################################
# Final PCA'in Oluşturulması
################################

# Grafiğe bakarak veya belirli bir varyans oranını (örn: %90) hedefleyerek bileşen sayısına karar veriyoruz.
# Burada 3 bileşen seçtik.
pca = PCA(n_components=3)
pca_fit = pca.fit_transform(df)

# Seçilen 3 bileşenin açıkladığı varyans oranları.
pca.explained_variance_ratio_
# Bu 3 bileşen ile toplam varyansın ne kadarını açıkladığımızı görüyoruz.
np.cumsum(pca.explained_variance_ratio_)


################################
# BONUS: Principal Component Regression (PCR)
################################

# Veriyi baştan temiz bir şekilde okuyoruz.
df = pd.read_csv("datasets/Hitters.csv")
# Veri setinin boyutuna bakıyoruz.
df.shape

# PCA dönüşümü yapılmış verinin uzunluğunu kontrol ediyoruz.
len(pca_fit)

# Sayısal değişkenleri tekrar belirliyoruz.
num_cols = [col for col in df.columns if df[col].dtypes != "O" and "Salary" not in col]
len(num_cols)

# Kategorik değişkenleri (PCA'e dahil etmemiştik) şimdi geri alıyoruz.
# Çünkü regresyon modelinde bunları da kullanmak isteyebiliriz.
others = [col for col in df.columns if col not in num_cols]

# PCA'den elde ettiğimiz bileşenleri bir DataFrame'e çeviriyoruz.
# İsimlerini PC1, PC2, PC3 olarak veriyoruz.
pd.DataFrame(pca_fit, columns=["PC1","PC2","PC3"]).head()

# Diğer (kategorik) değişkenlere bakıyoruz.
df[others].head()

# PCA bileşenleri ile diğer (kategorik/hedef) değişkenleri yan yana birleştiriyoruz.
# Böylece hem boyut indirgenmiş sayısal veriler hem de kategorik veriler tek bir çatıda toplanıyor.
final_df = pd.concat([pd.DataFrame(pca_fit, columns=["PC1","PC2","PC3"]),
                      df[others]], axis=1)
final_df.head()


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

def label_encoder(dataframe, binary_col):
    """
    Verilen dataframe içindeki binary (iki sınıflı) kategorik değişkeni 
    Label Encoding yöntemiyle sayısal hale getirir.

    Args:
        dataframe (pd.DataFrame): İşlem yapılacak veri seti.
        binary_col (str): Dönüştürülecek sütun adı.

    Returns:
        pd.DataFrame: Dönüştürülmüş veri seti.
    """
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

# Kategorik değişkenleri modele sokabilmek için encode ediyoruz.
for col in ["NewLeague", "Division", "League"]:
    label_encoder(final_df, col)

# Eksik değerleri siliyoruz.
final_df.dropna(inplace=True)

# Bağımlı değişken (Salary) ve bağımsız değişkenleri (X) ayırıyoruz.
y = final_df["Salary"]
X = final_df.drop(["Salary"], axis=1)

# Linear Regression Modeli kuruyoruz.
lm = LinearRegression()

# 5 katlı çapraz doğrulama (Cross Validation) ile modelin başarısını (RMSE) ölçüyoruz.
# neg_mean_squared_error negatif döndüğü için eksi ile çarpıp karekökünü alıyoruz.
rmse = np.mean(np.sqrt(-cross_val_score(lm, X, y, cv=5, scoring="neg_mean_squared_error")))

# Hedef değişkenin ortalamasına bakarak hatanın büyüklüğünü yorumluyoruz.
y.mean() 


# Decision Tree Regressor (CART) Modeli deniyoruz.
cart = DecisionTreeRegressor()
# Yine 5 katlı çapraz doğrulama ile hatamızı hesaplıyoruz.
rmse = np.mean(np.sqrt(-cross_val_score(cart, X, y, cv=5, scoring="neg_mean_squared_error")))

# Hiperparametre Optimizasyonu yapıyoruz.
# Ağaç derinliği ve bölünme kriterleri için aralıklar belirliyoruz.
cart_params = {'max_depth': range(1, 11),
               "min_samples_split": range(2, 20)}

# GridSearchCV ile belirlediğimiz parametre aralıklarında en iyi kombinasyonu arıyoruz.
# n_jobs=-1 ile tüm işlemcileri kullanıyoruz.
cart_best_grid = GridSearchCV(cart,
                              cart_params,
                              cv=5,
                              n_jobs=-1,
                              verbose=True).fit(X, y)

# Bulunan en iyi parametrelerle final modelimizi kuruyoruz.
cart_final = DecisionTreeRegressor(**cart_best_grid.best_params_, random_state=17).fit(X, y)

# Final modelin hatasını tekrar hesaplıyoruz.
rmse = np.mean(np.sqrt(-cross_val_score(cart_final, X, y, cv=5, scoring="neg_mean_squared_error")))


################################
# BONUS: PCA ile Çok Boyutlu Veriyi 2 Boyutta Görselleştirme
################################

################################
# Breast Cancer
################################

# Pandas görüntüleme ayarlarını yapıyoruz, tüm sütunları görmek istiyoruz.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

# Breast Cancer veri setini okuyoruz.
df = pd.read_csv("datasets/breast_cancer.csv")

# Hedef değişkeni (diagnosis) ve gereksiz id sütununu ayırıyoruz.
y = df["diagnosis"]
X = df.drop(["diagnosis", "id"], axis=1)


def create_pca_df(X, y):
    """
    Verilen özellik seti (X) üzerinde PCA uygulayarak veriyi 2 boyuta indirger 
    ve hedef değişken (y) ile birleştirerek yeni bir DataFrame oluşturur.

    Args:
        X (pd.DataFrame): Özellikler matrisi.
        y (pd.Series): Hedef değişken.

    Returns:
        pd.DataFrame: PC1, PC2 ve hedef değişkeni içeren DataFrame.
    """
    # Veriyi standartlaştırıyoruz (PCA için zorunlu).
    X = StandardScaler().fit_transform(X)
    # 2 bileşenli PCA nesnesi oluşturuyoruz.
    pca = PCA(n_components=2)
    # Dönüşümü uyguluyoruz.
    pca_fit = pca.fit_transform(X)
    # Sonuçları DataFrame'e çeviriyoruz.
    pca_df = pd.DataFrame(data=pca_fit, columns=['PC1', 'PC2'])
    # Hedef değişkeni de yanına ekliyoruz.
    final_df = pd.concat([pca_df, pd.DataFrame(y)], axis=1)
    return final_df

# Fonksiyonu kullanarak PCA uygulanmış veri setini elde ediyoruz.
pca_df = create_pca_df(X, y)

def plot_pca(dataframe, target):
    """
    PCA sonucu elde edilen 2 boyutlu veriyi (PC1 ve PC2) hedef değişkene göre renklendirerek görselleştirir.

    Args:
        dataframe (pd.DataFrame): PCA uygulanmış veri seti (PC1, PC2 ve target sütunlarını içermeli).
        target (str): Hedef değişkenin sütun adı.
    """
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('PC1', fontsize=15)
    ax.set_ylabel('PC2', fontsize=15)
    ax.set_title(f'{target.capitalize()} ', fontsize=20)

    # Hedef değişkendeki benzersiz sınıfları alıyoruz.
    targets = list(dataframe[target].unique())
    # Her sınıf için rastgele bir renk seçiyoruz.
    colors = random.sample(['r', 'b', "g", "y"], len(targets))

    # Her bir sınıfı döngüyle gezip grafiğe ekliyoruz.
    for t, color in zip(targets, colors):
        indices = dataframe[target] == t
        ax.scatter(dataframe.loc[indices, 'PC1'], dataframe.loc[indices, 'PC2'], c=color, s=50)
    ax.legend(targets)
    ax.grid()
    plt.show()

# Sonuçları görselleştiriyoruz.
plot_pca(pca_df, "diagnosis")


################################
# Iris
################################

import seaborn as sns
# Iris veri setini Seaborn kütüphanesinden yüklüyoruz.
df = sns.load_dataset("iris")

# Hedef değişkeni ve özellikleri ayırıyoruz.
y = df["species"]
X = df.drop(["species"], axis=1)

# PCA fonksiyonumuzu çağırıyoruz.
pca_df = create_pca_df(X, y)

# Görselleştirme fonksiyonumuzu çağırıyoruz.
plot_pca(pca_df, "species")


################################
# Diabetes
################################

# Diabetes veri setini okuyoruz.
df = pd.read_csv("datasets/diabetes.csv")

# Hedef değişkeni (Outcome) ayırıyoruz.
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

# PCA uyguluyoruz.
pca_df = create_pca_df(X, y)

# Görselleştiriyoruz.
plot_pca(pca_df, "Outcome")
