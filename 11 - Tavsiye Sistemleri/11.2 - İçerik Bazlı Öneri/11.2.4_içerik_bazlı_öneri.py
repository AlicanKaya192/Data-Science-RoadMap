#############################
# Content Based Recommendation (İçerik Temelli Tavsiye)
#############################

#############################
# Film Overview'larına Göre Tavsiye Geliştirme
#############################

# 1. TF-IDF Matrisinin Oluşturulması
# 2. Cosine Similarity Matrisinin Oluşturulması
# 3. Benzerliklere Göre Önerilerin Yapılması
# 4. Çalışma Scriptinin Hazırlanması

#################################
# 1. TF-IDF Matrisinin Oluşturulması
#################################

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# https://www.kaggle.com/rounakbanik/the-movies-dataset
# Veri setini okuyoruz.
df = pd.read_csv("Datasets ( Genel )/movies_metadata.csv", low_memory=False)  # DtypeWarning kapamak icin
df.head()
df.shape

# Filmlerin özetlerine (overview) göz atalım.
df["overview"].head()

# TF-IDF Vectorizer nesnesini oluşturuyoruz.
# stop_words="english": İngilizce'de yaygın kullanılan ve ayırt ediciliği olmayan kelimeleri (the, a, an, in vb.) çıkarıyoruz.
tfidf = TfidfVectorizer(stop_words="english")

# Eksik değerleri boş string ile dolduruyoruz, aksi halde hata alırız.
# df[df['overview'].isnull()]
df['overview'] = df['overview'].fillna('')

# Metin verisini TF-IDF skorlarına dönüştürüyoruz.
# fit_transform: Hem kelime hazinesini öğrenir hem de dönüşümü yapar.
tfidf_matrix = tfidf.fit_transform(df['overview'])

# Matrisin boyutlarına bakalım: (Film Sayısı, Eşsiz Kelime Sayısı)
tfidf_matrix.shape

df['title'].shape

# Oluşan özellik isimlerini (kelimeleri) görebiliriz.
tfidf.get_feature_names_out()

# Matrisi array formatında görmek istersek (Büyük veride bellek sorunu yaratabilir, dikkat!)
tfidf_matrix.toarray()


#################################
# 2. Cosine Similarity Matrisinin Oluşturulması
#################################

# Filmler arasındaki benzerliği hesaplamak için Cosine Similarity kullanıyoruz.
# Bu işlem, her bir filmin diğer tüm filmlerle olan benzerlik skorunu hesaplar.
cosine_sim = cosine_similarity(tfidf_matrix,
                               tfidf_matrix)

# Matrisin boyutu (Film Sayısı x Film Sayısı) olacaktır.
cosine_sim.shape

# 2. filmin diğer filmlerle olan benzerlik skorlarını görelim.
cosine_sim[1]


#################################
# 3. Benzerliklere Göre Önerilerin Yapılması
#################################

# Film isimleri ve indexleri eşleştiren bir seri oluşturuyoruz.
# Bu sayede film isminden indexe, indexten film ismine erişebileceğiz.
indices = pd.Series(df.index, index=df['title'])

# Tekrar eden film isimleri var mı kontrol edelim.
indices.index.value_counts()

# Tekrar edenleri çıkarıp sonuncusunu tutuyoruz.
indices = indices[~indices.index.duplicated(keep='last')]

# Örnek: Cinderella filminin indexi nedir?
indices["Cinderella"]

indices["Sherlock Holmes"]

# Sherlock Holmes filminin indexini alalım.
movie_index = indices["Sherlock Holmes"]

# Bu filmin diğer filmlerle olan benzerlik skorlarını alalım.
cosine_sim[movie_index]

# Skorları bir DataFrame'e çevirelim.
similarity_scores = pd.DataFrame(cosine_sim[movie_index],
                                 columns=["score"])

# En benzer 10 filmi getirelim.
# Kendisi (0. index) hariç en yüksek skorlu 10 filmi alıyoruz (1'den 11'e kadar).
movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index

# Bu indexlere karşılık gelen film isimlerini getirelim.
df['title'].iloc[movie_indices]

#################################
# 4. Çalışma Scriptinin Hazırlanması
#################################

def content_based_recommender(title, cosine_sim, dataframe):
    """
    Verilen film başlığına göre içerik tabanlı öneriler sunar.
    
    Parameters:
    title (str): Öneri istenen filmin başlığı.
    cosine_sim (numpy.ndarray): Cosine Similarity matrisi.
    dataframe (pd.DataFrame): Film veri seti.
    
    Returns:
    pd.Series: Önerilen filmlerin başlıkları.
    """
    # index'leri olusturma
    indices = pd.Series(dataframe.index, index=dataframe['title'])
    indices = indices[~indices.index.duplicated(keep='last')]
    
    # title'ın index'ini yakalama
    movie_index = indices[title]
    
    # title'a gore benzerlik skorlarını hesapalama
    similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=["score"])
    
    # kendisi haric ilk 10 filmi getirme
    movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index
    
    return dataframe['title'].iloc[movie_indices]

# Örnek kullanımlar:
content_based_recommender("Sherlock Holmes", cosine_sim, df)

content_based_recommender("The Matrix", cosine_sim, df)

content_based_recommender("The Godfather", cosine_sim, df)

content_based_recommender('The Dark Knight Rises', cosine_sim, df)


def calculate_cosine_sim(dataframe):
    """
    Veri seti üzerinden TF-IDF matrisini ve Cosine Similarity matrisini hesaplar.
    
    Parameters:
    dataframe (pd.DataFrame): Film veri seti.
    
    Returns:
    numpy.ndarray: Cosine Similarity matrisi.
    """
    tfidf = TfidfVectorizer(stop_words='english')
    dataframe['overview'] = dataframe['overview'].fillna('')
    tfidf_matrix = tfidf.fit_transform(dataframe['overview'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim


# Fonksiyonu kullanarak matrisi hesaplayalım ve öneri alalım.
cosine_sim = calculate_cosine_sim(df)
content_based_recommender('The Dark Knight Rises', cosine_sim, df)
# 1 [90, 12, 23, 45, 67]
# 2 [90, 12, 23, 45, 67]
# 3
