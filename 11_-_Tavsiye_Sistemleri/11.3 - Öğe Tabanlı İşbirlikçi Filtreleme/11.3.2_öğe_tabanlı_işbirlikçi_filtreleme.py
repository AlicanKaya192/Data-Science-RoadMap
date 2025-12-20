###########################################
# Item-Based Collaborative Filtering (Öğe Tabanlı İşbirlikçi Filtreleme)
###########################################

# Veri seti: https://grouplens.org/datasets/movielens/
# Amaç: Kullanıcıların beğendiği filmlere benzer (korelasyonu yüksek) diğer filmleri önermek.
# Mantık: "Matrix" filmini beğenenler genellikle "Inception" filmini de beğenir.

# Adım 1: Veri Setinin Hazırlanması
# Adım 2: User Movie Df'inin Oluşturulması
# Adım 3: Item-Based Film Önerilerinin Yapılması
# Adım 4: Çalışma Scriptinin Hazırlanması

######################################
# Adım 1: Veri Setinin Hazırlanması
######################################
import pandas as pd
pd.set_option('display.max_columns', 500)

# Veri setlerini okuyoruz.
# movie.csv: Film bilgilerini içerir (movieId, title).
# rating.csv: Kullanıcı puanlarını içerir (userId, movieId, rating).
movie = pd.read_csv('Datasets ( Genel )/movie.csv')
rating = pd.read_csv('Datasets ( Genel )/rating.csv')

# İki veri setini movieId üzerinden birleştiriyoruz.
df = movie.merge(rating, how="left", on="movieId")
df.head()


######################################
# Adım 2: User Movie Df'inin Oluşturulması
######################################

# Hedefimiz: Satırlarda kullanıcıların (userId), sütunlarda filmlerin (title) olduğu,
# değerlerin ise verilen puanlar (rating) olduğu bir matris oluşturmak.

df.head()
df.shape

# Eşsiz film sayısı
df["title"].nunique()

# Hangi filme kaç yorum yapılmış?
df["title"].value_counts().head()

# Nadir filmleri (1000'den az yorum alanlar) analizden çıkarıyoruz.
# Çünkü az yorum alan filmlerin benzerlik hesaplamaları yanıltıcı olabilir.
comment_counts = pd.DataFrame(df["title"].value_counts())
rare_movies = comment_counts[comment_counts["title"] <= 1000].index

# Sadece yaygın filmleri (common_movies) tutuyoruz.
common_movies = df[~df["title"].isin(rare_movies)]
common_movies.shape
common_movies["title"].nunique()
df["title"].nunique()

# Pivot table ile matrisi oluşturuyoruz.
user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")

# Matrisin boyutu ve sütunları
user_movie_df.shape
user_movie_df.columns


######################################
# Adım 3: Item-Based Film Önerilerinin Yapılması
######################################

# Bir film seçelim: Matrix, The (1999)
movie_name = "Matrix, The (1999)"
# movie_name = "Ocean's Twelve (2004)"

# Seçilen filmin kullanıcılar tarafından verilen puanlarını alıyoruz.
movie_name_column = user_movie_df[movie_name]

# Bu filmin puanları ile diğer tüm filmlerin puanları arasındaki korelasyonu hesaplıyoruz.
# corrwith: Sütun bazlı korelasyon hesaplar.
user_movie_df.corrwith(movie_name_column).sort_values(ascending=False).head(10)

# Rastgele bir film seçip öneri yapalım:
movie_name = pd.Series(user_movie_df.columns).sample(1).values[0]
print(f"Rastgele Seçilen Film: {movie_name}")
movie_name_column = user_movie_df[movie_name]
user_movie_df.corrwith(movie_name_column).sort_values(ascending=False).head(10)


def check_film(keyword, user_movie_df):
    """
    Verilen anahtar kelimeyi içeren film isimlerini listeler.
    
    Parameters:
    keyword (str): Aranacak kelime.
    user_movie_df (pd.DataFrame): User-Movie matrisi.
    
    Returns:
    list: Eşleşen film isimleri.
    """
    return [col for col in user_movie_df.columns if keyword in col]

check_film("Insomnia", user_movie_df)


######################################
# Adım 4: Çalışma Scriptinin Hazırlanması
######################################

def create_user_movie_df():
    """
    Veri setini okur, ön işleme yapar ve User-Movie matrisini oluşturur.
    
    Returns:
    pd.DataFrame: User-Movie matrisi.
    """
    import pandas as pd
    # Veri setlerini okuma (Dosya yollarını kontrol ediniz)
    movie = pd.read_csv('Datasets ( Genel )/movie.csv')
    rating = pd.read_csv('Datasets ( Genel )/rating.csv')
    
    # Birleştirme
    df = movie.merge(rating, how="left", on="movieId")
    
    # Nadir filmleri eleme
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    
    # Pivot table oluşturma
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

# Matrisi oluşturuyoruz (Bu işlem biraz zaman alabilir)
user_movie_df = create_user_movie_df()


def item_based_recommender(movie_name, user_movie_df):
    """
    Verilen film ismine göre en benzer filmleri önerir.
    
    Parameters:
    movie_name (str): Referans film ismi.
    user_movie_df (pd.DataFrame): User-Movie matrisi.
    
    Returns:
    pd.Series: Benzerlik skorlarına göre sıralanmış film listesi.
    """
    # Filmin puanlarını al
    movie_name_column = user_movie_df[movie_name]
    # Korelasyon hesapla ve sırala
    return user_movie_df.corrwith(movie_name_column).sort_values(ascending=False).head(10)

# Örnek Kullanım:
item_based_recommender("Matrix, The (1999)", user_movie_df)

# Rastgele bir film seçip öneri alalım:
movie_name = pd.Series(user_movie_df.columns).sample(1).values[0]
print(f"Rastgele Seçilen Film: {movie_name}")
item_based_recommender(movie_name, user_movie_df)