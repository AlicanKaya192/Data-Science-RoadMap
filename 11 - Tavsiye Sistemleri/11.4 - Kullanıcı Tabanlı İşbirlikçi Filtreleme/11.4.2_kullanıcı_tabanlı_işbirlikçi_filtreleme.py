############################################
# User-Based Collaborative Filtering (Kullanıcı Tabanlı İşbirlikçi Filtreleme)
#############################################

# Amaç: Bir kullanıcıya, benzer zevklere sahip diğer kullanıcıların beğendiği filmleri önermek.
# Mantık: "Ali ve Ayşe benzer filmleri beğeniyorsa, Ali'nin beğenip Ayşe'nin henüz izlemediği filmleri Ayşe'ye önerebiliriz."

# Adım 1: Veri Setinin Hazırlanması
# Adım 2: Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
# Adım 3: Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişmek
# Adım 4: Öneri Yapılacak Kullanıcı ile En Benzer Davranışlı Kullanıcıların Belirlenmesi
# Adım 5: Weighted Average Recommendation Score'un Hesaplanması
# Adım 6: Çalışmanın Fonksiyonlaştırılması

#############################################
# Adım 1: Veri Setinin Hazırlanması
#############################################
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

def create_user_movie_df():
    """
    Veri setini okur, birleştirir ve User-Movie matrisini oluşturur.
    
    Returns:
    pd.DataFrame: User-Movie matrisi (Satırlar: UserID, Sütunlar: Film Başlıkları, Değerler: Puanlar).
    """
    import pandas as pd
    # Veri setlerini okuyoruz.
    movie = pd.read_csv('Datasets ( Genel )/movie.csv')
    rating = pd.read_csv('Datasets ( Genel )/rating.csv')
    
    # Verileri birleştiriyoruz.
    df = movie.merge(rating, how="left", on="movieId")
    
    # Nadir filmleri (1000'den az yorum alanlar) çıkarıyoruz.
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    
    # Pivot table oluşturuyoruz.
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

# Matrisi oluşturuyoruz.
user_movie_df = create_user_movie_df()

# Rastgele bir kullanıcı seçiyoruz (Öneri yapılacak hedef kullanıcı).
import pandas as pd
random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=45).values)


#############################################
# Adım 2: Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
#############################################

# Seçilen kullanıcının ID'si
random_user

# Kullanıcının verilerine erişiyoruz.
random_user_df = user_movie_df[user_movie_df.index == random_user]

# Kullanıcının izlediği (puan verdiği) filmleri listeliyoruz.
# notna(): NaN olmayan (yani puan verilmiş) değerleri seçer.
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()

# Kontrol amaçlı bir filme bakabiliriz.
user_movie_df.loc[user_movie_df.index == random_user,
                  user_movie_df.columns == "Silence of the Lambs, The (1991)"]

# Kullanıcının izlediği toplam film sayısı
len(movies_watched)



#############################################
# Adım 3: Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişmek
#############################################

# Sadece hedef kullanıcının izlediği filmleri içeren bir DataFrame oluşturuyoruz.
movies_watched_df = user_movie_df[movies_watched]

# Her bir kullanıcının, hedef kullanıcının izlediği filmlerden kaç tanesini izlediğini sayıyoruz.
user_movie_count = movies_watched_df.T.notnull().sum()

# Indexi resetleyerek düzgün bir DataFrame haline getiriyoruz.
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId", "movie_count"]

# En az 20 ortak film izleyen kullanıcıları görelim.
user_movie_count[user_movie_count["movie_count"] > 20].sort_values("movie_count", ascending=False)

# Hedef kullanıcının izlediği tüm filmleri (33 tane) izleyen kaç kişi var?
user_movie_count[user_movie_count["movie_count"] == 33].count()

# Ortak film sayısı 20'den fazla olan kullanıcıların ID'lerini alıyoruz.
# Bu eşik değeri (20) projeye göre değiştirilebilir veya oransal (%60) belirlenebilir.
users_same_movies = user_movie_count[user_movie_count["movie_count"] > 20]["userId"]

# Alternatif: Hedef kullanıcının izlediği filmlerin %60'ını izleyenleri seçmek.
# perc = len(movies_watched) * 60 / 100
# users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]

#############################################
# Adım 4: Öneri Yapılacak Kullanıcı ile En Benzer Davranışlı Kullanıcıların Belirlenmesi
#############################################

# Amacımız: Ortak film izleyenler arasında, puanlama davranışları hedef kullanıcıya en çok benzeyenleri bulmak.

# 1. Hedef kullanıcı ve diğer kullanıcıların verilerini bir araya getiriyoruz.
final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
                      random_user_df[movies_watched]])

# 2. Kullanıcılar arası korelasyon matrisini oluşturuyoruz.
# Transpozunu alıyoruz çünkü corr() fonksiyonu sütunlar (filmler) arası korelasyona bakar, biz kullanıcıları karşılaştırmak istiyoruz.
corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()

corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df = corr_df.reset_index()

# 3. Hedef kullanıcı ile korelasyonu %65'ten büyük olan kullanıcıları (Top Users) seçiyoruz.
top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][
    ["user_id_2", "corr"]].reset_index(drop=True)

top_users = top_users.sort_values(by='corr', ascending=False)

# Sütun ismini userId olarak değiştiriyoruz ki rating tablosuyla birleştirebilelim.
top_users.rename(columns={"user_id_2": "userId"}, inplace=True)

# Benzer kullanıcıların verdiği puanları ana veri setinden çekiyoruz.
rating = pd.read_csv('Datasets ( Genel )/rating.csv')
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')

# Hedef kullanıcının kendisini listeden çıkarıyoruz.
top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]


#############################################
# Adım 5: Weighted Average Recommendation Score'un Hesaplanması
#############################################

# Sadece benzer kullanıcıların verdiği puanlara bakmak yetmez, korelasyonu yüksek olan kullanıcının verdiği puana daha çok güvenmeliyiz.
# Bu yüzden puanları korelasyon ile ağırlıklandırıyoruz.

top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']

# Her film için ortalama ağırlıklı puanı hesaplıyoruz.
top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})

recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
recommendation_df = recommendation_df.reset_index()

# Ağırlıklı puanı 3.5'ten büyük olan filmleri getiriyoruz.
recommendation_df[recommendation_df["weighted_rating"] > 3.5]

# Puanına göre sıralıyoruz.
movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values("weighted_rating", ascending=False)

# Film isimlerini görmek için movie veri seti ile birleştiriyoruz.
movie = pd.read_csv('Datasets ( Genel )/movie.csv')
movies_to_be_recommend.merge(movie[["movieId", "title"]])



#############################################
# Adım 6: Çalışmanın Fonksiyonlaştırılması
#############################################

def create_user_movie_df():
    """
    User-Movie matrisini oluşturur.
    """
    import pandas as pd
    movie = pd.read_csv('Datasets ( Genel )/movie.csv')
    rating = pd.read_csv('Datasets ( Genel )/rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()

# perc = len(movies_watched) * 60 / 100
# users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]


def user_based_recommender(random_user, user_movie_df, ratio=60, cor_th=0.65, score=3.5):
    """
    Kullanıcı tabanlı işbirlikçi filtreleme ile film önerisi yapar.
    
    Parameters:
    random_user (int): Öneri yapılacak kullanıcı ID'si.
    user_movie_df (pd.DataFrame): User-Movie matrisi.
    ratio (int): Ortak izlenen film oranı eşiği (%).
    cor_th (float): Korelasyon eşiği.
    score (float): Öneri için minimum ağırlıklı puan eşiği.
    
    Returns:
    pd.DataFrame: Önerilen filmler ve puanları.
    """
    import pandas as pd
    # 1. Kullanıcının izlediği filmleri bul
    random_user_df = user_movie_df[user_movie_df.index == random_user]
    movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
    
    # 2. Aynı filmleri izleyen diğer kullanıcıları bul
    movies_watched_df = user_movie_df[movies_watched]
    user_movie_count = movies_watched_df.T.notnull().sum()
    user_movie_count = user_movie_count.reset_index()
    user_movie_count.columns = ["userId", "movie_count"]
    
    perc = len(movies_watched) * ratio / 100
    users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]

    # 3. Benzer kullanıcıları belirle (Korelasyon)
    final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
                          random_user_df[movies_watched]])

    corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
    corr_df = pd.DataFrame(corr_df, columns=["corr"])
    corr_df.index.names = ['user_id_1', 'user_id_2']
    corr_df = corr_df.reset_index()

    top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= cor_th)][
        ["user_id_2", "corr"]].reset_index(drop=True)

    top_users = top_users.sort_values(by='corr', ascending=False)
    top_users.rename(columns={"user_id_2": "userId"}, inplace=True)
    
    # 4. Weighted Average Recommendation Score hesapla
    rating = pd.read_csv('Datasets ( Genel )/rating.csv')
    top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')
    top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']

    recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
    recommendation_df = recommendation_df.reset_index()

    movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > score].sort_values("weighted_rating", ascending=False)
    movie = pd.read_csv('Datasets ( Genel )/movie.csv')
    return movies_to_be_recommend.merge(movie[["movieId", "title"]])


# Test edelim
random_user = int(pd.Series(user_movie_df.index).sample(1).values)
user_based_recommender(random_user, user_movie_df, cor_th=0.70, score=4)