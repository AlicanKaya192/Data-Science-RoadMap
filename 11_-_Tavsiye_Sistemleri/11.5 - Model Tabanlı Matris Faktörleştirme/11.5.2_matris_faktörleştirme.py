#############################
# Model-Based Collaborative Filtering: Matrix Factorization
#############################

# Amaç: Kullanıcılar ve filmler arasındaki gizli (latent) özellikleri öğrenerek boşlukları (henüz izlenmemiş filmleri) doldurmak.
# Yöntem: SVD (Singular Value Decomposition) kullanarak matris ayrıştırma işlemi yapacağız.

# !pip install surprise
import pandas as pd
from surprise import Reader, SVD, Dataset, accuracy
from surprise.model_selection import GridSearchCV, train_test_split, cross_validate
pd.set_option('display.max_columns', None)

# Adım 1: Veri Setinin Hazırlanması
# Adım 2: Modelleme
# Adım 3: Model Tuning
# Adım 4: Final Model ve Tahmin

#############################
# Adım 1: Veri Setinin Hazırlanması
#############################

# Veri setlerini okuyoruz.
movie = pd.read_csv('Datasets ( Genel )/movie.csv')
rating = pd.read_csv('Datasets ( Genel )/rating.csv')

# Verileri birleştiriyoruz.
df = movie.merge(rating, how="left", on="movieId")
df.head()

# Örnek olarak 4 film seçiyoruz ve bunlar üzerinden bir çalışma yapıyoruz.
movie_ids = [130219, 356, 4422, 541]
movies = ["The Dark Knight (2011)",
          "Cries and Whispers (Viskningar och rop) (1972)",
          "Forrest Gump (1994)",
          "Blade Runner (1982)"]

sample_df = df[df.movieId.isin(movie_ids)]
sample_df.head()

sample_df.shape

# Pivot table ile kullanıcı-film matrisini görelim (Sadece görsel kontrol için).
user_movie_df = sample_df.pivot_table(index=["userId"],
                                      columns=["title"],
                                      values="rating")

user_movie_df.shape

# Surprise kütüphanesi için veriyi uygun formata getiriyoruz.
# Reader nesnesi ile puan aralığını belirtiyoruz.
reader = Reader(rating_scale=(1, 5))

# Dataset.load_from_df ile veriyi yüklüyoruz. Sıralama önemli: userId, movieId, rating.
data = Dataset.load_from_df(sample_df[['userId',
                                       'movieId',
                                       'rating']], reader)

##############################
# Adım 2: Modelleme
##############################

# Veriyi eğitim ve test seti olarak ayırıyoruz (%75 Eğitim, %25 Test).
trainset, testset = train_test_split(data, test_size=.25)

# SVD (Singular Value Decomposition) modelini oluşturuyoruz.
svd_model = SVD()

# Modeli eğitiyoruz.
svd_model.fit(trainset)

# Test seti üzerinde tahmin yapıyoruz.
predictions = svd_model.test(testset)

# Başarı metriği olarak RMSE (Root Mean Square Error) hesaplıyoruz.
accuracy.rmse(predictions)

# Belirli bir kullanıcı ve film için tahmin yapalım.
# uid=1.0 (Kullanıcı ID), iid=541 (Blade Runner Film ID)
svd_model.predict(uid=1.0, iid=541, verbose=True)

svd_model.predict(uid=1.0, iid=356, verbose=True)

# Kullanıcının gerçekte verdiği puanları kontrol edelim.
sample_df[sample_df["userId"] == 1]

##############################
# Adım 3: Model Tuning
##############################

# Hiperparametre optimizasyonu için Grid Search yapıyoruz.
# n_epochs: İterasyon sayısı
# lr_all: Öğrenme oranı (Learning Rate)
param_grid = {'n_epochs': [5, 10, 20],
              'lr_all': [0.002, 0.005, 0.007]}

# GridSearchCV ile en iyi parametreleri arıyoruz.
gs = GridSearchCV(SVD,
                  param_grid,
                  measures=['rmse', 'mae'],
                  cv=3,
                  n_jobs=-1,
                  joblib_verbose=True)

gs.fit(data)

# En iyi skoru ve parametreleri görelim.
gs.best_score['rmse']
gs.best_params['rmse']


##############################
# Adım 4: Final Model ve Tahmin
##############################

# Modelin özelliklerini inceleyelim.
dir(svd_model)
svd_model.n_epochs

# Grid Search'ten gelen en iyi parametrelerle final modelimizi oluşturuyoruz.
svd_model = SVD(**gs.best_params['rmse'])

# Verinin tamamını eğitim seti olarak kullanıyoruz (Train-Test ayrımı yapmadan).
data = data.build_full_trainset()

# Modeli tüm veri ile eğitiyoruz.
svd_model.fit(data)

# Final tahmin yapalım.
# Kullanıcı 1, Film 541 (Blade Runner) için tahmin edilen puan.
svd_model.predict(uid=1.0, iid=541, verbose=True)