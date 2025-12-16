################################################
# Random Forests, GBM, XGBoost, LightGBM, CatBoost
################################################

import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# !pip install catboost
# !pip install xgboost
# !pip install lightgbm

# Pandas çıktılarında tüm sütunların görünmesini sağlar.
pd.set_option('display.max_columns', None)
# Pandas çıktılarında genişlik ayarını yaparak satırların alt alta kaymasını engeller.
pd.set_option('display.width', 500)

# Uyarı mesajlarını (warning) görmezden gelmek için filtreleme yapar.
warnings.simplefilter(action='ignore', category=Warning)

# Veri setinin yüklenmesi
df = pd.read_csv("datasets/diabetes.csv")

# Bağımlı ve bağımsız değişkenlerin ayrılması
# Hedef değişkenimiz (y) "Outcome" sütunudur.
y = df["Outcome"]
# Bağımsız değişkenler (X) "Outcome" dışındaki tüm sütunlardır.
X = df.drop(["Outcome"], axis=1)

################################################
# Random Forests
################################################
# Bagging temelli bir ensemble yöntemidir.
# Birden fazla karar ağacının ürettiği tahminlerin bir araya getirilmesiyle çalışır.
# Varyansı düşürür ve aşırı öğrenmeye (overfitting) karşı dirençlidir.

# Model nesnesinin oluşturulması
# random_state=17 ile sonuçların tekrarlanabilir olmasını sağlıyoruz.
rf_model = RandomForestClassifier(random_state=17)
# Modelin varsayılan parametrelerini görüntülüyoruz.
rf_model.get_params()

# Modelin CV ile değerlendirilmesi (Hiperparametre optimizasyonu öncesi)
# 10 katlı çapraz doğrulama (10-fold CV) ile modelin başarısını ölçüyoruz.
cv_results = cross_validate(rf_model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
# Doğruluk (Accuracy) skorunun ortalamasını alıyoruz.
cv_results['test_accuracy'].mean()
# F1 skorunun ortalamasını alıyoruz.
cv_results['test_f1'].mean()
# ROC AUC skorunun ortalamasını alıyoruz.
cv_results['test_roc_auc'].mean()


# Hiperparametre arama ızgarasının (grid) belirlenmesi
# Denenecek parametre kombinasyonlarını bir sözlük içinde tanımlıyoruz.
rf_params = {"max_depth": [5, 8, None], # Ağacın maksimum derinliği
             "max_features": [3, 5, 7, "auto"], # Her bölünmede göz önünde bulundurulacak değişken sayısı
             "min_samples_split": [2, 5, 8, 15, 20], # Bir düğümün bölünmesi için gereken minimum örnek sayısı
             "n_estimators": [100, 200, 500]} # Oluşturulacak ağaç sayısı


# GridSearchCV ile en iyi parametrelerin bulunması
# Olası tüm kombinasyonları deneyerek en iyi performansı veren seti bulur.
# n_jobs=-1 ile işlemci çekirdeklerinin tamamını kullanır.
rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

# En iyi parametreleri görüntülüyoruz.
rf_best_grid.best_params_

# Final modelin en iyi parametrelerle kurulması
# Bulunan en iyi parametreleri modele set ediyoruz ve modeli tüm veriyle tekrar eğitiyoruz.
rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y)

# Final modelin CV ile değerlendirilmesi
# Optimize edilmiş modelin performansını tekrar 10 katlı CV ile ölçüyoruz.
cv_results = cross_validate(rf_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()


def plot_importance(model, features, num=len(X), save=False):
    """
    Bu fonksiyon, eğitilmiş bir ağaç tabanlı modelin değişken önem düzeylerini görselleştirir.
    Hangi değişkenin modelin tahminlerinde daha etkili olduğunu anlamamızı sağlar.

    Args:
        model: Eğitilmiş makine öğrenmesi modeli (RandomForest, GBM, XGBoost vb.).
        features: Bağımsız değişkenlerin bulunduğu DataFrame (X).
        num: Görselleştirilecek en önemli değişken sayısı (varsayılan: tüm değişkenler).
        save: Grafiğin kaydedilip kaydedilmeyeceği bilgisi (True/False).
    """
    # Modelin feature_importances_ özelliğini kullanarak değişken önemlerini alıyoruz.
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    
    # Görselleştirme için grafik boyutunu ayarlıyoruz.
    plt.figure(figsize=(10, 10))
    # Yazı boyutunu ayarlıyoruz.
    sns.set(font_scale=1)
    # Değişkenleri önem sırasına göre sıralayıp barplot (çubuk grafik) ile çizdiriyoruz.
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    # Grafik başlığını ekliyoruz.
    plt.title('Features')
    # Grafik düzenini sıkılaştırıyoruz (elemanların birbirine girmemesi için).
    plt.tight_layout()
    # Grafiği ekrana basıyoruz.
    plt.show()
    # Eğer save=True ise grafiği dosyaya kaydediyoruz.
    if save:
        plt.savefig('importances.png')

plot_importance(rf_final, X)

def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    """
    Bu fonksiyon, modelin belirli bir hiperparametresinin farklı değerleri için eğitim ve doğrulama skorlarını hesaplar ve görselleştirir.
    Modelin aşırı öğrenme (overfitting) veya eksik öğrenme (underfitting) durumunu analiz etmeye yardımcı olur.

    Args:
        model: Eğitilecek makine öğrenmesi modeli.
        X: Bağımsız değişkenler.
        y: Bağımlı değişken.
        param_name: Değerleri değiştirilecek hiperparametre ismi (örn: 'max_depth').
        param_range: Hiperparametrenin alacağı değerler aralığı.
        scoring: Başarı değerlendirme metriği (varsayılan: 'roc_auc').
        cv: Çapraz doğrulama kat sayısı (varsayılan: 10).
    """
    # validation_curve fonksiyonu ile belirtilen parametre aralığında skorları hesaplıyoruz.
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    # Eğitim skorlarının ortalamasını alıyoruz.
    mean_train_score = np.mean(train_score, axis=1)
    # Test (doğrulama) skorlarının ortalamasını alıyoruz.
    mean_test_score = np.mean(test_score, axis=1)

    # Eğitim skorlarını çizdiriyoruz.
    plt.plot(param_range, mean_train_score,
             label="Training Score", color='b')

    # Doğrulama skorlarını çizdiriyoruz.
    plt.plot(param_range, mean_test_score,
             label="Validation Score", color='g')

    # Grafik başlığı ve eksen etiketlerini ekliyoruz.
    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    # Lejantı (açıklama kutusu) en uygun yere koyuyoruz.
    plt.legend(loc='best')
    # Grafiği gösteriyoruz.
    plt.show(block=True)

val_curve_params(rf_final, X, y, "max_depth", range(1, 11), scoring="roc_auc")


################################################
# GBM (Gradient Boosting Machines)
################################################
# Boosting temelli bir yöntemdir.
# Zayıf öğrenicileri (genellikle karar ağaçları) ardışık olarak ekleyerek güçlü bir model oluşturur.
# Her yeni ağaç, önceki ağaçların hatalarını (residuals) düzeltmeye çalışır.

# Model nesnesinin oluşturulması
gbm_model = GradientBoostingClassifier(random_state=17)

gbm_model.get_params()

# Modelin CV ile değerlendirilmesi
# 5 katlı çapraz doğrulama ile modelin başarısını ölçüyoruz.
cv_results = cross_validate(gbm_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
# Doğruluk (Accuracy) skorunun ortalamasını alıyoruz.
cv_results['test_accuracy'].mean()
# 0.7591715474068416
# F1 skorunun ortalamasını alıyoruz.
cv_results['test_f1'].mean()
# 0.634
# ROC AUC skorunun ortalamasını alıyoruz.
cv_results['test_roc_auc'].mean()
# 0.82548

# Hiperparametre arama ızgarası
# GBM için optimize edilecek parametreleri belirliyoruz.
gbm_params = {"learning_rate": [0.01, 0.1], # Öğrenme oranı (her ağacın katkısı)
              "max_depth": [3, 8, 10], # Ağaç derinliği
              "n_estimators": [100, 500, 1000], # Ağaç sayısı
              "subsample": [1, 0.5, 0.7]} # Her ağaç için kullanılacak veri oranı

# GridSearchCV işlemi
# En iyi parametreleri bulmak için arama yapıyoruz.
gbm_best_grid = GridSearchCV(gbm_model, gbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

# En iyi parametreleri görüntülüyoruz.
gbm_best_grid.best_params_

# Final modelin kurulması
# Bulunan en iyi parametrelerle final modelimizi oluşturuyoruz.
gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_, random_state=17, ).fit(X, y)


# Final modelin CV ile değerlendirilmesi
# Optimize edilmiş modelin performansını tekrar 5 katlı CV ile ölçüyoruz.
cv_results = cross_validate(gbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()


################################################
# XGBoost (eXtreme Gradient Boosting)
################################################
# GBM'in hız ve performans açısından optimize edilmiş halidir.
# Ölçeklenebilir, dağıtık ve hızlıdır.
# Düzenlileştirme (Regularization) parametreleri içerir, bu da overfitting'i engeller.

# Model nesnesinin oluşturulması
# use_label_encoder=False uyarısını almamak için eklenmiştir.
xgboost_model = XGBClassifier(random_state=17, use_label_encoder=False)
xgboost_model.get_params()

# Modelin CV ile değerlendirilmesi
# 5 katlı çapraz doğrulama ile modelin başarısını ölçüyoruz.
cv_results = cross_validate(xgboost_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
# Doğruluk (Accuracy) skorunun ortalamasını alıyoruz.
cv_results['test_accuracy'].mean()
# 0.75265
# F1 skorunun ortalamasını alıyoruz.
cv_results['test_f1'].mean()
# 0.631
# ROC AUC skorunun ortalamasını alıyoruz.
cv_results['test_roc_auc'].mean()
# 0.7987

# Hiperparametre arama ızgarası
# XGBoost için optimize edilecek parametreleri belirliyoruz.
xgboost_params = {"learning_rate": [0.1, 0.01], # Adım boyutu
                  "max_depth": [5, 8], # Ağaç derinliği
                  "n_estimators": [100, 500, 1000], # Ağaç sayısı
                  "colsample_bytree": [0.7, 1]} # Her ağaçta kullanılacak değişken oranı

# GridSearchCV işlemi
# En iyi parametreleri bulmak için arama yapıyoruz.
xgboost_best_grid = GridSearchCV(xgboost_model, xgboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

# Final modelin kurulması
# Bulunan en iyi parametrelerle final modelimizi oluşturuyoruz.
xgboost_final = xgboost_model.set_params(**xgboost_best_grid.best_params_, random_state=17).fit(X, y)

# Final modelin CV ile değerlendirilmesi
# Optimize edilmiş modelin performansını tekrar 5 katlı CV ile ölçüyoruz.
cv_results = cross_validate(xgboost_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()



################################################
# LightGBM
################################################
# Microsoft tarafından geliştirilen, ağaç tabanlı bir gradient boosting framework'üdür.
# Leaf-wise (yaprak odaklı) büyüme stratejisi izler, bu da daha derin ağaçlar ve daha iyi doğruluk sağlar.
# Büyük veri setlerinde çok hızlıdır.

# Model nesnesinin oluşturulması
lgbm_model = LGBMClassifier(random_state=17)
lgbm_model.get_params()

# Modelin CV ile değerlendirilmesi
# 5 katlı çapraz doğrulama ile modelin başarısını ölçüyoruz.
cv_results = cross_validate(lgbm_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

# Doğruluk (Accuracy) skorunun ortalamasını alıyoruz.
cv_results['test_accuracy'].mean()
# F1 skorunun ortalamasını alıyoruz.
cv_results['test_f1'].mean()
# ROC AUC skorunun ortalamasını alıyoruz.
cv_results['test_roc_auc'].mean()

# Hiperparametre arama ızgarası
# LightGBM için optimize edilecek parametreleri belirliyoruz.
lgbm_params = {"learning_rate": [0.01, 0.1], # Öğrenme oranı
               "n_estimators": [100, 300, 500, 1000], # Ağaç sayısı
               "colsample_bytree": [0.5, 0.7, 1]} # Her ağaçta kullanılacak değişken oranı

# GridSearchCV işlemi
# En iyi parametreleri bulmak için arama yapıyoruz.
lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

# Final modelin kurulması
# Bulunan en iyi parametrelerle final modelimizi oluşturuyoruz.
lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

# Final modelin CV ile değerlendirilmesi
# Optimize edilmiş modelin performansını tekrar 5 katlı CV ile ölçüyoruz.
cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

# Hiperparametre yeni değerlerle (Daha geniş arama)
# İlk aramadan elde edilen sonuçlara göre arama uzayını daraltıyor veya genişletiyoruz.
lgbm_params = {"learning_rate": [0.01, 0.02, 0.05, 0.1],
               "n_estimators": [200, 300, 350, 400],
               "colsample_bytree": [0.9, 0.8, 1]}

# Yeni parametre aralığı ile tekrar GridSearchCV işlemi yapıyoruz.
lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

# Yeni en iyi parametrelerle final modelimizi güncelliyoruz.
lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

# Güncellenmiş modelin performansını tekrar ölçüyoruz.
cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()


# Hiperparametre optimizasyonu sadece n_estimators için.
# Diğer parametreleri sabit tutup sadece ağaç sayısını optimize ediyoruz.
lgbm_model = LGBMClassifier(random_state=17, colsample_bytree=0.9, learning_rate=0.01)

# Sadece n_estimators için geniş bir arama yapıyoruz.
lgbm_params = {"n_estimators": [200, 400, 1000, 5000, 8000, 9000, 10000]}

# GridSearchCV işlemi
lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

# En iyi n_estimators değeri ile final modelimizi oluşturuyoruz.
lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

# Final modelin performansını tekrar ölçüyoruz.
cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()


################################################
# CatBoost
################################################
# Yandex tarafından geliştirilen, kategorik değişkenleri otomatik işleyebilen bir kütüphanedir.
# Symmetric trees (simetrik ağaçlar) kullanır, bu da tahmin süresini hızlandırır.
# Genellikle varsayılan parametrelerle bile iyi sonuçlar verir.

# Model nesnesinin oluşturulması
# verbose=False ile eğitim sırasındaki çıktıları kapatıyoruz.
catboost_model = CatBoostClassifier(random_state=17, verbose=False)

# Modelin CV ile değerlendirilmesi
# 5 katlı çapraz doğrulama ile modelin başarısını ölçüyoruz.
cv_results = cross_validate(catboost_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

# Doğruluk (Accuracy) skorunun ortalamasını alıyoruz.
cv_results['test_accuracy'].mean()
# F1 skorunun ortalamasını alıyoruz.
cv_results['test_f1'].mean()
# ROC AUC skorunun ortalamasını alıyoruz.
cv_results['test_roc_auc'].mean()


# Hiperparametre arama ızgarası
# CatBoost için optimize edilecek parametreleri belirliyoruz.
catboost_params = {"iterations": [200, 500], # Ağaç sayısı
                   "learning_rate": [0.01, 0.1], # Öğrenme oranı
                   "depth": [3, 6]} # Ağaç derinliği


# GridSearchCV işlemi
# En iyi parametreleri bulmak için arama yapıyoruz.
catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

# Final modelin kurulması
# Bulunan en iyi parametrelerle final modelimizi oluşturuyoruz.
catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_, random_state=17).fit(X, y)

# Final modelin CV ile değerlendirilmesi
# Optimize edilmiş modelin performansını tekrar 5 katlı CV ile ölçüyoruz.
cv_results = cross_validate(catboost_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()


################################################
# Feature Importance
################################################
# Modellerin hangi değişkenlere ne kadar önem verdiğini görselleştiririz.
# Bu, modelin kararlarını yorumlamak için kritiktir.

def plot_importance(model, features, num=len(X), save=False):
    """
    Bu fonksiyon, eğitilmiş bir ağaç tabanlı modelin değişken önem düzeylerini görselleştirir.
    Hangi değişkenin modelin tahminlerinde daha etkili olduğunu anlamamızı sağlar.

    Args:
        model: Eğitilmiş makine öğrenmesi modeli (RandomForest, GBM, XGBoost vb.).
        features: Bağımsız değişkenlerin bulunduğu DataFrame (X).
        num: Görselleştirilecek en önemli değişken sayısı (varsayılan: tüm değişkenler).
        save: Grafiğin kaydedilip kaydedilmeyeceği bilgisi (True/False).
    """
    # Modelin feature_importances_ özelliğini kullanarak değişken önemlerini alıyoruz.
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    
    # Görselleştirme için grafik boyutunu ayarlıyoruz.
    plt.figure(figsize=(10, 10))
    # Yazı boyutunu ayarlıyoruz.
    sns.set(font_scale=1)
    # Değişkenleri önem sırasına göre sıralayıp barplot (çubuk grafik) ile çizdiriyoruz.
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    # Grafik başlığını ekliyoruz.
    plt.title('Features')
    # Grafik düzenini sıkılaştırıyoruz (elemanların birbirine girmemesi için).
    plt.tight_layout()
    # Grafiği ekrana basıyoruz.
    plt.show()
    # Eğer save=True ise grafiği dosyaya kaydediyoruz.
    if save:
        plt.savefig('importances.png')

# Random Forest modeli için değişken önem düzeylerini çizdiriyoruz.
plot_importance(rf_final, X)
# GBM modeli için değişken önem düzeylerini çizdiriyoruz.
plot_importance(gbm_final, X)
# XGBoost modeli için değişken önem düzeylerini çizdiriyoruz.
plot_importance(xgboost_final, X)
# LightGBM modeli için değişken önem düzeylerini çizdiriyoruz.
plot_importance(lgbm_final, X)
# CatBoost modeli için değişken önem düzeylerini çizdiriyoruz.
plot_importance(catboost_final, X)


################################
# Hyperparameter Optimization with RandomSearchCV (BONUS)
################################
# GridSearchCV tüm kombinasyonları denerken, RandomSearchCV rastgele kombinasyonlar seçer.
# Büyük arama uzaylarında daha hızlı sonuç almak için kullanılır.

rf_model = RandomForestClassifier(random_state=17)

# Geniş bir parametre aralığı tanımlanır
# np.random.randint ile belirli aralıklarda rastgele tamsayılar seçiyoruz.
rf_random_params = {"max_depth": np.random.randint(5, 50, 10),
                    "max_features": [3, 5, 7, "auto", "sqrt"],
                    "min_samples_split": np.random.randint(2, 50, 20),
                    "n_estimators": [int(x) for x in np.linspace(start=200, stop=1500, num=10)]}

# RandomizedSearchCV nesnesinin oluşturulması
# n_iter=100 ile 100 farklı rastgele kombinasyon deneneceğini belirtiyoruz.
rf_random = RandomizedSearchCV(estimator=rf_model,
                               param_distributions=rf_random_params,
                               n_iter=100,  # denenecek parametre sayısı
                               cv=3,
                               verbose=True,
                               random_state=42,
                               n_jobs=-1)

# Modeli eğitiyoruz.
rf_random.fit(X, y)

# En iyi parametreleri görüntülüyoruz.
rf_random.best_params_

# Final modelin en iyi parametrelerle kurulması
# Bulunan en iyi parametrelerle final modelimizi oluşturuyoruz.
rf_random_final = rf_model.set_params(**rf_random.best_params_, random_state=17).fit(X, y)

# Final modelin CV ile değerlendirilmesi
# Optimize edilmiş modelin performansını tekrar 5 katlı CV ile ölçüyoruz.
cv_results = cross_validate(rf_random_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

# Doğruluk (Accuracy) skorunun ortalamasını alıyoruz.
cv_results['test_accuracy'].mean()
# F1 skorunun ortalamasını alıyoruz.
cv_results['test_f1'].mean()
# ROC AUC skorunun ortalamasını alıyoruz.
cv_results['test_roc_auc'].mean()


################################
# Analyzing Model Complexity with Learning Curves (BONUS)
################################
# Model karmaşıklığının (örneğin max_depth, n_estimators) model başarısı üzerindeki etkisini görselleştiririz.
# Overfitting ve underfitting durumlarını analiz etmek için kullanılır.

def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    """
    Bu fonksiyon, modelin belirli bir hiperparametresinin farklı değerleri için eğitim ve doğrulama skorlarını hesaplar ve görselleştirir.
    Modelin aşırı öğrenme (overfitting) veya eksik öğrenme (underfitting) durumunu analiz etmeye yardımcı olur.

    Args:
        model: Eğitilecek makine öğrenmesi modeli.
        X: Bağımsız değişkenler.
        y: Bağımlı değişken.
        param_name: Değerleri değiştirilecek hiperparametre ismi (örn: 'max_depth').
        param_range: Hiperparametrenin alacağı değerler aralığı.
        scoring: Başarı değerlendirme metriği (varsayılan: 'roc_auc').
        cv: Çapraz doğrulama kat sayısı (varsayılan: 10).
    """
    # validation_curve fonksiyonu ile belirtilen parametre aralığında skorları hesaplıyoruz.
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    # Eğitim skorlarının ortalamasını alıyoruz.
    mean_train_score = np.mean(train_score, axis=1)
    # Test (doğrulama) skorlarının ortalamasını alıyoruz.
    mean_test_score = np.mean(test_score, axis=1)

    # Eğitim skorlarını çizdiriyoruz.
    plt.plot(param_range, mean_train_score,
             label="Training Score", color='b')

    # Doğrulama skorlarını çizdiriyoruz.
    plt.plot(param_range, mean_test_score,
             label="Validation Score", color='g')

    # Grafik başlığı ve eksen etiketlerini ekliyoruz.
    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    # Lejantı (açıklama kutusu) en uygun yere koyuyoruz.
    plt.legend(loc='best')
    # Grafiği gösteriyoruz.
    plt.show(block=True)


# İncelenecek hiperparametreler ve değer aralıklarını bir liste içinde tanımlıyoruz.
rf_val_params = [["max_depth", [5, 8, 15, 20, 30, None]],
                 ["max_features", [3, 5, 7, "auto"]],
                 ["min_samples_split", [2, 5, 8, 15, 20]],
                 ["n_estimators", [10, 50, 100, 200, 500]]]


# Model nesnesini oluşturuyoruz.
rf_model = RandomForestClassifier(random_state=17)

# Döngü ile her bir parametre için validation curve çizdiriyoruz.
for i in range(len(rf_val_params)):
    val_curve_params(rf_model, X, y, rf_val_params[i][0], rf_val_params[i][1])

# İlk parametre setinin değer aralığını görüntülüyoruz.
rf_val_params[0][1]