
################################################################################################
# Scoutium Yetenek Avcılığı Sınıflandırma Projesi
################################################################################################

# İş Problemi:
# Scout’lar tarafından izlenen futbolcuların özelliklerine verilen puanlara göre, oyuncuların sınıfının
# (average, highlighted) tahminlenmesi.

# Veri Seti Hikayesi:
# Veri seti Scoutium’dan maçlarda gözlemlenen futbolcuların özelliklerine göre scoutların değerlendirdiği
# futbolcuların özellik puanlarını ve futbolcuların puanlandığı maçtaki sınıfını içermektedir.

# scoutium_attributes.csv
# task_response_id: Bir scoutun bir maçta bir takımın kadrosundaki tüm oyunculara dair değerlendirmelerinin kümesi
# match_id: İlgili maçın id'si
# evaluator_id: Değerlendiricinin(scout'un) id'si
# player_id: İlgili oyuncunun id'si
# position_id: İlgili oyuncunun o maçta oynadığı pozisyonun id’si
# analysis_id: Bir scoutun bir maçta bir oyuncuya dair özellik değerlendirmelerini içeren küme
# attribute_id: Oyuncuların değerlendirildiği her bir özelliğin id'si
# attribute_value: Bir scoutun bir oyuncunun bir özelliğine verdiği değer(puan)

# scoutium_potential_labels.csv
# task_response_id: Bir scoutun bir maçta bir takımın kadrosundaki tüm oyunculara dair değerlendirmelerinin kümesi
# match_id: İlgili maçın id'si
# evaluator_id: Değerlendiricinin(scout'un) id'si
# player_id: İlgili oyuncunun id'si
# potential_label: Bir scoutun bir maçta bir oyuncuyla ilgili nihai kararını belirten etiket. (hedef değişken)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
import warnings
warnings.filterwarnings("ignore")

################################################################
# Adım 1: Veri Yükleme ve Birleştirme
################################################################

# Dosya yollarının belirlenmesi
attributes_path = "/Users/tuce/Desktop/Data-Science-RoadMap/Datasets ( Genel )/scoutium_attributes.csv"
labels_path = "/Users/tuce/Desktop/Data-Science-RoadMap/Datasets ( Genel )/scoutium_potential_labels.csv"

# CSV dosyalarının okunması (Ayıraç olarak noktalı virgül kullanıldı)
df_attributes = pd.read_csv(attributes_path, sep=";")
df_labels = pd.read_csv(labels_path, sep=";")

print("Attributes Shape:", df_attributes.shape)
print("Labels Shape:", df_labels.shape)

# Attributes veri setinin pivot table ile düzenlenmesi.
# Her satır bir oyuncu-maç-scout eşleşmesini temsil etmeli, sütunlar ise özellikler (attribute_id) olmalı.
df_pivot = df_attributes.pivot_table(index=["task_response_id", "match_id", "evaluator_id", "player_id", "position_id"],
                                     columns="attribute_id",
                                     values="attribute_value").reset_index()

# Sütun isimlerinin string'e çevrilmesi (attribute_id'ler sayısal kalmasın)
df_pivot.columns = [str(col) for col in df_pivot.columns]

print("Pivot Table Shape:", df_pivot.shape)

# Pivot tablosunun etiketlerle (potential_label) birleştirilmesi
df = pd.merge(df_pivot, df_labels, on=["task_response_id", "match_id", "evaluator_id", "player_id"], how="left")

print("Merged DataFrame Shape:", df.shape)
print(df.head())

################################################################
# Adım 2: Veri Temizleme ve Ön İşleme
################################################################

# Sadece 'average' ve 'highlighted' sınıflarını içeren satırların filtrelenmesi
print("Sınıf Dağılımı:\n", df["potential_label"].value_counts())
df = df[df["potential_label"].isin(["average", "highlighted"])]

# position_id sütunu da özellik olarak kullanılabilir.

# Bağımlı değişkenin (potential_label) sayısal hale getirilmesi
le = LabelEncoder()
df["potential_label"] = le.fit_transform(df["potential_label"])
# 0: average, 1: highlighted (Genellikle alfabetik sıra)
print("Encoded Classes:", le.classes_)

# Sayısal değişkenlerin belirlenmesi (ID'ler ve hedef değişken hariç)
num_cols = [col for col in df.columns if col not in ["task_response_id", "match_id", "evaluator_id", "player_id", "potential_label"]]

# Veri setinin X (bağımsız değişkenler) ve y (bağımlı değişken) olarak ayrılması
X = df[num_cols]
y = df["potential_label"]

print("X Shape:", X.shape)
print("y Shape:", y.shape)

################################################################
# Adım 3: Modelleme (Makine Öğrenmesi)
################################################################

# Verinin standartlaştırılması
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Modellerin tanımlanması
models = [
    ('RF', RandomForestClassifier(random_state=46)),
    ('GBM', GradientBoostingClassifier(random_state=46)),
    ('XGBoost', XGBClassifier(random_state=46, use_label_encoder=False, eval_metric='logloss')),
    ('LightGBM', LGBMClassifier(random_state=46, verbose=-1))
]

print("\n################# Base Models Cross Validation #################")
for name, model in models:
    cv_results = cross_validate(model, X_scaled, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
    print(f"########## {name} ##########")
    print(f"Accuracy: {cv_results['test_accuracy'].mean():.4f}")
    print(f"F1 Score: {cv_results['test_f1'].mean():.4f}")
    print(f"ROC AUC: {cv_results['test_roc_auc'].mean():.4f}")


################################################################
# Adım 4: Hiperparametre Optimizasyonu (Örnek: Random Forest)
################################################################

print("\n################# Hyperparameter Optimization (Random Forest) #################")

rf_model = RandomForestClassifier(random_state=46)

rf_params = {"max_depth": [5, 8, None],
             "max_features": [3, 5, 7, "auto"],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [100, 200, 500]}

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=1).fit(X_scaled, y)

print("RF Best Params:", rf_best_grid.best_params_)

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=46).fit(X_scaled, y)

cv_results = cross_validate(rf_final, X_scaled, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
print(f"########## RF Final Model ##########")
print(f"Accuracy: {cv_results['test_accuracy'].mean():.4f}")
print(f"F1 Score: {cv_results['test_f1'].mean():.4f}")
print(f"ROC AUC: {cv_results['test_roc_auc'].mean():.4f}")

################################################################
# Adım 5: Değişken Önem Düzeyleri
################################################################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

print("\nFeature Importances (Top 10):")
feature_imp = pd.DataFrame({'Value': rf_final.feature_importances_, 'Feature': X.columns})
print(feature_imp.sort_values("Value", ascending=False).head(10))

# plot_importance(rf_final, X) # Görselleştirme için yorumu kaldırabilirsiniz.
