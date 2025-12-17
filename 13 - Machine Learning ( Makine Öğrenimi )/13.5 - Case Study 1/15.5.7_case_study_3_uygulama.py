# ============================================================
# TELCO CHURN PREDICTION
# ============================================================

# ------------------------------------------------------------
# 1. GEREKLİ KÜTÜPHANELER
# ------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

import missingno as msno

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

# ------------------------------------------------------------
# 2. VERİ SETİNİN OKUNMASI
# ------------------------------------------------------------
# PDF'te anlatılan Telco Churn veri seti okunur
df = pd.read_csv("Datasets ( Genel )/Telco-Customer-Churn.csv")

# ------------------------------------------------------------
# GÖREV 1 : KEŞİFÇİ VERİ ANALİZİ (EDA)
# ------------------------------------------------------------

# ------------------------------------------------------------
# Adım 1: Numerik ve kategorik değişkenleri yakalama
# ------------------------------------------------------------
cat_cols = [col for col in df.columns if df[col].dtype == "O"]
num_cols = [col for col in df.columns if df[col].dtype != "O" and col != "Churn"]

# ------------------------------------------------------------
# Adım 2: Tip hatası olan değişkenleri düzeltme
# ------------------------------------------------------------
# TotalCharges sayısal olmalı ama string gelmiştir
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# ------------------------------------------------------------
# Adım 3: Numerik ve kategorik değişkenlerin dağılımı
# ------------------------------------------------------------
for col in num_cols:
    df[col].hist(bins=20)
    plt.title(col)
    plt.show()

for col in cat_cols:
    print(df[col].value_counts())
    print("-" * 40)

# ------------------------------------------------------------
# Adım 4: Kategorik değişkenler ile hedef değişken ilişkisi
# ------------------------------------------------------------
for col in cat_cols:
    print(pd.crosstab(df[col], df["Churn"], normalize="index"))
    print("-" * 50)

# ------------------------------------------------------------
# Adım 5: Aykırı gözlem kontrolü
# ------------------------------------------------------------
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    print(col, "aykırı var mı:", df[(df[col] < lower) | (df[col] > upper)].any().any())

# ------------------------------------------------------------
# Adım 6: Eksik gözlem analizi
# ------------------------------------------------------------
msno.matrix(df)
plt.show()

# ------------------------------------------------------------
# GÖREV 2 : FEATURE ENGINEERING
# ------------------------------------------------------------

# ------------------------------------------------------------
# Adım 1: Eksik değer işlemleri
# ------------------------------------------------------------
# TotalCharges eksik değerleri medyan ile doldurulur
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# ------------------------------------------------------------
# Adım 2: Yeni değişkenler oluşturma
# ------------------------------------------------------------
# Müşteri yeni mi eski mi
df["NEW_CUSTOMER"] = np.where(df["tenure"] < 12, "Yes", "No")

# Ortalama aylık harcama
df["AVG_CHARGE"] = df["TotalCharges"] / (df["tenure"] + 1)

# ------------------------------------------------------------
# Adım 3: Encoding işlemleri
# ------------------------------------------------------------
# Hedef değişken binary hale getirilir
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Label Encoding uygulanacak değişkenler
binary_cols = [col for col in df.columns if df[col].nunique() == 2 and col != "Churn"]

le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

# One-Hot Encoding
df = pd.get_dummies(df, drop_first=True)

# ------------------------------------------------------------
# Adım 4: Numerik değişkenlerin standartlaştırılması
# ------------------------------------------------------------
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# ------------------------------------------------------------
# GÖREV 3 : MODELLEME
# ------------------------------------------------------------

# ------------------------------------------------------------
# Adım 1: Veri setinin ayrılması
# ------------------------------------------------------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# ------------------------------------------------------------
# Adım 2: Modellerin kurulması ve accuracy karşılaştırması
# ------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(name, "Accuracy:", accuracy_score(y_test, y_pred))

# ------------------------------------------------------------
# Adım 3: En iyi modeller için hiperparametre optimizasyonu
# ------------------------------------------------------------

rf_params = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5]
}

rf = RandomForestClassifier(random_state=42)

rf_cv = GridSearchCV(
    rf,
    rf_params,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

rf_cv.fit(X_train, y_train)

# En iyi parametreler
print("En iyi RF parametreleri:", rf_cv.best_params_)

# Optimize edilmiş model
final_model = rf_cv.best_estimator_

y_pred_final = final_model.predict(X_test)
print("Final Model Accuracy:", accuracy_score(y_test, y_pred_final))
