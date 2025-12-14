######################################################
# Diabetes Prediction with Logistic Regression
######################################################

# İş Problemi:

# Özellikleri belirtildiğinde kişilerin diyabet hastası olup
# olmadıklarını tahmin edebilecek bir makine öğrenmesi
# modeli geliştirebilir misiniz?

# Veri seti ABD'deki Ulusal Diyabet-Sindirim-Böbrek Hastalıkları Enstitüleri'nde tutulan büyük veri setinin
# parçasıdır. ABD'deki Arizona Eyaleti'nin en büyük 5. şehri olan Phoenix şehrinde yaşayan 21 yaş ve üzerinde olan
# Pima Indian kadınları üzerinde yapılan diyabet araştırması için kullanılan verilerdir. 768 gözlem ve 8 sayısal
# bağımsız değişkenden oluşmaktadır. Hedef değişken "outcome" olarak belirtilmiş olup; 1 diyabet test sonucunun
# pozitif oluşunu, 0 ise negatif oluşunu belirtmektedir.

# Değişkenler
# Pregnancies: Hamilelik sayısı
# Glucose: Glikoz.
# BloodPressure: Kan basıncı.
# SkinThickness: Cilt Kalınlığı
# Insulin: İnsülin.
# BMI: Beden kitle indeksi.
# DiabetesPedigreeFunction: Soyumuzdaki kişilere göre diyabet olma ihtimalimizi hesaplayan bir fonksiyon.
# Age: Yaş (yıl)
# Outcome: Kişinin diyabet olup olmadığı bilgisi. Hastalığa sahip (1) ya da değil (0)


# 1. Exploratory Data Analysis
# 2. Data Preprocessing
# 3. Model & Prediction
# 4. Model Evaluation
# 5. Model Validation: Holdout
# 6. Model Validation: 10-Fold Cross Validation
# 7. Prediction for A New Observation


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
from sklearn.model_selection import train_test_split, cross_validate

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)



######################################################
# Exploratory Data Analysis
######################################################

# Veri setinin okunması
df = pd.read_csv("datasets/diabetes.csv")

##########################
# Target'ın Analizi
##########################

# Hedef değişkenin (Outcome) dağılımının incelenmesi
# 1: Diyabet, 0: Sağlıklı
df["Outcome"].value_counts()

# Sınıf dağılımının görselleştirilmesi
sns.countplot(x="Outcome", data=df)
plt.show()

# Sınıfların yüzdelik oranları
100 * df["Outcome"].value_counts() / len(df)

##########################
# Feature'ların Analizi
##########################

df.head()

# Kan basıncı değişkeninin histogramı
df["BloodPressure"].hist(bins=20)
plt.xlabel("BloodPressure")
plt.show()

# Sayısal değişkenlerin histogramını çizen fonksiyon
def plot_numerical_col(dataframe, numerical_col):
    dataframe[numerical_col].hist(bins=20)
    plt.xlabel(numerical_col)
    plt.show(block=True)


for col in df.columns:
    plot_numerical_col(df, col)

# Hedef değişken dışındaki tüm değişkenlerin listesi
cols = [col for col in df.columns if "Outcome" not in col]


# for col in cols:
#     plot_numerical_col(df, col)

df.describe().T

##########################
# Target vs Features
##########################

# Diyabet durumuna göre ortalama hamilelik sayısı
df.groupby("Outcome").agg({"Pregnancies": "mean"})

# Hedef değişkene göre sayısal değişkenlerin ortalamasını alan fonksiyon
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

# Tüm sayısal değişkenlerin hedef değişkene göre analizi
for col in cols:
    target_summary_with_num(df, "Outcome", col)



######################################################
# Data Preprocessing (Veri Ön İşleme)
######################################################
df.shape
df.head()

# Eksik değer kontrolü
df.isnull().sum()

df.describe().T

# Aykırı değer kontrolü
for col in cols:
    print(col, check_outlier(df, col))

# Insulin değişkenindeki aykırı değerlerin baskılanması
replace_with_thresholds(df, "Insulin")

# Değişkenlerin ölçeklendirilmesi (RobustScaler: Aykırı değerlere dayanıklı ölçeklendirme)
for col in cols:
    df[col] = RobustScaler().fit_transform(df[[col]])

df.head()


######################################################
# Model & Prediction
######################################################

# Bağımlı ve bağımsız değişkenlerin ayrılması
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

# Lojistik Regresyon modelinin kurulması
log_model = LogisticRegression().fit(X, y)

# Modelin sabit değeri (bias)
log_model.intercept_
# Modelin katsayıları (weights)
log_model.coef_

# Tahmin yapılması
y_pred = log_model.predict(X)

y_pred[0:10]

y[0:10]


######################################################
# Model & Prediction
######################################################

y = df["Outcome"]

X = df.drop(["Outcome"], axis=1)

log_model = LogisticRegression().fit(X, y)
log_model.intercept_
log_model.coef_

y_pred = log_model.predict(X)
y_pred[0:10]

y[0:10]








######################################################
# Model Evaluation
######################################################

# Karmaşıklık Matrisi (Confusion Matrix) görselleştirmesi
def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()

plot_confusion_matrix(y, y_pred)

# Başarı metriklerinin (Accuracy, Precision, Recall, F1) raporlanması
print(classification_report(y, y_pred))


# Accuracy: 0.78
# Precision: 0.74
# Recall: 0.58
# F1-score: 0.65

# ROC AUC (Receiver Operating Characteristic - Area Under Curve)
# Modelin sınıflandırma başarısını ölçen bir metrik (1'e ne kadar yakınsa o kadar iyi)
y_prob = log_model.predict_proba(X)[:, 1]
roc_auc_score(y, y_prob)
# 0.83939


######################################################
# Model Validation: Holdout
######################################################

# Veri setinin eğitim ve test seti olarak ayrılması (%80 Eğitim, %20 Test)
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20, random_state=17)

# Modelin eğitim seti üzerinde kurulması
log_model = LogisticRegression().fit(X_train, y_train)

# Test seti üzerinde tahmin yapılması
y_pred = log_model.predict(X_test)
# Test seti için olasılık değerlerinin hesaplanması (ROC AUC için gerekli)
y_prob = log_model.predict_proba(X_test)[:, 1]

# Test seti başarı metrikleri
print(classification_report(y_test, y_pred))

# Accuracy: 0.78
# Precision: 0.74
# Recall: 0.58
# F1-score: 0.65

# Accuracy: 0.77
# Precision: 0.79
# Recall: 0.53
# F1-score: 0.63

# ROC Eğrisinin çizdirilmesi
plot_roc_curve(log_model, X_test, y_test)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()

# Test seti için AUC skoru
roc_auc_score(y_test, y_prob)


######################################################
# Model Validation: 10-Fold Cross Validation
######################################################

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

# Tüm veri ile modelin tekrar kurulması (CV işlemi kendi içinde bölme yapar)
log_model = LogisticRegression().fit(X, y)

# 5 Katlı Çapraz Doğrulama (5-Fold Cross Validation)
# Veri seti 5 parçaya bölünür, 4'ü ile eğitim 1'i ile test yapılır. Bu işlem 5 kez tekrarlanır.
cv_results = cross_validate(log_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])



# Accuracy: 0.78
# Precision: 0.74
# Recall: 0.58
# F1-score: 0.65

# Accuracy: 0.77
# Precision: 0.79
# Recall: 0.53
# F1-score: 0.63


# Elde edilen skorların ortalaması (Daha güvenilir bir başarı ölçümü)
cv_results['test_accuracy'].mean()
# Accuracy: 0.7721

cv_results['test_precision'].mean()
# Precision: 0.7192

cv_results['test_recall'].mean()
# Recall: 0.5747

cv_results['test_f1'].mean()
# F1-score: 0.6371

cv_results['test_roc_auc'].mean()
# AUC: 0.8327

######################################################
# Prediction for A New Observation
######################################################

X.columns

# Rastgele bir kullanıcı seçimi
random_user = X.sample(1, random_state=45)

# Seçilen kullanıcı için diyabet tahmini
log_model.predict(random_user)