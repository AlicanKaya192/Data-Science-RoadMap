import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ======================================================
# GÖREV 1: CHURN (MÜŞTERİ TERKİ) ANALİZİ
# ======================================================
print("--- GÖREV 1: CHURN ANALİZİ SONUÇLARI ---")

# 1. Veri Setini Oluşturma (PDF Sayfa 3'teki tablo)
# Gerçek Değer: 1 (Churn), 0 (Non-Churn)
# Olasılık: Modelin verdiği 1 olma ihtimali
data = {
    'Gercek_Deger': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    'Olasilik':     [0.7, 0.8, 0.65, 0.9, 0.45, 0.5, 0.55, 0.35, 0.4, 0.25]
}

df = pd.DataFrame(data)

# 2. Tahmin Oluşturma (Eşik Değer: 0.5)
# Olasılık >= 0.5 ise 1 (Churn), değilse 0 (Non-Churn) diyoruz.
df['Tahmin'] = df['Olasilik'].apply(lambda x: 1 if x >= 0.5 else 0)

# 3. Confusion Matrix (Karmaşıklık Matrisi) Hesaplama
cm = confusion_matrix(df['Gercek_Deger'], df['Tahmin'])
tn, fp, fn, tp = cm.ravel()

print(f"Confusion Matrix:\n{cm}")
print(f"TP (Doğru Pozitif): {tp}, FP (Yanlış Pozitif): {fp}")
print(f"FN (Yanlış Negatif): {fn}, TN (Doğru Negatif): {tn}")

# 4. Metriklerin Hesaplanması
acc = accuracy_score(df['Gercek_Deger'], df['Tahmin'])
prec = precision_score(df['Gercek_Deger'], df['Tahmin'])
rec = recall_score(df['Gercek_Deger'], df['Tahmin'])
f1 = f1_score(df['Gercek_Deger'], df['Tahmin'])

print(f"Accuracy (Doğruluk) : {acc:.2f}")
print(f"Precision (Kesinlik): {prec:.2f}")
print(f"Recall (Duyarlılık) : {rec:.2f}")
print(f"F1 Skor             : {f1:.2f}")
print("\n" + "="*40 + "\n")


# ======================================================
# GÖREV 2: FRAUD (DOLANDIRICILIK) ANALİZİ
# ======================================================
print("--- GÖREV 2: FRAUD ANALİZİ VE YORUM ---")

# PDF Sayfa 4'teki matris değerlerini elle giriyoruz
# Tablo:
#                 Tahmin:1 (Fraud)   Tahmin:0 (Normal)
# Gerçek:1 (Fraud)       5                  5
# Gerçek:0 (Normal)      90                 900

tp_fraud = 5
fn_fraud = 5
fp_fraud = 90
tn_fraud = 900

# Toplam veri sayısı
toplam = tp_fraud + fn_fraud + fp_fraud + tn_fraud

# 1. Metrik Hesaplamaları (Manuel Formüllerle)
accuracy_f = (tp_fraud + tn_fraud) / toplam
precision_f = tp_fraud / (tp_fraud + fp_fraud)
recall_f = tp_fraud / (tp_fraud + fn_fraud)
f1_f = 2 * (precision_f * recall_f) / (precision_f + recall_f)

print(f"Accuracy (Doğruluk) : {accuracy_f:.3f} (%90.5 - Yüksek görünüyor ama yanıltıcı)")
print(f"Precision (Kesinlik): {precision_f:.3f} (Çok düşük!)")
print(f"Recall (Duyarlılık) : {recall_f:.3f} (Sadece %50 başarı)")
print(f"F1 Skor             : {f1_f:.3f}")

# 2. Yorum (Soruda istenen 'Gözden kaçırılan durum')
print("\n--- DEĞERLENDİRME / YORUM ---")
print("Veri Bilimi ekibinin gözden kaçırdığı durum: DENGESİZ VERİ SETİ (Imbalanced Data)")
print("- Model %90.5 doğruluk oranına sahip olsa da, dolandırıcılık tespitinde başarısızdır.")
print("- Precision (0.053): Modelin 'Dolandırıcı' dediği her 100 kişiden sadece 5'i gerçek dolandırıcıdır. Çok fazla yanlış alarm (False Positive) var.")
print("- Recall (0.500): Gerçek dolandırıcıların yarısını (5 tanesini) kaçırmışız (False Negative).")
print("- Sonuç: Fraud gibi nadir olaylarda Accuracy yerine F1 Skor, Precision ve Recall değerlerine odaklanılmalıdır.")