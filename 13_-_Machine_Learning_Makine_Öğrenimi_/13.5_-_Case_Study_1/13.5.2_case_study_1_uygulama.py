import pandas as pd
import numpy as np

# --- ADIM 1: Veri Setini Hazırlama ---
# PDF dosyasının 3. sayfasındaki [cite: 16] ham verileri buraya aktarıyoruz.
# x: Deneyim Yılı, y: Gerçek Maaş
data = {
    'Deneyim_Yili_x': [5, 7, 3, 3, 2, 7, 3, 10, 6, 4, 8, 1, 1, 9, 1],
    'Maas_y': [600, 900, 550, 500, 400, 950, 540, 1200, 900, 550, 1100, 460, 400, 1000, 380]
}

# Verileri Pandas DataFrame formatına çeviriyoruz (Tablo görünümü için)
df = pd.DataFrame(data)

# --- ADIM 2: Model Denklemini Kurma ---
# PDF sayfa 3'te [cite: 10] belirtilen sabit (bias) ve ağırlık (weight) değerleri:
bias = 275
weight = 90

# Doğrusal Regresyon Formülü: y' = b + wx [cite: 10]
# Yani: Tahmini Maaş = 275 + (90 * Deneyim Yılı)
def model_tahmin(x):
    return bias + (weight * x)

# --- ADIM 3: Tahmin ve Hata Hesaplamaları ---
# Şimdi tablodaki her bir deneyim yılı için maaş tahmini yapalım [cite: 11]

# 1. Tahmin Sütunu (y')
df['Maas_Tahmini_y_pred'] = df['Deneyim_Yili_x'].apply(model_tahmin)

# 2. Hata Sütunu (Gerçek - Tahmin) (y - y') 
df['Hata'] = df['Maas_y'] - df['Maas_Tahmini_y_pred']

# 3. Hata Kareleri Sütunu (y - y')^2 
# Bu değerler MSE hesaplarken lazım olacak. Negatif hatalardan kurtulmak için karesini alıyoruz.
df['Hata_Karesi'] = df['Hata'] ** 2

# 4. Mutlak Hata Sütunu |y - y'| 
# Bu değerler MAE hesaplarken lazım olacak. Sadece hatanın büyüklüğüne bakıyoruz.
df['Mutlak_Hata'] = df['Hata'].abs()

# Tablonun son halini görelim (PDF Sayfa 4'teki tablonun dolu hali)
print("--- Doldurulmuş Veri Tablosu ---")
print(df)
print("\n" + "-"*30 + "\n")

# --- ADIM 4: Hata Metriklerinin Hesaplanması ---
# Modelin başarısını ölçmek için MSE, RMSE ve MAE skorlarını hesaplıyoruz [cite: 12]

n = len(df) # Gözlem sayısı (n) [cite: 25]

# 1. MSE (Mean Squared Error) - Hata Kareler Ortalaması
# Formül: Toplam(Hata Kareleri) / n [cite: 24]
mse = df['Hata_Karesi'].mean()

# 2. RMSE (Root Mean Squared Error) - Kök Ortalama Kare Hata
# Formül: MSE'nin karekökü [cite: 26]
rmse = np.sqrt(mse)

# 3. MAE (Mean Absolute Error) - Ortalama Mutlak Hata
# Formül: Toplam(Mutlak Hata) / n [cite: 27]
mae = df['Mutlak_Hata'].mean()

# --- SONUÇLARI YAZDIRMA ---
print("--- Model Başarı Metrikleri ---")
print(f"MSE  (Hata Kareler Ortalaması): {mse:.2f}")
print(f"RMSE (Kök Ortalama Kare Hata): {rmse:.2f}")
print(f"MAE  (Ortalama Mutlak Hata)  : {mae:.2f}")

# Yorum:
print("\n--- Değerlendirme ---")
print(f"Bu model ortalama olarak gerçek maaşlardan yaklaşık {mae:.2f} birim sapmaktadır (MAE).")
print(f"Ancak büyük hataları daha çok cezalandıran RMSE değerimiz {rmse:.2f} çıkmıştır.")