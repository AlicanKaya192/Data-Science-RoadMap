################################
# SORTING REVIEWS
################################

import pandas as pd
import math
import scipy.stats as st

# Pandas görüntüleme ayarlarını yapıyoruz.
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


################################
# Up-Down Diff Score = (up ratings) - (down ratings)
################################

# Yorum 1: 600 beğeni, 400 beğenmeme. Toplam 1000.
# Yorum 2: 5500 beğeni, 4500 beğenmeme. Toplam 10000.

# Basit fark alma yöntemi.
# Beğeni sayısından beğenmeme sayısını çıkarıyoruz.
def score_up_down_diff(up, down):
    return up - down

# Review 1 Score: 200
score_up_down_diff(600, 400)

# Review 2 Score: 1000
# Yorum 2 daha yüksek skor alıyor ama oranlara bakarsak Yorum 1 (%60) Yorum 2 (%55)'den daha iyi.
# Bu yöntem frekansı (sayıyı) ödüllendirir ama oranı göz ardı eder.
score_up_down_diff(5500, 4500)


################################
# Score = Average rating = (up ratings) / (all ratings)
################################

# Beğeni oranı hesabı.
def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

score_average_rating(600, 400)   # Review 1 Score: 0.60
score_average_rating(5500, 4500) # Review 2 Score: 0.55
# Burada oranlar doğru çalışıyor.

# Ancak frekans (oy sayısı) düşük olduğunda sorun çıkar.
# Review 1: 2 beğeni, 0 beğenmeme. Toplam 2. Oran: 1.0 (%100)
# Review 2: 100 beğeni, 1 beğenmeme. Toplam 101. Oran: 0.99 (%99)

score_average_rating(2, 0)     # Review 1 Score: 1.0
score_average_rating(100, 1)   # Review 2 Score: 0.99

# Sadece 2 kişinin beğendiği bir yorum, 100 kişinin beğendiği (1 kişi beğenmemiş) yorumdan
# daha üstte çıkıyor. Bu sosyal ispat (social proof) açısından yanlıştır.


################################
# Wilson Lower Bound Score
################################

# Bu yöntem, ikili (binary) ölçümler (beğendi/beğenmedi) için bir güven aralığı hesaplar.
# Bu güven aralığının alt sınırını (Lower Bound) skor olarak kabul eder.
# Yani %95 güvenle bu yorumun beğeni oranı EN AZ kaçtır? sorusuna cevap verir.
# Oy sayısı arttıkça aralık daralır ve alt sınır yukarı çıkar (güven artar).

def wilson_lower_bound(up, down, confidence=0.95):
    """ 
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
      Eğer skorlar 1 - 5 arasındaysa 1 - 3 negatif, 4 - 5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
      Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

      Parameters
        ----------
        up: int
            up count
        down: int
            down count
        confidence: float
            confidence
        
        Returns
        -------
        wilson score: float

     """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    wilson_score = (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)
    return wilson_score

# 2 oy alanın alt sınırı çok düşüktür (belirsizlik yüksek).
wilson_lower_bound(600, 400)   # Review 1 Score
wilson_lower_bound(5500, 4500) # Review 2 Score

wilson_lower_bound(2, 0)     # Review 1 Score
# 101 oy alanın alt sınırı daha yüksektir (belirsizlik az).
wilson_lower_bound(100, 1)   # Review 2 Score
# Sonuç olarak WLB, 100 oy alanı 2 oy alanın önüne geçirir. Doğru sıralama budur.


################################
# Case Study
################################

# Örnek bir veri seti oluşturuyoruz.
up = [15, 70, 14, 4, 2, 5, 8, 37, 21, 52, 28, 147, 61, 30, 23, 40, 37, 61, 54, 18, 12, 68]
down = [0, 2, 2, 2, 15, 2, 6, 5, 23, 8, 12, 2, 1, 1, 5, 1, 2, 6, 2, 0, 2, 2]
comments = pd.DataFrame({'up': up, 'down': down})

# score_pos_neg_diff: Basit farka göre hesaplama
comments['score_pos_neg_diff'] = comments.apply(lambda x: score_up_down_diff(x['up'], x['down']), axis=1)

# score_average_rating: Orana göre hesaplama
comments['score_average_rating'] = comments.apply(lambda x: score_average_rating(x['up'], x['down']), axis=1)

# wilson_lower_bound: Wilson skoruna göre hesaplama
comments['wilson_lower_bound'] = comments.apply(lambda x: wilson_lower_bound(x['up'], x['down']), axis=1)


# Wilson skoruna göre sıraladığımızda, hem oranı yüksek hem de oy sayısı güvenilir olanların
# en üstte yer aldığını görürüz.
comments.sort_values('wilson_lower_bound', ascending=False)