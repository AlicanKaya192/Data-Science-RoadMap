
##################################################
# List Comprehensions
##################################################

# ###############################################
# # GÖREV 1: List Comprehension yapısı kullanarak car_crashes verisindeki numeric değişkenlerin isimlerini büyük harfe çeviriniz ve başına NUM ekleyiniz.
# ###############################################
#
# # Beklenen Çıktı
#
# # ['NUM_TOTAL',
# #  'NUM_SPEEDING',
# #  'NUM_ALCOHOL',
# #  'NUM_NOT_DISTRACTED',
# #  'NUM_NO_PREVIOUS',
# #  'NUM_INS_PREMIUM',
# #  'NUM_INS_LOSSES',
# #  'ABBREV']
#
# # Notlar:
# # Numerik olmayanların da isimleri büyümeli.
# # Tek bir list comp yapısı ile yapılmalı.

import warnings
warnings.filterwarnings("ignore")

import seaborn as sns
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df = sns.load_dataset("car_crashes")
df.head()
df.columns
df.info()
df.dtypes


# DÖNGÜ HALİ
l = []
for col in df.columns:
    if df[col].dtype != "O":
        l.append("NUM_" + col.upper())
    else:
        l.append(col.upper())

print(l)


# LIST COMP.
["NUM_" + col.upper() if df[col].dtype != "O" else col.upper() for col in df.columns]



# ###############################################
# # GÖREV 2: List Comprehension yapısı kullanarak car_crashes verisindeki isminde "no" barındırmayan değişkenlerin isimlerininin sonuna "FLAG" yazınız.
# ###############################################
#
# # Notlar:
# # Tüm değişken isimleri büyük olmalı.
# # Tek bir list comp ile yapılmalı.
#
# # Beklenen çıktı:
#
# # ['TOTAL_FLAG',
# #  'SPEEDING_FLAG',
# #  'ALCOHOL_FLAG',
# #  'NOT_DISTRACTED',
# #  'NO_PREVIOUS',
# #  'INS_PREMIUM_FLAG',
# #  'INS_LOSSES_FLAG',
# #  'ABBREV_FLAG']


"an" in "Furkan"  # True
"x" not in "Furkan"  # True
"f" in "Furkan"  # False => BÜYÜK KÜÇÜK HARF DUYARLILIĞI


# DÖNGÜ
l = []
for col in df.columns:
    if "no" not in col:
        l.append(col.upper() + "_FLAG")
    else:
        l.append(col.upper())
print(l)


# LIST COMP.
[col.upper() + "_FLAG" if "no" not in col else col.upper() for col in df.columns]


# ###############################################
# # Görev 3: List Comprehension yapısı kullanarak aşağıda verilen değişken isimlerinden FARKLI olan değişkenlerin isimlerini seçiniz ve yeni bir dataframe oluşturunuz.
# ###############################################
#
#
# # Notlar:
# # Önce yukarıdaki listeye göre list comprehension kullanarak new_cols adında yeni liste oluşturunuz.
# # Sonra df[new_cols] ile bu değişkenleri seçerek yeni bir df oluşturunuz adını new_df olarak isimlendiriniz.
#
# # Beklenen çıktı:
#
# #    total  speeding  alcohol  not_distracted  ins_premium  ins_losses
# # 0 18.800     7.332    5.640          18.048      784.550     145.080
# # 1 18.100     7.421    4.525          16.290     1053.480     133.930
# # 2 18.600     6.510    5.208          15.624      899.470     110.350
# # 3 22.400     4.032    5.824          21.056      827.340     142.390
# # 4 12.000     4.200    3.360          10.920      878.410     165.630
#

# DÖNGÜ HALİ
l = []
og_list = ["abbrev", "no_previous"]
for col in df.columns:
    if col not in og_list:
        l.append(col)
print(l)


# List Comp.
new_cols = [col for col in df.columns if col not in og_list]
# # BONUS - new_cols oluşturmayı pandas'tan drop ile yapabilirsiniz.
# df.columns
# new_cols = df.columns.drop(og_list)
# new_cols = df.columns.drop(["abbrev", "no_previous"])


new_df = df[new_cols]   # df[new_cols]  =>  df[ ['total', 'speeding', 'alcohol', 'not_distracted', 'ins_premium', 'ins_losses'] ]
new_df.head()



###############################
# BONUS - Zaman Karşılaştırması
###############################
# 1. soru için
from time import time
t1 = time()
l = []
for col in df.columns:
    if df[col].dtype != "O":
        l.append("NUM_" + col.upper())
    else:
        l.append(col.upper())

t2 = time()

["NUM_" + col.upper() if df[col].dtype != "O" else col.upper() for col in df.columns]

t3 = time()

print("Loop Tamamlanma Süresi:", t2-t1)
print("List Comp. Tamamlanma Süresi:", t3-t2)
print("Hız farkı:", (t2-t1)/(t3-t2))


