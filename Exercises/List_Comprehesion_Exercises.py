######################
# BURADA Kİ EGZERSİZLER İÇİN ÖNCELİKLE Çalışma_Ortamı_Ayarları içerikleri daha sonra Data_Structures içeriğini çalışınız.
######################

# GÖREV 1: List Comprehension yapısı kullanarak car_crashes verisindeki numeric değişkenlerin isimlerini büyük harfe
# çeviriniz ve başına NUM ekleyiniz.

# Beklenençıktı:
# # ['NUM_TOTAL',
#    'NUM_SPEEDING',
#    'NUM_ALCOHOL',
#    'NUM_NOT_DISTRACTED',
#    'NUM_NO_PREVIOUS',
#    'NUM_INS_PREMIUM',
#    'NUM_INS_LOSSES',
#    'ABBREV']

import seaborn as sns

df = sns.load_dataset("car_crashes")

new_columns = ["NUM_" + col.upper() if df[col].dtype != "object" else col for col in df.columns]

df.columns = new_columns

print(df.columns)


# GÖREV 2: List Comprehension yapısı kullanarak car_crashes verisinde isminde "no" barındırmayan değişkenlerin
# isimlerinin sonuna "FLAG" yazınız.

import seaborn as sns

df = sns.load_dataset("car_crashes")

new_columns = [col.upper() + "_FLAG" if "no" not in col.lower() else col for col in df.columns]

df.columns = new_columns

print(df.columns)


# GÖREV 3: List Comprehension yapısı kullanarak aşağıda verilen değişken isimlerinden FARKLI olan değişkenlerin isimlerini
# seçiniz ve yeni bir data frame oluşturunuz.

# og_list = ["abbrev", "no_previous"]

import seaborn as sns

df = sns.load_dataset("car_crashes")

og_list = ["abbrev", "no_previous"]

new_cols = [col for col in df.columns if col not in og_list]

new_df = df[new_cols]

print(new_df.head())
