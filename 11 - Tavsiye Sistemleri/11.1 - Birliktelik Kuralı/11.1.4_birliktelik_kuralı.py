############################################
# ASSOCIATION RULE LEARNING (BİRLİKTELİK KURALI ÖĞRENİMİ)
############################################

# 1. Veri Ön İşleme
# 2. ARL Veri Yapısını Hazırlama (Invoice-Product Matrix)
# 3. Birliktelik Kurallarının Çıkarılması
# 4. Çalışmanın Scriptini Hazırlama
# 5. Sepet Aşamasındaki Kullanıcılara Ürün Önerisinde Bulunmak

############################################
# 1. Veri Ön İşleme
############################################

# Gerekli kütüphanelerin yüklenmesi ve ayarların yapılması
# !pip install mlxtend
import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
# Çıktının tek bir satırda olmasını sağlar, böylece daha okunaklı olur.
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

# Veri setinin okunması
# https://archive.ics.uci.edu/ml/datasets/Online+Retail+II
# Online Retail II veri seti, 2010-2011 yıllarını kapsayan bir e-ticaret verisidir.
df_ = pd.read_excel("Datasets ( Genel )/online_retail_II.xlsx",
                    sheet_name="Year 2010-2011")
df = df_.copy()
df.head()

# Veri setini tanıma: Betimsel istatistikler ve eksik değer kontrolü
df.describe().T
df.isnull().sum()
df.shape

def retail_data_prep(dataframe):
    """
    Veri ön işleme adımlarını gerçekleştirir.
    
    Parameters:
    dataframe (pd.DataFrame): İşlenecek veri seti.
    
    Returns:
    pd.DataFrame: Ön işleme yapılmış veri seti.
    """
    # Eksik değerlerin silinmesi
    dataframe.dropna(inplace=True)
    # İade faturalarının (Invoice ID'si 'C' ile başlayanlar) çıkarılması
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    # Miktar ve fiyatın 0'dan büyük olduğu gözlemlerin seçilmesi
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    return dataframe

df = retail_data_prep(df)


def outlier_thresholds(dataframe, variable):
    """
    Belirtilen değişken için aykırı değer eşiklerini hesaplar.
    
    Parameters:
    dataframe (pd.DataFrame): Veri seti.
    variable (str): Eşik değerleri hesaplanacak değişken ismi.
    
    Returns:
    low_limit (float): Alt sınır.
    up_limit (float): Üst sınır.
    """
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    """
    Belirtilen değişkendeki aykırı değerleri eşik değerleri ile baskılar.
    
    Parameters:
    dataframe (pd.DataFrame): Veri seti.
    variable (str): Baskılanacak değişken ismi.
    """
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def retail_data_prep(dataframe):
    """
    Veri ön işleme adımlarını gerçekleştirir (Aykırı değer baskılama dahil).
    
    Parameters:
    dataframe (pd.DataFrame): İşlenecek veri seti.
    
    Returns:
    pd.DataFrame: Ön işleme yapılmış veri seti.
    """
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe

df = retail_data_prep(df)
df.isnull().sum()
df.describe().T


############################################
# 2. ARL Veri Yapısını Hazırlama (Invoice-Product Matrix)
############################################

# Hedefimiz, faturaların satırlarda, ürünlerin sütunlarda olduğu ve 
# eğer o faturada o ürün varsa 1, yoksa 0 yazan bir matris oluşturmak.

df.head()

# Örnek Matris Yapısı:
# Description   NINE DRAWER OFFICE TIDY   SET 2 TEA TOWELS I LOVE LONDON    SPACEBOY BABY GIFT SET
# Invoice
# 536370                              0                                 1                       0
# 536852                              1                                 0                       1
# 536974                              0                                 0                       0
# 537065                              1                                 0                       0
# 537463                              0                                 0                       1

# Fransa'daki işlemleri filtreleyelim (Veri seti büyük olduğu için tek bir ülke üzerinden gidiyoruz)
df_fr = df[df['Country'] == "France"]

# Fatura ve ürün bazında toplam miktarları görelim
df_fr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).head(20)

# Pivot table benzeri bir yapı oluşturmak için unstack kullanıyoruz
df_fr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().iloc[0:5, 0:5]

# NaN değerleri 0 ile dolduruyoruz
df_fr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().fillna(0).iloc[0:5, 0:5]

# Miktarları 1 ve 0'a dönüştürüyoruz (Sepette var mı yok mu?)
df_fr.groupby(['Invoice', 'StockCode']). \
    agg({"Quantity": "sum"}). \
    unstack(). \
    fillna(0). \
    applymap(lambda x: 1 if x > 0 else 0).iloc[0:5, 0:5]


def create_invoice_product_df(dataframe, id=False):
    """
    Fatura-Ürün matrisini oluşturur.
    
    Parameters:
    dataframe (pd.DataFrame): Veri seti.
    id (bool): Ürün isimleri yerine StockCode kullanılsın mı?
    
    Returns:
    pd.DataFrame: Fatura-Ürün matrisi (1 ve 0'lardan oluşan).
    """
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)

fr_inv_pro_df = create_invoice_product_df(df_fr)

fr_inv_pro_df = create_invoice_product_df(df_fr, id=True)


def check_id(dataframe, stock_code):
    """
    Verilen StockCode'a ait ürün ismini (Description) yazdırır.
    
    Parameters:
    dataframe (pd.DataFrame): Veri seti.
    stock_code (int/str): Ürün kodu.
    """
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)


check_id(df_fr, 10120)

############################################
# 3. Birliktelik Kurallarının Çıkarılması
############################################

# Apriori algoritması ile sık geçen ürün birlikteliklerini (itemsets) buluyoruz.
# min_support=0.01: En az %1 oranında birlikte görülen ürünleri al.
frequent_itemsets = apriori(fr_inv_pro_df,
                            min_support=0.01,
                            use_colnames=True)

# Support değerine göre sıralayalım
frequent_itemsets.sort_values("support", ascending=False)

# Birliktelik kurallarını çıkarıyoruz.
# metric="support": Support değerine göre filtreleme yap.
# min_threshold=0.01: Support değeri 0.01'den büyük olan kuralları getir.
rules = association_rules(frequent_itemsets,
                          metric="support",
                          min_threshold=0.01)

# Kuralları filtreleyerek daha anlamlı sonuçlara ulaşalım:
# Support > 0.05 (Birlikte görülme olasılığı %5'ten büyük)
# Confidence > 0.1 (X alındığında Y alınma olasılığı %10'dan büyük)
# Lift > 5 (X alındığında Y alınma olasılığı 5 kat artıyor)
rules[(rules["support"]>0.05) & (rules["confidence"]>0.1) & (rules["lift"]>5)]

# Örnek bir ürünün ismini kontrol edelim
check_id(df_fr, 21086)

# Filtrelenmiş kuralları confidence değerine göre sıralayalım
rules[(rules["support"]>0.05) & (rules["confidence"]>0.1) & (rules["lift"]>5)]. \
sort_values("confidence", ascending=False)

############################################
# 4. Çalışmanın Scriptini Hazırlama
############################################

# Bu bölümde, yukarıda parça parça yaptığımız işlemleri fonksiyonlaştırarak
# tek bir akış (pipeline) haline getiriyoruz.

def outlier_thresholds(dataframe, variable):
    """
    Aykırı değer eşiklerini hesaplar.
    """
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    """
    Aykırı değerleri eşik değerleri ile baskılar.
    """
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def retail_data_prep(dataframe):
    """
    Veri ön işleme işlemlerini yapar.
    """
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe


def create_invoice_product_df(dataframe, id=False):
    """
    Fatura-Ürün matrisini oluşturur.
    """
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)


def check_id(dataframe, stock_code):
    """
    StockCode'a karşılık gelen ürün ismini bulur.
    """
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)


def create_rules(dataframe, id=True, country="France"):
    """
    Verilen veri seti ve ülke için birliktelik kurallarını oluşturur.
    
    Parameters:
    dataframe (pd.DataFrame): Veri seti.
    id (bool): StockCode kullanılsın mı?
    country (str): Hangi ülke için kurallar oluşturulacak?
    
    Returns:
    pd.DataFrame: Oluşturulan birliktelik kuralları.
    """
    dataframe = dataframe[dataframe['Country'] == country]
    dataframe = create_invoice_product_df(dataframe, id)
    frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
    return rules

# Veriyi baştan okuyup scripti test edelim
df = df_.copy()

df = retail_data_prep(df)
rules = create_rules(df)

# Kuralları filtreleyip sıralayalım
rules[(rules["support"]>0.05) & (rules["confidence"]>0.1) & (rules["lift"]>5)]. \
sort_values("confidence", ascending=False)

############################################
# 5. Sepet Aşamasındaki Kullanıcılara Ürün Önerisinde Bulunmak
############################################

# Örnek Senaryo:
# Kullanıcı sepete bir ürün ekledi (Örnek ürün ID: 22492).
# Bu ürünü alanlar başka neler almış? (Association Rules kullanarak öneri yapacağız)

product_id = 22492
check_id(df, product_id)

# Kuralları 'lift' değerine göre sıralıyoruz. (Lift: Birlikteliğin gücü)
sorted_rules = rules.sort_values("lift", ascending=False)

recommendation_list = []

# Kurallar içinde gezerek, eğer antecedents (öncül) kısmında bizim ürünümüz varsa,
# consequents (ardıl) kısmındaki ürünü öneri listesine ekliyoruz.
for i, product in enumerate(sorted_rules["antecedents"]):
    for j in list(product):
        if j == product_id:
            recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

# İlk 3 öneriyi görelim
recommendation_list[0:3]

# Önerilen ürünlerin isimlerine bakalım
check_id(df, 22326)

def arl_recommender(rules_df, product_id, rec_count=1):
    """
    Verilen ürün ID'sine göre birliktelik kurallarını kullanarak ürün önerisinde bulunur.
    
    Parameters:
    rules_df (pd.DataFrame): Birliktelik kuralları veri seti.
    product_id (int/str): Referans ürün ID'si.
    rec_count (int): Kaç tane öneri yapılacağı.
    
    Returns:
    list: Önerilen ürünlerin ID listesi.
    """
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]


# Fonksiyonu test edelim
arl_recommender(rules, 22492, 1)
arl_recommender(rules, 22492, 2)
arl_recommender(rules, 22492, 3)