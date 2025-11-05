##############################################
# CUSTOMER LIFETIME VALUE (Müşteri Yaşam Boyu Değeri)
##############################################

# 1. Veri Hazırlama
# 2. Average Order Value (average_order_value = total_price / total_transaction)
# 3. Purchase Frequency (total transaction / total_number_of_customers)
# 4. Repeat Rate & Churn Rate (birden fazla alışveriş yapan müşteri sayısı / tüm müşteriler)
# 5. Profit Margin (profit_margin = total_price * 0.10)
# 6. Customer Value (customer_value = average_order_value * purchase_frequency)
# 7. Customer Lifetime Value (CLTV = (customer_value / churn_rate) x profit_margin)
# 8. Segmentlerin Oluşturulması
# 9. BONUS: Tüm İşlemlerin Fonksiyonlaştırılması


##############################################
# 1. Veri Hazırlama
##############################################

# Veri Seti Hikayesi
# https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

# Online Retail II isimli veri seti İngiltere merkezli online bir satış mağazasının
# 01/12/2009 - 09/12/2011 tarihleri arasındaki satışlarını içeriyor.

# Değişkenler
# InvoiceNo: Fatura numarası. Her işleme yani faturaya ait eşsiz numara. C ile başlıyorsa iptal edilen işlem
# StockCode: Ürün kodu. Her bir ürün için eşsiz numara.
# Description: Ürün ismi
# Quantity: Ürün adedi. Faturalardaki ürünlerden kaçar tane satıldığını ifade etmektedir.
# InvoiceDate: Fatura tarihi ve zamanı
# UnitPrice: Ürün fiyatı (Sterlin cinsinden)
# CustomerID: Eşsiz müşteri numarası
# Country: Ülke ismi. Müşterinin yaşadığı ülke.

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df_ = pd.read_excel("Datasets/Müşteri_Yaşam_Boyu_Değeri_Dataset/online_retail_II.xlsx", sheet_name='Year 2009-2010')
df = df_.copy()
df.head()
df.isnull().sum()

df = df[~df["Invoice"].str.contains("C", na=False)]
df.describe().T

df = df[df["Quantity"] > 0]

df.dropna(inplace=True)

df["Total_Price"] = df["Quantity"] * df["Price"]

cltv_c = df.groupby("Customer ID").agg({"Invoice": lambda x: x.nunique(),
                                        "Quantity": lambda x: x.sum(),
                                        "TotalPrice": lambda x: x.sum()})

cltv_c.columns = ["total_transaction", "total_unit", "total_price"]

#####################################################################################
# 2. Average Order Value (average_order_value = total_price / total_transaction)
#####################################################################################

cltv_c.head()

cltv_c["average_order_value"] =cltv_c["total_price"] / cltv_c["total_transaction"]


#####################################################################################
# 3. Purchase Frequency (total transaction / total_number_of_customers)
#####################################################################################

cltv_c.head()

cltv_c["purchase_frequency"] = cltv_c["total_transaction"] / cltv_c.shape[0]


#####################################################################################
# 4. Repeat Rate & Churn Rate (birden fazla alışveriş yapan müşteri sayısı / tüm müşteriler)
#####################################################################################

repeat_rate = cltv_c[cltv_c["total_transaction"] > 1].shape[0] / cltv_c.shape[0]

churn_rate = 1 - repeat_rate


#####################################################################################
# 5. Profit Margin (profit_margin = total_price * 0.10)
#####################################################################################

cltv_c["profit_margin"] = cltv_c["total_price"] * 10


#####################################################################################
# 6. Customer Value (customer_value = average_order_value * purchase_frequency)
#####################################################################################

cltv_c["customer_value"] = cltv_c["average_order_value"] * cltv_c["purchase_frequency"]


#####################################################################################
# 7. Customer Lifetime Value (CLTV = (customer_value / churn_rate) x profit_margin)
#####################################################################################

cltv_c["cltv"] = (cltv_c["customer_value"] / churn_rate) * cltv_c["profit_margin"]

cltv_c.sort_values(by="cltv", ascending=False).head()


#####################################################################################
# 8. Segmentlerin Oluşturulması
#####################################################################################

cltv_c.sort_values(by="cltv", ascending=False).head()
cltv_c.sort_values(by="cltv", ascending=False).tail()

cltv_c["segment"] = pd.qcut(cltv_c["cltv"], q=4, labels=["D", "C", "B", "A"])

cltv_c.sort_values(by="cltv", ascending=False).head()

cltv_c.groupby("segment").agg({"count", "mean", "sum"})

cltv_c.to_csv("cltv_c.csv")

#####################################################################################
# 9. BONUS: Tüm İşlemlerin Fonksiyonlaştırılması
#####################################################################################

def create_cltv_c(dataframe, profit=0.10):

    # Veriyi Hazırlama
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[(dataframe["Quantity"] > 0)]
    dataframe.dropna(inplace=True)
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    cltv_c = dataframe.groupby("Customer ID").agg({"Invoice": lambda x: x.nunique(),
                                                   "Quantity": lambda x: x.sum(),
                                                   "TotalPrice": lambda x: x.sum()})
    cltv_c.columns = ["total_transaction", "total_unit", "total_price"]

    # avg_order_value
    cltv_c["avg_order_value"] = cltv_c["total_price"] / cltv_c["total_transaction"]

    # purchase_frequency
    cltv_c["purchase_frequency"] = cltv_c["total_transaction"] / cltv_c.shape[0]

    # repeat rate & churn rate
    repeat_rate = cltv_c[cltv_c["total_transaction"] > 1].shape[0] / cltv_c.shape[0]
    churn_rate = 1 - repeat_rate

    # profit margin
    cltv_c["profit_margin"] = cltv_c["total_price"] * profit

    # customer value
    cltv_c["customer_value"] = cltv_c["average_order_value"] * cltv_c["purchase_frequency"]

    # Customer Lifetime Value
    cltv_c["cltv"] = (cltv_c["customer_value"] / churn_rate) * cltv_c["profit_margin"]

    # Segment
    cltv_c["segment"] = pd.qcut(cltv_c["scaled_cltv"], 4, labels=["D", "C", "B", "A"])

    return cltv_c


df = df_.copy()

clv = create_cltv_c(df)
