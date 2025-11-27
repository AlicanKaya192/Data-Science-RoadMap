import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date

from numpy.distutils.conv_template import header
from pandas.io.pytables import dropna_doc
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: "%.3f" % x)
pd.set_option('display.width', 500)

def load():
    data = pd.read_csv("Datasets/Feature Engineering/Outliers/titanic.csv")
    return data

def load_application_train():
    data = pd.read_csv("Datasets/Feature Engineering/Outliers/application_train.csv")
    return data

###################################
# Label Encoding & Binary Encoding
###################################

# Label Encoding karşımıza Binary Encoding olarak da çıkabilir.
# E[er bir kategorik değişkenin 2 sınıfı varsa bu 0 - 1 olarak kodlanırsa buna binary encoding denir.
# Bir kategorik değişken label encoder a sokulursa ve ikiden fazla sınıfı varsa bu durumda label encoding yapılmış olur.
# Label Encoding > Binary Encoding. Genel ismi Label Encoding 'dir.


df = load()
df.head()
df["Sex"].head()

# Bir Label Encoder kullanmamızın, one-hot encoder kullanmamızın sebebi algoritmaların bizden beklediği bir standart format
# var, veriyi buna uygun hale getirmek. "Tek sebebi bu mu ?". "Hayır, tek sebebi bu değil."

# Örneğin; bazen one-hot encoding işlemlerinde amacımız bir kategorik değişkenin önemli olabilecek sınıfını
# değişkenleştirerek ona bir değer atfetmek olacaktır.

# Özetle; iki çerçeveden encoding işlemlerini değerlendiriyoruz.
# 1. Kullanacak olduğumuz modelleme tekniklerinin bizden beklediği bir standart var.
# 2. Bu standartla beraber bizim model tahmin performansını iyileştirebileceğimiz, geliştirebileceğimiz bazı noktalar var.
# Bunlardan dolayı bu işlemleri yapmak istiyoruz.

le = LabelEncoder()
# LabelEncoder nesnemizi getiriyoruz.
# Bu nesneyi fit_transform metodunu kullanarak cinsiyet değişkenine uygulamamız lazım.

le.fit_transform(df["Sex"])[0:5]
# le.fit_transoform() dediğimizde; önce bu label encoder nesnesini değişkene fit et, encoder uygula, ondan sonra
# değerlerini dönüştür. fit_transform() 2 basamaklı bir işlem.
# Birince basamakta ilgili dönüştürme işlemi yapılır. "Dönüştürdüm, ne yapayım ?" Eskisi var, yenisi var elimizde.
# Eskisi eski hali, yenisi dönüştürdüğüm hali. "Ne yapacağım bunları ?" sorusunun yanıtı için transform diyor ki,
# "Dönüştür, ikisi artık son hali olsun."

# DİKKAT!!! - "Bu değiştirme işlemi neye göre gerçekleşiyor ?". Alfabetik sıraya göre ilk gördüğü değere 0 değerini
# verir label encoder. Eğer burada 5 tane sınıf olsaydı aynı durum söz konusu olacaktı. Alfabetik sıraya göre
# 0, 1, 2, 3, 4, 5 şeklinde isimlendirme yapılacaktı.

# Diyelim ki; hangisine 0, hangisine 1 verdiğimizi unuttuk, bilmiyoruz. Bu, şu anlama geliyor; dönüştürme işleminden
# sonra bir yerlerde 0 'ın hangi sınıf olduğunu, 1 'in hangi sınıf olduğunu unuttuk. Öğrenme ihtiyacımız var.
# "Bunu nasıl tespit edebiliriz ?". inverse_transform() adında bir metodumuz var.

le.inverse_transform([0, 1])
# le, label encoder nesnesinin içerisindeki dönüştürme bilgileri bu sınıf yapısının içerisinde tutuluyor.
# le nesnesini kullanarak inverse_transform() dediğimizde "Şu değerin karşılığı neydi ?", "Bu değerin karşılığı neydi ?"
# diye sorduğumuzda, bu değerlerin karşılıkları gelecek.


def label_encoder(dataframe, binary_col):
    # LabelEncoder sınıfını içe aktar (kategorik verileri sayısal değerlere dönüştürmek için)
    labelencoder = LabelEncoder()

    # Belirtilen kolondaki kategorik verileri 0 ve 1 gibi sayısal değerlere dönüştür
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])

    # Güncellenmiş dataframe'i geri döndür
    return dataframe


df = load()

# Şimdi, veri setime bunu uygulamak istiyorum ama bir problemimiz var. "Nedir o ?". Bir tane değişken olduğunda bunu
# kolayca uyguladık. Yaygın problememiz ise, bunu ölçeklenebilir yapıyor olmak. "Elimizde eğer yüzlerce değişken varsa
# nasıl yapacağız ?". Bu durumda binary_col 'ları seçebiliriz. İki seçeneğimiz var.

# DİKKAT!!!  - Label Encoder uygulamak için;
# 1 - Şuan da gördüğümüz bu yöntemi uygulayabiliriz.
# 2 - One-hot encoder uygulayabiliriz.
# One-hot encoder uygularken get_dummies() metodunu kullanırız. get_dummies() metodunu kullanırken drop_first = True
# yaparsak iki sınıflı kategorik değişkenler de label encoder 'dan geçirilmiş olur. Bu burada dursun probleme dönelim.

# "Elimde yüzlerce değişken olduğunda ne yapacağız ?". İki sınıflı kategorik değişkenleri seçmenin bir yolunu bulsak.
# Bu yolu bulabilirsem ve iki sınıflı kategorik değişkenleri label encoder 'dan geçirirsek bu durumda problem çözülür.

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float] and df[col].nunique() == 2]
# DİKKAT!!! - Diyoruz ki; Değişkenin sütunlarında gez. Gezdiğin değişkenin tipine bak.
# Eğer bu değişkenin tipi integer ya da float değilse; DİKKAT!!! - Bir değişkenin tipi integer sa ve 2 sınıf varsa bununla
# ilgilenmiyoruz, zaten binary encode edilmiş o ve number unique sayısı 2 olanları seçiyoruz.

# Peki burada neden len(nunique()) kullanmadık nqunique() kullandık ? Çünkü len ile sorguladığımızda bize eksik değeri
# olan sınıfları da verir fakat nqunique() ile sorgualadığımız da bize eksik olmayan eşsiz değer sayısını verir.

# Diyelim ki; her şey yolunda ve binary_cols da 10 tane vardı, "Nasıl işlem yapacağız ?".
for col in binary_cols:
    label_encoder(df, col)

# Bu veri setinde sadece 1 değişken encode edildiği için anlaşılmamış veya etkileyici olmamış olabilir. Şimdi farklı bir
# veri seti ile aynı işlemleri yapalım.

df = load_application_train()
df.shape

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float] and df[col].nunique() == 2]

df[binary_cols].head()

for col in binary_cols:
    label_encoder(df, col)

# Burada EMERGENCYSTATE_MODE değişkenin eksik değerleri de doldurmuş. Böyle bir durumda ne yapmamız lazım ?
# EMERGENCYSTATE_MODE değişkeninde eksik değerler (NaN) vardı.
# LabelEncoder, NaN değerleri "bir kategori gibi" algılayıp otomatik olarak bir sınıfa dönüştürdü.
# (Örn: NaN -> 0, 'Yes' -> 1 veya tam tersi)
#
# Ancak NaN değerini bir kategori gibi işlemek doğru değildir. NaN, "bilinmiyor" anlamına gelir.
# Bu yüzden LabelEncoding işleminden önce kategorik değişkenlerdeki eksik değerleri doldurmamız gerekir.
#
# Ne yapmalıyız?
# 1) LabelEncoder uygulamadan önce eksik değerleri uygun bir yöntemle doldurmalıyız:
#    - df[col].fillna("Unknown", inplace=True)
#    - ya da en sık görülen sınıf ile doldurabiliriz (mode).
#
# 2) Ardından LabelEncoder uyguladığımızda NaN artık kategori gibi davranmaz.
#
# Örnek:
# df['EMERGENCYSTATE_MODE'].fillna("Unknown", inplace=True)
# label_encoder(df, "EMERGENCYSTATE_MODE")
#
# ÖZET:
# Eksik değer var + LabelEncoder = yanlış kategori üretir.
# Eksik değerleri önce doldur -> sonra LabelEncode et.


###################################
# ÖZET
###################################

# LabelEncoder sadece 2 sınıflı (binary) kategorik değişkenlerde kullanılmalıdır.
# Çünkü çok sınıflı (örneğin 10, 50, 100 sınıf) değişkenlerde LabelEncoder sıralı bir ilişki yaratır
# (örn: A=0, B=1, C=2...), bu da modelin yanlış öğrenmesine ve performansın düşmesine neden olur.
# 2'den fazla sınıf varsa LabelEncoder yerine:
# - One-Hot Encoding (get_dummies)
# - Target Encoding
# - Frequency Encoding
# gibi yöntemler tercih edilmelidir.

