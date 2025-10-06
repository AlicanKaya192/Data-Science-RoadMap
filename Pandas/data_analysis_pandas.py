##############
# PANDAS
##############

# "Nedir Pandas ?" Veri manipülasyonu ya da veri analizi denildiğinde akla gelen ilk python kütüphanelerinden biridir.
# Öncelikle ekonometrik ve finansal çalışmalar için doğmuş, daha sonra veri analitiği denildiğinde en sık kullanılan kütüphahen
# haline gelmiştir. Temeli 2008 yılında atılmıştır. Birçok farklı veri yapısıyla çalışma imkanı sağlamaktadır.
# Birçok farklı kaynaktan veriyi okuma imkanına sahiptir. Veri analitiği genel başlığı altında, makine öğrenmesinden
# veri bilimine, veri analizinden derin öğrenmeye, veri varsa ve python varsa bir şekilde mutlaka elimizin dokunacağı
# dokunması gereken en önemli kütüphanelerden birisidir.


# Pandas Series
# Veri Okuma (Reading Data)
# Veriye Hızlı Bakış (Quick Look At Data)
# Pandas'ta Seçim İşlemleri (Selection in Pandas)
# Toplulaştırma ve Gruplama (Aggregation & Grouping)
# Apply ve Lambda
# Birleştirme (Join) İşlemleri


##################
# PANDAS SERIES
##################

import pandas as pd
from pandas import set_option

s = pd.Series([10, 77, 12, 4, 5])

# Pandas serisi oluşturma. Bu bir metottur ve der ki; "Bana bir liste ya da farklı tipte bir veri ver ki, ben bunu pandas serisine çevireyim."
# Pandas serisinde index bilgisi iç özelliktir.

# DİKKAT!!!! - Veri yapıları bizim için şu açıdan önemlidir;
# Bir veri yapısının liste mi, tuple mı, set mi, pandas serisi mi, pandas dataframe mi yoksa numpy array'i mi olduğunu bilmek fonksiyonların
# bizden beklediği ihtiyaçları daha doğru bir şekilde yerine getirme imkanı sağlar ya da karşılaşabileceğimiz bazı problemleri daha iyi çözme
# imkanı sağlar. "Nasıl Yani ?" Örneğin;
# Kullanacak olduğumuz bir fonksiyon bizden dataframe beklerken biz ona bir pandas serisi gönderdiğimizde hata verecektir ve alana yeni
# başlayanlar bu hataları okumak ve hataları internetten araştırmak konusunda biraz zorluk çekmektedir. Dolayısıyla eğer veri yapılarını iyi bilirsek
# çıktılarda hata aldığımız durumlarda ki çok yaygınca fonksiyonların beklentileri karşılamadığında hata alınır.

s.index # Index bilgisine ulaşabiliriz.
s.dtype # İçerisindeki verinin tip bilgisini verir.
s.size # İçerisindeki eleman sayısını verir.
s.ndim # Boyut bilgisini verir.

# Pandas serileri tek boyutludur ve index bilgileri vardır.
# "Bu serinin içerisindeki değerlerin kendilerine erişmek istersek ne yapabiliriz ?"

s.values # İçerisindeki değerlere ulaşırız.

# DİKKAT!!!! - Bir pandas serisinin sonuna value ifadesini girdiğimizde ve değerlerine erişmek istediğimizde bu durumda
# zaten indeks bilgisiyle ilgilenmiyorum demiş olduğumuzdan dolayı bize bunu bir numpy array'i olarak döndürür.

s.head() # İlk 5 gözlemi verir. s.head(3) Bu şekilde kullanarak ilk 3 ü getirebiliriz.
s.tail() # Son 5 gözlemi verir. s.tail(3) Bu şekilde kullanarak son 3 ü getirebiliriz.


#################################
# VERİ OKUMA (Reading Data)
#################################
import pandas as pd

# Pandas birçok farklı tipteki veriyi kolay bir şekilde okuma imkanı sağlar. İçerisinde bulunan metotlar aracılığıyla
# csv, txt, excel ya da diğer bazı özel dosya formatlarını okuyabilir ve üzerinde çalışabiliriz.

# Öncelikle en yaygın kullanıma sahip olan csv dosyalarını okumayı değerlendirelim.

# Diyelim ki; bilgisayarımızda, çalışma projemizde ya da herhangi bir yerde bir csv dosyası var.

df = pd.read_csv("dosya_yolu") # Veri okuma işlemi
df.head()

# Farklı veri yapıları olursa ne yapacağım. pd ifadesine ctrl ye basılı tutarak tıklıyoruz
# ve __init__.py dosyasına girerek ctrl + f diyerek read yazıyoruz. Kullanabileceğimiz metotlar karşımıza geliyor.


#########################################
# VERİ HIZLI BAKIŞ (Quick Look At Data)
#########################################
import pandas as pd
import seaborn as sns

df = sns.load_dataset("titanic")
df.head()
df.tail()
df.shape # Dataframe'in boyut bilgisini alırız.
df.info() # Genel bilgi verir. Değişken tiplerini öğrenmek istersek vb.
df.columns # Dataframe değişken isimlerini verir.
df.index
df.describe().T
# Elimizdeki bir dataframe'in eğer hızlı bir şekilde özet istatistiklerine erişmek istersek describe() kullanıyoruz.
# Sonuna bunun transpose'unu al diyerek daha okunabilir bir formatta gelmesini sağlayabiliriz.

# "Detaylarına girmeden sadece veri setinde en az bir tane dahi olsa bir eksiklik var mı ?" sorusunu yöneltmek istediğimizi düşünelim.
# Bunun için isnull() metodu, sonrasında values, bu values'lardan herhangi birisinde isnull durumu var mı demek için de any() ifadesi kullanılır.
df.isnull().values.any()

# Değişkenlerde ki eksiklik durumu incelenmek istenirse ne yapılır.
df.isnull().sum()

# Diyelim ki, kategorik değişkenlerden birisinin ki, cinsiyet kadın - erkek şeklinde bir kategorik değişkendir.
# Bunun içerisinde kaç tane sınıf olduğu ve bu sınıfların kaçar tane olduğu bilgisine erişmek istediğimizi düşünelim.

df["sex"].value_counts() # Erkek ve kadın bilgisi herbirinden kaçar tane olduğunu verir.


#########################################
# PANDAS'TA SEÇİM İŞLEMLERİ (Selection in Pandas)
#########################################
import pandas as pd
import seaborn as sns

df = sns.load_dataset("titanic")
df.head()

df.index
df[0:13] # Slice işlemi 13 hariç
df.drop(0, axis=0).head() # 0.index'deki satırı sil. Head ile gözlemle. eğer axis=1 olsaydı o zaman sütun silinirdi.

delete_indexes = [1, 3, 5, 7]
# Eğer birden fazla silme işlemi yapmak istersek, silmek istediğimizi index değerlerini bir liste yaparak, drop içerisine
# yazdığımız 0 yerine listeyi çağırarak silebiliriz.

df.drop(delete_indexes, axis=0).head(10)
# Bu silme işlemleri kalıcı değildir. Eğer kalıcı yapmak istersek;
# 1 - Bu dataframe'i tekrar atarız yani df = df.drop(delete_indexes, axis=0).head(10)
# 2 - inplace argümanını kullanarak bu değişikliği kalıcı hale getirebiliriz. Atama işlemi yapmadan axis'den sonra
# inplace=True argümanını kullanarak bu işlemi kalıcı olarka yap bilgisini veririz.
df.drop(delete_indexes, axis=0, inplace=True).head(10)

# DİKKAT!!! - inplace argümanı çok yaygın bir şekilde kullanılan birçok metotta, metot uygulandığında bir değişiklik yapıldığında
# bu değişikliğin kalıcı olması gerektiği bilgisini veren bir argümandır, parametredir. Bu sebeple genelde bir işlem yaptığınızda
# ve bunun kalıcı olmasını istediğinizde aklınıza inplace gelebilir, bir çok metot ile kullanılmaktadır.


############################
# Değişkeni Index'e Çevirmek
############################

# Örneğin; yaş değişkenini index'e, index'deki değeri de bir değişken olarak değişkenlerin arasına almak istediğimizi düşünelim.

df["age"].head()
df.age.head()

df.index = df["age"]
df.drop("age", axis=1).head()
df.drop("age", axis=1, inplace=True).head()


############################
# Index'i Değişkene Çevirmek
############################

df.index

# 1.Yol

df["age"] = df.index
# Eğer girdiğimiz string ifade,dataframe in içerisinde varsa bu durumda bu değişken seçilirken eğer girmiş olduğumuz
# ifade dataframe'in içerisinde yoksa bu durumda yeni değişken eklendiği anlaşılır.
# DİKKAT!!! - Bir dataframe'e yeni değişken eklemek için o dataframe'in içerisinde olmayan bir isimlendirme girersek yeni değişken ekleriz.

# 2.Yol

df.reset_index().head()
# 1. index'te yer alan değeri silecektir. İkincisi, bunu sütun olarak yani yeni bir değişken olarak ekleyecektir.

df = df.reset_index()


############################
# Değişkenler Üzerinde İşlemler
############################
import pandas as pd
import seaborn as sns

pd.set_option('display.max_columns', None) # Gelecek max sütun limitini kaldırarak çıktıdaki 3 noktalardan kurtuluyoruz.
df = sns.load_dataset("titanic")
df.head()

"age" in df # Değişken veri seti içerisinde var mı onu sorgularız.

df["age"].head()
df.age.head()
type(df["age"].head()) # Series olarak döner.

df[["age"]].head()
type(df[["age"]].head()) # Dataframe olarak döner.
# Tek bir değişken seçerken seçimin sonucunun dataframe olarak kalmasını istiyorsak, bu durumda iki köşeli parantez kullanarak
# seçim yapmamız gerekmektedir.

df[["age", "alive"]]
# Birden fazla değişken seçme

col_names = ["age", "adult_male", "alive"]
df[col_names]

df["age2"] # Eğer dataframe içerisinde olmayan bir değer girersek bu değişken eklenecektir.
df["age2"] = df["age"] ** 2 # Burada yaşın karesi yeni bir değişken olarak veri setine eklenecektir.

df.drop("age2", axis=1).head()

# Birden fazla silme işlemi için
col_names = ["age", "adult_male", "alive"]
df.drop(col_names, axis=1).head()

# Diyelimki veri setinde belirli string ifadeleri silmek istiyorum ne yapabilirim ?

df.loc[:, df.columns.str.contains("age")].head()
# loc dataframe'lerde seçme işlemi için kullanılan özel yapıdır.
# .str string'lere uygulanan bir metot.
# Contains metodu string'lere uygulanan bir metot. contains() der ki;
# "Bana bir string ifade ver, ben benden önceki string de bu var mı yok mu diye bunu arayım."

# Bunları silmek istiyorum peki onu nasıl yapacağım ? Burada değildir demenin yolu ~ (tilda işareti)
df.loc[:, ~df.columns.str.contains("age")].head()
# Tilda ile bunun dışındakileri seç bilgisini göndeririz ve age ile ilgili bütün değişkenleri siler.


############################
# Loc & ILoc
############################
import pandas as pd
import seaborn as sns

pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

# loc & iloc yapısı dataframe'lerde seçim işlemleri için kullanılan özel yapılardır.
# Özetle iloc numpy'dan, listelerden alışık olduğumuz klasik, integer based yani index bilgisi vererek seçim yapma işlemlerini ifade eder.
# iloc - integer based selection
# loc - mutlak olarak index'lerdeki label'lara göre seçim yapar. label based selection.


# iloc: integer based selection
df.iloc[0:3] # satır
# 0'dan 3'e kadar

df.iloc[0, 0]
# 0. satır 0. sütundaki elemanı seç getir.


# loc: label based selection
df.loc[0:3] # satır
# Index'te label, yani etiket nasılsa onu mutlak olarak seçer. Yani burada 3 de gelecektir.

df.iloc[0:3, "age"] # Hata verir.
df.iloc[0:3, 0:3] # Doğru kullanımı. 3.satır 3.sutuna kadar.

df.loc[0:3, "age"] # Sadece yaş sütununu getirir 3 dahil

col_names = ["age", "embarked", "alive"]
df.loc[0:3, col_names] # 3 dahil birden fazla değişkeni verir.


############################
# Koşullu Seçim (Conditional Selection)
############################

import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()


df[df["age"] > 50].head() # Yaşı 50'den büyük olan kişiler geldi.
df[df["age"] > 50]["age"].count() # Yaşı 50'den büyük olan kişi sayısını döner.

df.loc[df["age"] > 50, "class"].head() # Yaşı 50'den büyük olanların sınıf bilgisi gelir.
df.loc[df["age"] > 50, ["age", "class"]].head()

# Peki 2 koşul nasıl gireceğiz

df.loc[(df["age"] > 50) & (df["sex"] == "male"), ["age", "class"]].head()

# DİKKAT!!! - Eğer birden fazla koşul giriyorsak bunları parantez içlerine almamız gerekir. Aksi takdirde hata verecektir.

df.loc[(df["age"] > 50)
       & (df["sex"] == "male")
       & (df["embark_town"] == "Cherbourg"),
       ["age", "class", "embark_town"]].head()  # Okunabirlik için bu şekilde yapılması daha iyi olur.

df_new = df.loc[(df["age"] > 50)
       & (df["sex"] == "male")
       & ((df["embark_town"] == "Cherbourg") | (df["embark_town"] == "Southampton")),
       ["age", "class", "embark_town"]].head()

# Veya operatörünü kullanak istersek eğer kullanmak istediğimiz kısmı parantezler içerisine almamız gerekir.

df_new["embark_town"].value_counts()


############################
# Toplulaştırma ve Gruplama (Aggretion & Grouping)
############################

# count()     # Değer sayısı (NaN hariç)
# first()     # İlk değer
# last()      # Son değer
# mean()      # Ortalama
# median()    # Medyan
# min()       # En düşük değer
# max()       # En yüksek değer
# std()       # Standart sapma
# var()       # Varyans
# sum()       # Toplam
# - pivot table

# Buradaki fonksiyonlar toplulaştırma fonksiyonlarıdır. Kullancak olduğumuz groupby() işlemiyle hepsi kullanılabilirdir. pivot table hariç.

import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

df["age"].mean() # Bu şekilde yaş ortalaması alırız. Fakat eğer cinsiyete göre istersek...

df.groupby("sex")["age"].mean() # Yaş değişkeninin ortalamasını aldık.

# Diyelim ki sadece ortalamasını değil bir de toplamını da almak istiyorum.

df.groupby("sex").agg({"age": "mean"}) # Bu kullanıma alışmak bir öncekinden çok daha önemlidir.

df.groupby("sex").agg({"age": ["mean", "sum"]})

df.groupby("sex").agg({"age": ["mean", "sum"],
                       "survived": "mean"})

df.groupby(["sex", "embark_town"]).agg({"age": "mean",
                                        "survived": "mean"})

df.groupby(["sex", "embark_town", "class"]).agg({"age": "mean",
                                                 "survived": "mean"})

df.groupby(["sex", "embark_town", "class"]).agg({"age": "mean",
                                                 "survived": "mean",
                                                 "sex": "count"})


############################
# Pivot Table
############################
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

# Pivot table groupby işlemlerine benzer şekilde veri setini kırılımlar açısından değerlendirmek ve ilgilendiğimiz özet
# istatistiği bu kırılımlar açısından görme imkanı sağlar.

df.pivot_table("survived", "sex", "embark_town")
# pivot table der ki;
# Benim 1. argüman values argümanı, bana kesişimlerde neyi görmek istiyorsan onu gir.
# 2. argüman index'te yani satırda hangi değişkeni görmek istiyorsun onu gir.
# 3. argüman peki sütunda hangi değişkeni görmek istiyorsun der.

# pivot_table'ın ön tanımlı değeri mean'dir yani ortalamadır. Bundan dolayı survived değişkeninin ortalaması gelecektir.

df.pivot_table("survived", "sex", "embark_town", aggfunc="std")

df.pivot_table("survived", "sex", ["embark_town", "class"])
# "survived" sütununu özetliyoruz (aggfunc default olarak mean)
# Satırda "sex" sütununa göre groupluyor
# Sütunlarda ["embark_town", "class"] ikilisi ile çok seviyeli sütun oluşturuyor
# Yani her embark_town ve class kombinasyonu için ortalama survived değerini gösteriyor

df["new_age"] = pd.cut(df["age"], [0, 10, 18, 25, 40, 90])

# cut ve qcut fonksiyonları elimizdeki sayısal değişkenleri kategorik değişkene çevirmek için en yaygın kullanılan iyi ayrı fonksiyondur.
# Genelde sayısal değişkeni hangi kategorilere bölmek istediğinizi biliyorsanız ve bunu belirtmek isterseniz bu durumda cut() fonksiyonu kullanılır.
# Eğer elinizdeki sayısal değişkeni tanımıyorsanız dolayısıyla çeyreklik değerlerine göre bölünsün istiyorsanız bu durumda qcut() kullanılır.
# "Bu ne demek ?" Örneğin; yaş değişkenini kategorik değişkene çevirmek istediğinizde age değişkeni adındaki değişkeni tanıyorum.
# Yani örneğin, diyebilirim ki; "0 ile 10 arasına çocuk de, 10 ile 18 arasına genç de, 18 ile 28 arasına da genç de. Ya da işte daha
# büyüklerine olgun de." diyebiliyorum.

# Örneğin age değişkenini tanıyorum. Dolayısıyla bu sayısal değişkeni kategorik değişkene çevirirken, çevirecek olduğum sınıfları tanımlayabiliyorum.
# "Tanımlayamazsam ne olacak?" İşte tamamlayamazsam qcut fonksiyonunu kullanmam lazım ki, qcut() fonksiyonu otomatik olarak değerleri küçükten büyüğe
# doğru sıralar ve yüzdelik çeyrek değerlerine göre bunları gruplara böler, kategorilere böler. Çok yaygın kullanılır.

# Peki yaş kırılımlarında hayatta kalanları inceleyebilir miyiz?

df["new_age"] = pd.cut(df["age"], [0, 10, 18, 25, 40, 90])
df.pivot_table("survived", "sex", ["new_age", "class"])

pd.set_option('display.width', 500) # Gelen kodları daha fazla görebilmek için


############################
# Apply ve Lambda
############################
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()

# apply() satır ya da sütunlar'da otomatik olarak fonksiyon çalıştırma imkanı sağlar. Yani bir dataframe'e apply ile istediğimiz bir fonksiyonu uygulayabiliriz.
# satır ya da sütunlar'da uygulabiliriz.

# "Lambda ne işe yarar." lambda() bir fonksiyon tanımlama şeklidir. Tıpkı def gibi. Fakat farkı kullan-at fonksiyondur.
# Yani kod akışı esnasında bir kere kullanayım atayım daha sonra işim yok diyebileceğimiz noktalarda fonksiyon tanımlamak yerine,
# fonksiyon tanımlamadan kullan-at bir şekilde fonksiyon kullanma imkanı sağlar. Şimdi hemen ilerleyelim.

df["age2"] = df["age"] * 2
df["age3"] = df["age"] * 5

# Diyelim ki, bu veri seti içerisindeki age değişkenlerinin 10'a bölünmesini istiyoruz. "Normalde bunu nasıl gerçekleştiririz."
# Tek tek bu değişkenleri seçeriz.

# df["age"] / 10.head # Bu şekilde head kullanamayız.
(df["age"] / 10).head() # Bu şekilde head kullanabiliriz.
(df["age2"] / 10).head()
(df["age3"] / 10).head()

for col in df.columns:
    if "age" in col:
        print(col)

for col in df.columns:
    if "age" in col:
        print((df[col] / 10).head())

for col in df.columns:
    if "age" in col:
        df[col] = df[col] / 10

# apply ve lambda ile nasıl yaparız

df[["age", "age2", "age3"]].apply(lambda x: x / 10).head()

# az önceki gibi programatik hale getirelim

df.loc[:, df.columns.str.contains("age")].apply(lambda x: x / 10).head()

df.loc[:, df.columns.str.contains("age")].apply(lambda x: (x - x.mean()) / x.std()).head()

# DataFrame'deki sütun adları arasında "age" ifadesini içeren tüm sütunları seçer.
# Daha sonra her bir seçilen sütun (x) için standartlaştırma işlemi uygular:
# (x - x.mean()) / x.std() → yani her değerden sütunun ortalamasını çıkarır ve
# sonucu sütunun standart sapmasına böler.
# Bu işlem her "age" sütununu ortalaması 0, standart sapması 1 olacak şekilde ölçekler.
# Son olarak, .head() metodu ile bu dönüştürülmüş verinin ilk 5 satırını gösterir.

# Peki lambda tarafında, lambda yerine normal tanımlanmış fonksiyon kullanabilir miyiz?

def standart_scaler(col_name):
    return (col_name - col_name.mean()) / col_name.std()

df.loc[:, df.columns.str.contains("age")].apply(standart_scaler).head()

# Anlaşılması gereken temel nokta şudur; apply fonksiyonu bize satırlarda ya da sütunlarda elimizdeki belirli bir fonksiyonu,
# bu satır ya da sütunlara uygulama imkanı sağlar ve bu oldukça değerlidir.

# Peki bu yapılan işlemleri kaydetmek istiyorsak...

df.loc[:, ["age", "age2", "age3"]] = df.loc[:, df.columns.str.contains("age")].apply(standart_scaler).head()

df.loc[:, df.columns.str.contains("age")] = df.loc[:, df.columns.str.contains("age")].apply(standart_scaler).head()


############################
# Birleştirme İşlemleri (Join) İşlemleri
############################
import numpy as np
import pandas as pd

m = np.random.randint(1, 30, size=(5, 3))

df1 = pd.DataFrame(m, columns=["var1", "var2", "var3"])
# 0'dan dataframe oluşturmaya yarar.
# 1.argümanına veri yapısı veriyoruz. Bu liste olabilir, sözlük olabilir, numpy array'i olabilir. Bunu dataframe'e çevirir.
# 2 argümanada değişkenlerin isimlerini gireriz. Biz 5'e 3'lük oluşturduğumuz için 3 tane sütun olacak.

df2 = df1 + 99
# Burada ise df1'in değerlerinin hepsine 99 ekledik ve 2. dataframe'i oluşturduk.

# 2 şekilde birleştirme yapacağız. 1. hızlı, seri, pratik ve yaygınca kullanım alanı bulan concat() metodunu kullanacağız.
# Diğeri ise merge() yöntemi olacak.

# Eğer 2 dataframe'i alt alta birleştirmek istersek.
pd.concat([df1, df2])

# İki DataFrame'i (df1 ve df2) dikey olarak (varsayılan olarak satır bazında) birleştirir.
# Ancak DİKKAT!!! — bu birleştirme işlemi sırasında, her iki DataFrame'in de kendi index numaraları korunur.
# Yani index değerleri çakışabilir veya sıralı olmayabilir.

# Eğer yeni bir sıralı index isteniyorsa, "ignore_index=True" parametresi kullanılmalıdır:
pd.concat([df1, df2], ignore_index=True)
# Ön tanım değeri 0 olduğundan, pd.concat() fonksiyonu varsayılan olarak satır (axis=0) bazında birleştirme yapar.
# Yani df2, df1'in altına eklenir. Eğer sütun bazında birleştirme istenirse axis=1 olarak belirtilmelidir.


##################################
# Merge ile Birleştirme İşlemleri
##################################

# Bu da daha detaylı olarak çeşitli birleştirme işlemleri yapılmasına imkan tanır.

df1 = pd.DataFrame({"employees": ["john", "dennis", "mark", "maria"],
                    "group": ["accounting", "engineering", "engineering", "hr"]})

df2 = pd.DataFrame({"employees": ["mark", "john", "dennis", "maria"],
                    "start_date": [2010, 2009, 2014, 2019]})

# Elimde 2 tane dataframe var çalışan, group ve başlangıç tarihlerini bir arada istiyorum.

pd.merge(df1, df2)
# Ortak sütun adlarına göre iki DataFrame'i birleştirir (INNER JOIN mantığıyla).
# Yani her iki DataFrame’de de ortak olan sütun değerleri üzerinden eşleşme yapılır.
# Ortak sütun adı belirtilmezse, otomatik olarak aynı isimli sütunlar birleştirme için kullanılır.

pd.merge(df1, df2, on="employee")
# Birleştirme işleminin "employee" sütunu üzerinden yapılacağını açıkça belirtir.
# Yani df1 ve df2 içinde "employee" sütunundaki eşleşen değerlere göre satırlar birleştirilir.

# Amaç: Her çalışanın müdürünün bilgisine erişmek istiyoruz.
df3 = pd.merge(df1, df2)

df4 = pd.DataFrame({
    'group': ['accounting', 'engineering', 'hr'],
    'manager': ['Caner', 'Mustafa', 'Berkcan']})

# Bu dataframe'de müdür bilgisi var fakat çalışan bilgisi yok, grup bilgisi var.
# Ben eğer diğer tabloya bu tabloyu getirmek istiyorsam 2 tabloyu gruba göre birleştirmeliyim.

pd.merge(df3, df4)









