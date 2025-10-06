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
# loc - mutlak olarak index'lerdeki label'lara göre seçim yapar.





















