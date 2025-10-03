###########################################
# PYTHON İLE VERİ ANALİZİ (DATA ANALYSIS WITH PYTHON)
###########################################
# - NumPy
# - Pandas
# - Veri Görselleştirme: Matplotlib & Seaborn
# - Gelişmiş Fonksiyonel Keşifçi Veri Analizi (Advanced Functional Exploratory Data Analysis)

###########################################
# NUMPY (NUMERICAL PYTHON)
###########################################


# Neden NumPy ? (Why Numpy?)
# NumPy Array'i Oluşturmak (Creating Numpy Arrays)
# NumPy Array Özellikleri (Attibutes of Numpy Arrays)
# Yeniden Şekillendirme (Reshaping)
# Index Seçimi (Index Selection)
# Slicing
# Fancy Index
# Numpy'da Koşullu İşlemler (Conditions on Numpy)
# Matematiksel İşlemler (Mathematical Operations)

# NUMPY bilimsel hesaplamalar için kullanılır. Çok boyutlu array ler ve matrisler üzerinde yüksek performansta çalışma imkanı sağlar.
# Temelleri 1995 yılında atılmıştır. Matrix Sig adında python'ın kurucusu olan Guido van Rossum'un da içinde olduğu bir gruptur.
# Bu grubun amacı python programlama dilini kullanarak matematiksel ya da istatistiksel bazı ihtiyaçları da karşılayabilecek bir kütüphane,
# modül oluşturmaktır.

# Numpy içerisinde bir veri tutarken, bunu fix type adını verdiği sabitlenmiş tipte tutarak listelere kıyasla çok daha hızlı bir şekilde
# işlem yapma imkanı sağlar. Örneğin; bir listenin içerisinde 5 eleman varsa hepsi aynı tipte ise listeler, her bir elemanın tip ve boyut
# bilgilerini ve diğer bazı meta bilgileri ayrı ayrı tutarken, numpy der ki; "Ben sabit bir tipte veri tutarım." örneğin; integer olur.
# Bunun içerisinde sadece integer veri yapısı olduğundan dolayı değil, 5 tane 5.000 tane bile eleman olsa her biri için ayrı ayrı bir tip ya da
# boyut bilgisi ya da diğer meta bilgileri tutmayacak olduğundan dolayı verimli veri saklama imkanı sağlayarak hızlı bir şekilde array'ler üzerinde
# çalışma imkanı sağlar.

# Bir diğer özelliği, döngü yazmaya gerek olmadan array seviyesinde çok basit işlemlerle normalde daha çok çaba gerektiren işlemleri gerçekleştirme imkanı sağlamasıdır.

# "Neden numpy?" sorusunu kod ortamında değerlendirelim.


###########################################
# Neden NumPy ? (Why Numpy?)
###########################################

import numpy as np

a = [1, 2, 3, 4]
b = [2, 3, 4, 5]

ab = []  # 2 listenın elemanlarını çarparak yeni listeye klasik python yolu ile ekliyelim.

for i in range(len(a)):
    ab.append(a[i] * b[i])

a = np.array([1, 2, 3, 4])
b = np.array([2, 3, 4, 5])
a * b
# Listeleri numpy array'ine çeviriyoruz. Sonrasında ise bir döngü yazmadan, append() metodunu kullanmadan, boş bir liste oluşturup
# listenin içerisine eleman ekleme işlemi yapmadık. Çok kolay bir şekilde iki listenin elemanları çarpılmış oldu.
# Yüksek seviyeden ya da vektörel işlemler olarak ifade edilen durum budur.

# Kısaca; Numpy python da nümerik işlemler, hesaplamalı işlemler için geliştirilmiş bir kütüphanedir.
# Şuana kadar tanımı aldık evet fakat karşımıza "Neden Numpy ?" sorusu gelirse cevap şudur;
# İki sebeple tercih edilebilir;
#   - 1. Hız. "Bunu Nasıl Sağlar ?", "Neden numpy listelere göre daha hızlı?" Çünkü sabit tipte veri tutar, yani verimli veri saklar bundan dolayı hızlıdır.
#   - 2. Yüksek seviyeden işlemler yapma imkanı sağlamasıdır. "Bu ne demek ?", Normalde iki tane listeyi çarpmak istediğimizde böyle bir işlem seti yapmamız
#        gerekirken numpy kullanarak sadece a ve b yi çarptığımızda basit bir operatör ile bu işlem kolay bir şekilde gerçekleşmiş olacaktır.


###########################################
# NumPy Array'î Oluşturmak (Creating Numpy Arrays)
###########################################

# Numpy arary'leri diğer ifadesiyle ndarray nesneleri python'da kullanılan diğer bazı veri yapıları gibi bir veri yapısıdır.
# "Python'da hangi veri yapıları vardı?" Bir hatırlamak gerekirse; integer, float bunlar sayılardı. String diğer ifadesiyle
# karakter dizileri, listeler, setler, sözlükler ve tuple'lardı. Bunlar python'ın veri yapılarıdır. Tıpkı bu veri yapıları gibi numpy'ın da
# kendine ait bir veri yapısı vardır. Buna ndarray adı verilir ya da diğer ifadesiyle numpy array adı verilir.

# Numpy işlemlerini yapabilmek için öncelikle bir numpy array'ine ihtiyaç vardır. Pratikte genelde bu numpy array'leri sıfırdan oluşturulmaz.
# Çalışmanız sırasında yaygınca bazı veri yapılarından dönüştürülerek elde edilir. Fakat Sıfırdan nasıl oluşturulabileceğini de bir iki
# örnekle değerlendirmiş olacağız.

import numpy as np

np.array([1, 2, 3, 4, 5])

np.zeros(10, dtype=int) # 0'lardan oluşan, tipi integer olan array oluşturdu.

np.random.randint(0, 10, size=10) # 0 ile 10 arasında rastgele 10 tane integer seç.

np.random.normal()
# Bu fonksiyon diğer ifadesiyle metot der ki; "Benim 3 tane argümanım var." "Birincisine, oluşturmak istediğin normal dağılımlı
# kitlenin ortalamasını, ikincisine argümanı, üçüncüsüne de boyut bilgisini gir." girelim.

np.random.normal(10, 4, (3, 4))
# Ortalaması 10, standart sapması 4 olan örneğin; 3'e 4'lük bir array oluştur  diyelim ve gönderelim.

# array([[ 8.10842476,  7.47304466, 15.80491733,  7.21283921],
#        [ 7.60318195,  9.61420842,  6.18147271, 10.16740546],
#        [ 4.4203676 ,  6.27433964, 15.22928522,  7.7572609 ]])

# 3 satır, 4 sütundan oluşan, ortalaması 10 ve standart sapması 4 olacak şekilde normal dağılımlı sayılar oluşturulmuş oldu.
# Böylece sıfırdan array oluşturma işlemlerine birazcık değendik.


###############################
# NumPy Array Özellikleri (Attibutes of Numpy Arrays)
###############################
import numpy as np

# ndim: boyut sayısı
# shape: boyut bilgisi
# size: toplam eleman sayısı
# dtype: array veri tipi

a = np.random.randint(10, size=5)
a.ndim # Boyutu 1. "Neden?" Tek boyutlu bir array, bir matris değil, 2 boyutlu değil. Sadece boyut sayısına erişmek istersek.
a.shape # Tek boyutlu içerisinde 5 tane eleman var der. Eğer burada 2 boyut olsaydı her boyuttaki eleman sayısı gelecekti. Satır ve sütun sayısı için.
a.size # Toplam eleman sayısını verir.
a.dtype # İçerisinde tuttuğu tip bilgisini verir.

###############################
# Yeniden Şekillendirme (Reshaping)
###############################

# Elimizdeki bir numpy array in boyutunu değiştirmek istediğimizde reshape() metodunu kullanırız. "Bu ne anlama geliyor?" örnek üzerinde görelim.

import numpy as np
np.random.randint(1, 10, size=9) # tek boyutlu bir array'dir.
# Diyelim ki bunu 2 boyutlu bir array'e çevirmek istiyoruz. Örnek; 3'e 3'lük bir matrise çevirmek istiyoruz.

np.random.randint(1, 10, size=9).reshape(3, 3) # Bu şekilde 3'e 3'lük matris oluşturduk. Fakat şuan bunu saklayamıyoruz.

ar = np.random.randint(1, 10, size=9)
ar.reshape(3, 3) # Aynı işlem atama yapılarak da olur.

# Dikkatli edilmesi gereken kısım eğer 10 elemanlı olsaydı bu array, bunu 3'e 3'lük bir matrise çeviremezdik. 3'e 3'lük bir matriste 9
# eleman olması gerekiyor.















