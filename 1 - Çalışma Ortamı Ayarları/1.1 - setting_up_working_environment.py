###############################################
# Sayılar (Numbers) ve Karakter Dizileri (Strings)
###############################################

# Bir ifadenin karakter dizisi olduğunu söylemnin yolu etrafına 2 tırnak ya da tek tırnak atmaktayız.

print("Hello World")
print("Hello AI Era")


print(9) # int
print(9.2) # float (ondalık)

# type() fonksiyonu der ki; Eğer bana bir nesne sorarsanız ben size onun tip bilgisini veririm.
# Nesne ne demek; Python üzerinde çalıştığımız her şey bir nesne(obje)'dir. Bu objelerin tipleri vardır.

type(9) # int dönecektir.
type(9.2) # float dönecektir.
type("Python") # string dönecektir.

###############################################
# Atamalar ve Değişkenler (Assignments & Variables)
###############################################

# 9

# "Hello ai era"

# Bunları daha sonra programın herhangi bir yerinde kullanmak istersek eğer, tekrardan 9 veya string yazmak yerine
# bu ifadeleri değişkenler ile tutabiliyoruz.

a = 9
# Eşittir işareti atama yapmak demektir. A ifadesi bir değişkendir. Sağ tarafta ise tutmak istediğimiz nesne vardır.

b = "hello ai era" # Benzer şekilde burada string bir ifade atıyarak daha sonra istediğimiz zaman bunu kullanabiliriz.

# Atadığımız nesneleri çağırmak istediğimizde atadığımız değişken adını yazmamız yeterlidir.
# NOT : Atama yapmayı kalıcılaştırmak için önce atadığımız nesneyi bir defa çalıştırmamız gerekiyor.

a
b

c = 10

a * c # Yıldız çarpımı temsil ediyor.

a * 10

# Bu değerleri bir yere atamadığımız için sadece ekrana yazdırır.
# Eğer sonuçları kaydetmek ve daha sonra kullanmak istersek eğer bunu da atayarak kaydetmemiz gerekir.

d = a - c

# Şimdi ki kısımdan önce What is a virtual environment daha sonra Package Managment a bakılmalı

###############################################
# Virtual Environment (Sanal Ortam ) ve Paket Yönetimi (Package Managment)
###############################################

# Bilgisayarımızda ki sanal ortamların listelenmesi :
# conda env list

# Sanal ortam oluşturma :
# conda create -n myenv ( myenv yerine verilmek istenen isim yazılmalı )

# Sanal ortam aktif etme :
# conda activate myenv ( myenv yerine aktif edeceğiniz sanal ortam adı yazılmalı )

# Sanal ortam deaktif etme :
# conda deactivate

# Yüklü paketlerin listelenmesi :
# conda lıst

# Paket yükleme :
# conda install numpy

# Aynı anda birden fazla paket yükleme :
# conda install numpy scipy pandas ( aralarına boşluk bırakarak aynı anda birden fazla paket yükleyebiliriz )

# Paket silme :
# conda remove package_name

# Belirli bir versiyona göre paket yükleme :
# conda install numpy=1.20.1 ( pip'te 2 eşittir kullanılarak yükleme yapılır )

# Paket yükseltme için :
# conda upgrade numpy

# Tüm paketlerin yükseltilmesi :
# conda upgrade -all

# pip: pypi (python package index) paket yönetim aracı

# Paket yükleme :
# pip install paket_adı

# Paket yükleme versiyona göre :
# pip install pandas==1.2.1 ( bunu yaparken eğer yüklü ise pandas otomatik onu kaldırır daha sonra yükleme yapar )

# Bizde yüklü olan paketleri başka birine aktarmak istersek eğer bizim requirements.txt veya yaml dosyası oluşturmamız gerekir.
# yaml dosyasını oluşturmayı conda ile gerçekleştirebiliriz.
# conda env export > environment.yaml ( bunu pip ile de yapabiliriz ancak pip Dünya'sında requirements.txt, conda'da environment yaml olarak adlandırılır. )

# Oluşan dosya python projemizin altında oluşur. Dosyalarımızı listelemek için
# ls komutu, Windows da dir komutudur.

# Var olan bir ortamı tekrardan oluşturma veya birine verdiğimiz environment.yaml ı kurma
# conda env create -f environment.yaml