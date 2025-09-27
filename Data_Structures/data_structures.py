#############################################
# VERİ YAPILARI (DATA STRUCTURES)
#############################################
# - Veri Yapılarına Giriş ve Hızlı Özet
# - Sayılar (Numbers): int, float, complex
# - Karakter Dizileri (Strings): str
# - Boolean (TRUE - FALSE): bool
# - Liste (List)
# - Sözlük (Dictionary)
# - Demet (Tuple)
# - Set


#############################################
# Veri Yapılarına Giriş ve Hızlı Özet
#############################################

# Sayılar: integer
x = 46
type(x)

# Sayılar: float
x = 10.3
type(x)

# Sayılar: complex
x = 2j + 1
type(x)

# String
x = "Hello ai era"
type(x)

# Boolean
True
False
type(True)
5 == 4
3 == 2
1 == 1
type(3 == 2)

# Liste
x = ["btc", "eth", "xrp"]
type(x)

# Sözlük (dictionary) KEY ve VALUE olarak çalışır.
x = {"name": "Peter", "Age": 36}
type(x)

# Tuple ( demet )
x = ("python", "ml", "ds")
type(x)

# Set
x = {"name", "ml", "ds"}
type(x)

# NOT : Liste, tuple, set ve dictionary veri yapıları aynı zamanda Python Collections (Arrays) olarak geçmektedir.


#############################################
# Sayılar (Numbers): int, float, complex
#############################################

a = 5
b = 10.5

a * 3
a / 7
a * b / 10
a ** 2

# İşlemlerin aralarına boşluk bırakma sebebimiz PEP8 kanunları. Python geliştiriclerinin uygun gördüğü formatta kod
# yazma tarzıdır.

########################
# Tipleri Değiştirmek
########################

int(b) # Float bir değişkeni veya sayıyı int e çeviriyoruz.
float(a) # Int bir değişkeni veya sayıyı float a çeviriyoruz. Tam sayının sonuna .0 ekliyor.

int(a * b / 10)

c = a * b / 10
int(c)

#############################
# Karakter Dizisi (String)
#############################

print("John")
print('John') # Tek tırnak veya çift tırnak yazmanın farkı yoktur.

"John" # Print olmadan da yazdırabiliriz.

# NOT : Bir program ya da bir fonksiyon yazıyorsanız ve bu program ya da fonksiyon içerisinde ki bir noktada ekrana bir
# bilgi paylaşmak istiyorsanız ancak bu durumda print() kullanmanız gerekir.

name = "John"
name = 'John'

#################################
# Çok Satırlı Karakter Dizileri
#################################

long_str = """Veri Yapıları: Hızlı Özet,
Sayılar (Numbers): int, float, complex
Karakter Dizileri (Strings): str,
List, Dictionary, Tuple, Set,
Boolean (TRUE-FALSE): bool"""
# 3 tırnak içerisine yazdıklarımızın hepsini bir karakter dizisi olarak görecektir.


#############################################
# Karakter Dizilerinin Elemanlarına Erişmek
#############################################

name[0] # Köşeli parantezin içerisine index numarası yazılır. Index numarası 0'dan başlar.
name[3]
name[2]


#################################################
# Karakter Dizilerinde Slice (Dilimleme) İşlemi
#################################################

name[0:2] # 0. Index'den başla ve 2. Index e kadar git fakat 2 hariç.

long_str[0:10] # 0'dan 10'a kadar gitmiş olacak 10 hariç.


##########################################
# String İçerisinde Karakter Sorgulamak
##########################################

"veri" in long_str # veri yi long_str değişkeni içerisinde arar ancak CASE sensitive olduğundan dolayı false dönecek. Çünkü biz büyük V kullanmıştık.

"Veri" in long_str # True olarak döner.

"bool" in long_str # True olarak döner.

# Console tarafında \n ne demek ? O satırın aşağıda başladığını ifade eder.

##########################################
# String (Karakter Dizisi) Metodları
##########################################

# Metod Nedir ? = Çeşitli görevleri yerine getiren fonksiyon benzeri yapılardır.
# Diğer ifadesiyle class yani sınıflar içerisinde tanımlanan fonksiyonlardır.

# Öncelikle bir veri yapısının metotlarına nasıl erişebileceğimizi değerlendirelim.

dir(int) # int ile kullanılabilecek metotları getirir.
dir(str) # string ile kullanılabilecek metotları getirir.


##########
# len
##########

# len metodu gömülü olarak gelir.

name = "John"
type(name)
type(len) # bu bir fonksiyon veya metot dur döner.

len(name) # string ifadenin kaç elemanlı olduğunu döner.
len("vahitkeskin")
len("AlicanKaya")

# Kullanmış olduğumuz bir yapının metot mu yoksa fonksiyon mu olduğunu nasıl ayırt ederiz ?
# Eğer bir fonksiyon class yapısı içerisinde tanımlandıysa buna metot denir.
# Eğer bir class yapısı içerisinde değilse bu fonksiyondur.
# Metot ve fonksiyon aynı şeydir. Görevleri ve fonksiyonları itibariyle aynı şeylerdir. Çeşitli görevleri yerine getirirler.
# Fakat farklılaştığı nokta, fonksiyonların bağımsız, metotların class lar içerisinde tanımlanmış olmasıdır.


#######################
# upper() & lower(): küçük-büyük dönüşümleri
#######################

"miuul".upper() # girilen ifadeyi büyük harflere çevirir
"MIUUL".lower() # girilen ifadeyi küçük harflere çevirir


#######################
# replace: karakter değiştirir
#######################

hi = "Hello AI Era"
hi.replace("l", "p") # ilk ifade değiştirmek istediğimiz, 2. ifade gelmesini istediğimiz


#######################
# split: böler
#######################

"Hello AI Era".split() # Eğer bir argüman vermez isek default olarak boşluklara görer ayırır. Eğer verirsek o argümana göre.


#######################
# strip: kırpar
#######################

" ofofo ".strip() # Default olarak kırpma işlemini boşluğa göre yapar.
"ofofo".strip("o") # Verdiğimiz argümana göre yapar.


#######################
# capitalize: ilk harfi büyütür
#######################

"foo".capitalize()

dir("foo") # Kullanabileceğimiz metotları gösterir.

"foo".startswith("f") # Herhangi bir argüman almadan çalışmaz. Girilen argüman ile başlıyor ise True veya False döner.


#############################################
# List (Liste)
#############################################

# Python programlama dilinde en yaygın kullanılan veri yapılarından birisidir. Değiştirilebilir, sıralı ve kapsayıcı ,
# bir veri yapısıdır.

# - Değiştirelebilir.
# - Sıralıdır. Index işlemleri yapılabilir.
# - Kapsayıcıdır.

notes = [1, 2, 3, 4]
type(notes)

names = ["a", "b", "c", "d"]

not_nam = [1, 2, 3, "a", "b", True, [1,2, 3]] # Listeler kaspayıcı olduğu için içinde birden fazla veri yapısı tutabilir.

not_nam[0] # Değişken içerisinde ki ilk elemana erişmiş oluruz.
not_nam[5]
not_nam[6] # Liste içerisinde ki listeye erişir ama hepsini döner

not_nam[6][1] # Liste içerisinde ki listenin istediğimizi elemanına erişmek için kullanılan yöntem.

type(not_nam[6])
type(not_nam[6][1])

notes[0] = 99 # Listeler değiştirilebilir olduğundan index e denk gelen değiştirilebilir.

notes[0:4] # 0'dan 4 e kadar gider ancak 4.index de bulunan değeri dönmez.


#############################################
# List Methods (Liste Metotları)
#############################################

dir(notes)

# En yaygın kullanılan metot append metodudur. Diğerleri daha az kullanılır.

len(notes)
len(not_nam) # Liste içerisinde ki listeyi 1 eleman olarak alır.

#############################################
# append: Listelere eleman ekler.
#############################################

notes.append(100) # 100 değerini listeye ekler.


#############################################
# pop: indexe göre eleman siler.
#############################################

notes.pop(0) # 0.index e denk gelen değeri listeden siler.


#############################################
# insert: indexe ekler.
#############################################

notes.insert(1, 100) # Önce değer eklenmesini istediğimiz index girilir daha sonra girmek istediğimiz değer girilir.


#############################################
# Dictionary (Sözlük)
#############################################

# Değiştirilebilir.
# Sırasız. (3.7 versiyonundan sonra sıralı oldu.)
# Kapsayıcı

# key-value

dictionary = {"REG": "Regression",
              "Log": "Logistic Regression",
              "CART": "Classification and Reg"}

# Alt alta aralarına virgül koyarak key-value şeklinde girebiliriz.
# Okunabilirlik açısından virgülden sonra alta geçilmesi önerilir.

dictionary["REG"] # Burada REG i çağırdığımızda valuesu gelir.

dictionary = {"REG": ["RMSE", 10],
              "Log": ["MSE", 20],
              "CART": ["SSE", 30]}

# Value tarafına liste de girilebilir. Buraya daha uzun bir liste de girebiliriz, daha farklı veri yapıları da girebiliriz.
# Dolayısıyla key ilk değişken oluşturmada gördüğü görev gibi sözlük içerisinde de önemli bir görev görür.

dictionary["CART"] # Value de ki tüm listeyi döner.

dictionary["CART"][1] # Bu şekilde value tarafında döndüğü listeden istediğimiz elemana ulaşabiliriz.

dictionary = {"REG": 10,
              "Log": 20,
              "CART": 30} # Value tarafımız int değerlerden de oluşabilir.

dictionary["REG"]


#####################
# Key Sorgulama
#####################

"REG" in dictionary # TRUE DÖNER.
"VAR" in dictionary # FALSE DÖNER.


#####################
# Key'e Göre Value'ya Erişmek
#####################

dictionary["REG"]
dictionary.get("REG") # Bu şekilde de girilen parametrenin elemanlarına erişebiliriz.


#####################
# Value Değiştirmek
#####################

dictionary["REG"] = ["YSA", 10] # REG key'in value değerini değiştirdik.


#########################
# Tüm Key'lere Erişmek
#########################

dictionary.key() # Bütün keylere erişebiliriz.
dictionary.values() # Bütün value lara erişebiliriz.


#############################################
# Tüm Çiftleri Tuple Halinde Listeye Çevirme
#############################################

dictionary.items() # Her anahtar-değer çiftini (key, value) tuple olarak içeren bir dict_items görünümü döndürür


###################################
# Key-Value Değerini Güncellemek
###################################

dictionary.update({"REG": 11}) # Key ve yeni değerini girerek güncelleyebiliriz.


############################
# Yeni Key-Value Eklemek
############################

dictionary.update({"RF": 10}) # Bunu kullandığımızda sözlük yeni key değerinin olmadığını gördüğünde onu oluşturur.


############################
# Tuple (Demet)
############################

# Tuple lar listelerin değişime kapalı halidir.

# - Değiştirilemez.
# - Sıralıdır.
# - Kapsayıcıdır.

t = ("john", "mark", 1, 2)
type(t)

t[0]
t[0:3] # İlgili index e göre bir Slice işlemi yapacak.

# t[0] = 99 / Eleman değiştirme işlemi Tuple'da yapılamayacağı için hata verir.

t = list(t)
t[0] = 99
t = tuple(t) # Önce list'e dönüştürüp istediğimiz elemanı değiştirip daha sonra tuple yaparak değişiklik yapabiliriz.

# Tuple, listelere benzer ama daha güvenli bir şekilde çalışma imkanı sağlar. Dolayısıyla Üzerinde çalıştığımız
# bazı senaryolarda belirli bir çıktının tuple formatında olmasını ve değiştirilemiyor olmasını gözlemlemek istiyebiliriz.
# Tuple'ların kullanım sıklığı çok çok azdır.


############################
# Set
############################

# - Değiştirilebilir.
# - Sırasız + Eşsizdir.
# - Kapsayıcıdır.

############################
# difference(): İki kümenin farkı
############################

set1 = set([1, 3, 5]) # Liste üzerinden set oluşuyor. Önce girilene göre döner.
set2 = set([1, 2, 3])

# type() gereksiz kod alanı kaplamaması için python console tarafında da kullanılabilir.

# set1'de olup set2'de olmayanlar
set1.difference(set2) # 5 dönecek.
set1 - set2 # Buradaki kesişimi ifade etmenin bir diğer yolu matematiksel öperatördür. - işareti ile set1'de olup set2'de olmayan gelir.

# set2'de olup set1'de olmayanlar
set2.difference(set1) # 2 dönecek.
set2 - set1 # Buradaki kesişimi ifade etmenin bir diğer yolu matematiksel öperatördür. - işareti ile set2'de olup set1'de olmayan gelir.

############################
# symmetric_difference(): İki kümede de birbirlerine göre olmayanlar
############################

set1.symmetric_difference(set2) # 2 ile 5 döner.


############################
# intersection(): İki kümenin kesişimi
############################

set1 = set([1, 3, 5])
set2 = set([1, 2, 3])

set1.intersection(set2)

set1 & set2 # Buradaki kesişimi ifade etmenin bir diğer yolu matematiksel öperatördür. Ve işareti ile 2 kümenin kesişimi gelir.


############################
# union(): İki kümenin birleşimi
############################

set1.union(set2) # 2 kümenin birleşimi gerçekleşir.
set2.union(set1) # Bir önceki ile aynı işlevi görür.


############################
# isdisjoint(): İki kümenin kesişimi boş mu?
############################

# Kullanılan ifadenin başında is var ise bu genelde true veya false şeklinde dönüş yapar.

set1 = set([7, 8, 9])
set2 = set([5, 6, 7, 8, 9, 10])

set1.issubset(set2) # False döner
set2.issubset(set1)


############################
# issubset(): Alt küme olup olmadığını sorgular
############################

set1.issubset(set2) # True döner.
set2.issubset(set1) # False döner.


############################
# issuperset(): 1 küme diğer kümeyi kapsıyor mu
############################

set2.issuperset(set1) # True döner
set1.issuperset(set2) # False döner.
