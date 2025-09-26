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



