######################
# BURADA Kİ EGZERSİZLER İÇİN ÖNCELİKLE Çalışma_Ortamı_Ayarları içerikleri daha sonra Data_Structures içeriğini çalışınız.
######################

# GÖREV 1 : Verilen Değerlerin Veri Yapılarını İnceleyiniz.
x = 8
y= 3.2
z = 8j + 18
a = "Hello World"
b = True
c = 23 < 22
l = [1, 2, 3, 4]
d = {"Name": "Jake", "Age": 27, "Address": "Downtown"}
t = ("Machine Learning", "Data Science")
s = {"Python", "Machine Learning", "Data Science"}

print(type(x), type(y), type(z), type(a), type(b), type(c), type(l), type(d), type(t), type(s))

# x = int, y = float, z = complex, a = string, b = bool, c = bool, l = list, d = dictionary, t = tuple, s = set
# l = Sıraladır.
#     Kapsayıcıdır.
#     Değiştirilebilir.

# s = Sırasız + eşsiz
# #   Kapsayıcıdır.
# #   Değiştirilebilir.


# GÖREV 2 : Verilen string ifadenin tüm harflerini büyük harfe çeviriniz. Virgül ve nokta yerine space koyunuz, kelime kelime ayırınız.

text = "The goal is to turn data into information, and information into insight."

# Beklenen çıktı:
# ['THE', 'GOAL', 'IS', 'TO', 'TURN', 'DATA', 'INTO', 'INFORMATION', 'AND', 'INFORMATION', 'INTO', 'INSIGHT']

words = text.replace(',', ' ').replace('.', ' ').split()
words = [w.upper() for w in words]
print(words)

# YÖNTEM 2:
text.upper().replace(',', ' ').replace('.', ' ').split()


# GÖREV 3:  Verilen listeye aşağıdaki adımları uygulayınız.

lst = ["D", "A", "T", "A", "S", "C", "I", "E", "N", "C", "E"]

# Adım 1: Verilen listenin eleman sayısına bakınız.
# Adım 2: Sıfırıncı ve onuncu indeksteki elemanları çağırınız.
# Adım 3: Verilen liste üzerinden ["D", "A", "T", "A"] listesi oluşturunuz.
# Adım 4: Sekizinci indeksteki elemanı siliniz.
# Adım 5: Yeni bir eleman ekleyiniz.
# Adım 6: Sekizinci indekse "N" elemanını tekrar ekleyiniz.

# ADIM 1
print(len(lst))

# ADIM 2
print(lst[0], lst[10])

# ADIM 3
sub_list = lst[0:4]
print(sub_list)

# ADIM 4
removed = lst.pop(8)
print(lst)

# ADIM 5
lst.append("X")

# ADIM 6
lst.insert(8, "N")


# GÖREV 4:  Verilen sözlük yapısına aşağıdaki adımları uygulayınız.

dict = {'Christian': ["America", 18],
        'Daisy': ["England", 12],
        'Antonio': ["Spain", 22],
        'Dante': ["Italy", 25]
        }

# Adım 1: Dictionary içindeki key değerlerine erişiniz.
# Adım 2: Dictionary içindeki value değerlerine erişiniz.
# Adım 3: 'Daisy' key'ine ait 12 değerini 13 olarak güncelleyiniz.
# Adım 4: 'Ahmet' key değeri ve [Turkey, 24] value değerini dictionary'ye yeni bir değer olarak ekleyiniz.
# Adım 5: 'Antonio' key'ini dictionary'den siliniz.

# Adım 1
print(dict.keys())

# Adım 2
print(dict.values())

# Adım 3
dict["Daisy"][1] = 13

# Adım 4
dict['Ahmet'] = ["Turkey", 24]

# Adım 5
dict.pop('Antonio')


# GÖREV 5: Argüman olarak bir liste alan, listenin içerisindeki tek ve çift sayıları ayrı listelere atayan ve bu listeleri return eden fonksiyon yazınız.

l = [2, 13, 18, 93, 22]

def func(liste):
    even_list = []
    odd_list = []
    for item in liste:
        if item % 2 == 0:
            even_list.append(item)
        else:
            odd_list.append(item)
    return even_list, odd_list

func(l)


# GÖREV 6: Aşağıda verilen listede mühendislik ve tıp fakülterinde dereceye giren öğrencilerin isimleri bulunmaktadır.
# Sırasıyla ilk üç öğrenci mühendislik fakültesinin başarı sırasını temsil ederken son üç öğrencide tıp fakültesi öğrenci
# sırasına aittir. Enumarate kullanarak öğrenci derecelerini fakülte özelinde yazdırınız.

ogrenciler = ["Ali", "Veli", "Ayşe", "Talat", "Zeynep", "Ece"]

# Beklenençıktı:
# Mühendislik Fakültesi 1 . öğrenci: Ali
# Mühendislik Fakültesi 2 . öğrenci: Veli
# Mühendislik Fakültesi 3 . öğrenci: Ayşe
# Tıp Fakültesi 1 . öğrenci: Talat
# Tıp Fakültesi 2 . öğrenci: Zeynep
# Tıp Fakültesi 3 . öğrenci: Ece

# Mühendislik Fakültesi öğrencileri
for i, ogr in enumerate(ogrenciler[:3], 1):
    print(f"Mühendislik Fakültesi {i} . öğrenci: {ogr}")

# Tıp Fakültesi öğrencileri
for i, ogr in enumerate(ogrenciler[3:], 1):
    print(f"Tıp Fakültesi {i} . öğrenci: {ogr}")


# GÖREV 7: Aşağıda 3 adet liste verilmiştir. Listelerde sırası ile bir dersin kodu, kredisi ve kontenjan bilgileri yeralmaktadır.
# Zip kullanarak ders bilgilerini bastırınız.

ders_kodu = ["CMP1005", "PSY1001", "HUK1001", "SEN2204"]
kredi = [3,4,2,4]
kontenjan = [30,75,150,25]

# Beklenen Çıktı:
# Kredisi 3 olan CMP1005 kodlu dersin kontenjanı 30 kişidir.
# Kredisi 4 olan PSY1001 kodlu dersin kontenjanı 75 kişidir.
# Kredisi 2 olan HUK1005 kodlu dersin kontenjanı 150 kişidir.
# Kredisi 4 olan SEN2204 kodlu dersin kontenjanı 25 kişidir.

for kod, krd, kont in zip(ders_kodu, kredi, kontenjan):
    print(f"Kredisi {krd} olan {kod} kodlu dersin kontenjanı {kont} kişidir.")


# GÖREV 8: Aşağıda 2 adet set verilmiştir. Sizden istenilen eğer 1. küme 2. kümeyi kapsiyor ise ortak elemanlarını eğer kapsamıyor
# ise 2. kümenin 1. kümeden farkını yazdıracak fonksiyonu tanımlamanız beklenmektedir.

kume1 = set(["data", "python"])
kume2 = set(["data", "function", "qcut", "lambda", "python", "miuul"])

# BeklenenÇıktı:
# {'function', 'qcut', 'lambda', 'miuul'}

def kume_islem(kume1, kume2):
    if kume1.issuperset(kume2):   # kume1, kume2'yi kapsıyorsa
        print(kume1 & kume2)      # ortak elemanlar
    else:                         # kapsamıyorsa
        print(kume2 - kume1)      # kume2'nin kume1'den farkı

kume_islem(kume1, kume2)