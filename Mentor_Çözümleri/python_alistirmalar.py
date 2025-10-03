###############################################
# Python Alıştırmalar
###############################################


###############################################
# GÖREV 1: Veri yapılarının tipleriniz inceleyiniz.
###############################################

x = 8
type(x)

y = 3.2
type(y)

z = 8j + 18
type(z)

a = "Hello World"
type(a)

b = True
type(b)

c = 23 < 22
type(c)

l = [1, 2, 3, 4, "String", 3.2, False]
type(l)
# Sıralıdır
# Kapsayıcıdır
# Değiştirilebilir


d = {"Name": "Jake",
     "Age": [27, 56],
     "Adress": "Downtown"}
type(d)
# Değiştirilebilir
# Kapsayıcı
# Sırasız
# Key değerleri farklı olacak


t = ("Machine Learning", "Data Science")
type(t)
# Değiştirilemez
# Kapsayıcı
# Sıralı


s = {"Python", "Machine Learning", "Data Science", "Python"}
type(s)
# set([1,2,3]) # => set içerisinde iterable aldığı için [] verilir
# Değiştirilebilir
# Sırasız + Eşsiz
# Kapsayıcı


###############################################
# GÖREV 2: Verilen string ifadenin tüm harflerini büyük harfe çeviriniz. Virgül ve nokta yerine space koyunuz, kelime kelime ayırınız.
###############################################

text = "The goal is to turn data into information, and information into insight."


# YÖNTEM 1
new_text = text.upper()
new_text = new_text.replace(",", " ")
new_text = new_text.replace(".", " ")
new_text.split()


# YÖNTEM 2
text.upper().replace(",", " ").replace(".", " ").split()


###############################################
# GÖREV 3: Verilen liste için aşağıdaki görevleri yapınız.
###############################################

lst = ["D", "A", "T", "A", "S", "C", "I", "E", "N", "C", "E"]

# Adım 1: Verilen listenin eleman sayısına bakın.
len(lst)

# Adım 2: Sıfırıncı ve onuncu index'teki elemanları çağırın.
lst[0]
lst[10]



# Adım 3: Verilen liste üzerinden ["D","A","T","A"] listesi oluşturun.
data_list = lst[0:4]  # 0., 1., 2. ve 3. indexi alır
data_list

# Adım 4: Sekizinci index'teki elemanı silin.

lst.pop(8)
lst

# Bonus: .pop() sildiği elemanı çıktı olarak verir, bunu kaydedebiliriz.
# a = lst.pop(8)
# a
# lst

# YÖNTEM 2
# del lst[8]
# lst
### Örnek: Liste içerisindeki tüm 'A' değerlerini del ile sil
# for idx, i in enumerate(lst):
#     if i == "A":
#         del lst[idx]
# lst


# YÖNTEM 3
lst.remove("N")  # remove: ilk gördüğü karakteri kaldırır
lst
#
# lst.remove("A")
# lst

# Adım 5: Yeni bir eleman ekleyin.

lst.append(101)
lst

# # BONUS - EK BİLGİ
# # Başka bir ekleme yöntemi
# lst.append(["a", "b", "c"])  # eklenecek listeyi açmaz
# print(lst)
# lst.extend(["a", "b", "c"])  # lst = lst + ["a", "b", "c"]
# print(lst)



# Adım 6: Sekizinci index'e  "N" elemanını tekrar ekleyin.

lst.insert(8, "N")
lst


# # # # DİKKAT-KOPYALARKEN .copy() kullanmayı unutmayalım!!!
# lst = ["D", "A", "T", "A", "S", "C", "I", "E", "N", "C", "E"]
# new_lst = lst
# lst
# del new_lst[0]
# new_lst
# lst
#
# lst = ["D", "A", "T", "A", "S", "C", "I", "E", "N", "C", "E"]
# new_lst = lst.copy()
# lst
# del new_lst[0]
# new_lst
# lst

###############################################
# GÖREV 4: Verilen sözlük yapısına aşağıdaki adımları uygulayınız.
###############################################
d = {'Christian': ["America", 18],
        'Daisy': ["England", 12],
        'Antonio': ["Spain", 22],
        'Dante': ["Italy", 25]}


# Adım 1: Key değerlerine erişiniz.

d.keys()

# Adım 2: Value'lara erişiniz.

d.values()

# Adım 3: Daisy key'ine ait 12 değerini 13 olarak güncelleyiniz.
d.update({"Daisy": ["England", 13]})
d

###########
# 2. YÖNTEM - !!!
###########
d["Daisy"] = ["England", 13]
d

###########
# 3. YÖNTEM
###########
# Liste olduğu için
d["Daisy"][1] = 13
d


# Adım 4: Key değeri Ahmet value değeri [Turkey,24] olan yeni bir değer ekleyiniz.
d.update({"Ahmet": ["Turkey", 24]})
d

# 2. YÖNTEM
# NOT => Eğer key yoksa dict bunu oluşturur
d["Ahmet"] = ["Turkey", 24]
d

# Adım 5: Antonio'yu dictionary'den siliniz.

d.pop("Antonio")
d

# diğer yöntem => del
del d["Antonio"]
d

# DİKKAT-KOPYALARKEN .copy() kullanmayı unutmayalım!!!
# d_new = d
# d
# d_new
# d.pop("Daisy")
# d
# d_new
#
# # .copy()
# d_new = d.copy()
# d
# d_new
# d.pop("Ahmet")
# d
# d_new
###############################################
# GÖREV 5: Arguman olarak bir liste alan, listenin içerisindeki tek ve çift sayıları ayrı listelere atayan ve bu listeleri return eden fonskiyon yazınız.
###############################################

l = [2, 13, 18, 93, 22]


def func(lst):
    cift_list = []
    tek_list = []
    for i in lst:
        if i % 2 == 0:
            cift_list.append(i)
        else:
            tek_list.append(i)

    return cift_list, tek_list
cift, tek = func(l)
cift
tek


# LIST COMP. ÇÖZÜMÜ
def func(lst):
    cift_list = [i for i in lst if i % 2 == 0]
    tek_list = [i for i in lst if i % 2 != 0]
    return cift_list, tek_list
cift, tek = func(l)



def func(lst):
    cift_list, tek_list = [i for i in lst if i % 2 == 0], [i for i in lst if i % 2 != 0]
    return cift_list, tek_list
cift, tek = func(l)




def func(lst):
    return [i for i in lst if i % 2 == 0], [i for i in lst if i % 2 != 0]
cift, tek = func(l)
cift
tek

###############################################
# GÖREV 6: Aşağıda verilen listede mühendislik ve tıp fakülterinde dereceye giren öğrencilerin isimleri bulunmaktadır.
# Sırasıyla ilk üç öğrenci mühendislik fakültesinin başarı sırasını temsil ederken son üç öğrenci de tıp fakültesi öğrenci sırasına aittir.
# Enumarate kullanarak öğrenci derecelerini fakülte özelinde yazdırınız.
###############################################

ogrenciler = ["Ali", "Veli", "Ayşe", "Talat", "Zeynep", "Ece"]


for ind, ogr in enumerate(ogrenciler):
    if ind < 3:
        sira = ind + 1  # 0, 1, 2  yerine 1., 2., 3. yazabilmek için
        print("Mühendislik fakültesi", sira, ". öğrenci:", ogr)
    else:
        sira = ind - 2  # 3, 4, 5 yerine 1., 2., 3. yazabilmek için
        print(f"Tıp fakültesi {sira}. öğrenci: {ogr}")





###############################################
# GÖREV 7: Aşağıda 3 adet liste verilmiştir. Listelerde sırası ile bir dersin kodu, kredisi ve kontenjan bilgileri yer almaktadır. Zip kullanarak ders bilgilerini bastırınız.
###############################################

ders_kodu = ["CMP1005", "PSY1001", "HUK1005", "SEN2204"]
kredi = [3, 4, 2, 4]
kontenjan = [30, 75, 150, 25]

# list(zip(ders_kodu, kredi, kontenjan))
# dict(zip(ders_kodu, kredi))

for credit, kod, kont in zip(kredi, ders_kodu, kontenjan):
    print(f"Kredisi {credit} olan {kod} kodlu dersin kontenjanı {kont} kişidir.")



###############################################
# GÖREV 8: Aşağıda 2 adet set verilmiştir.
# Sizden istenilen eğer 1. küme 2. kümeyi kapsiyor ise ortak elemanlarını(kesişimlerini) eğer kapsamıyor ise 2. kümenin 1. kümeden farkını yazdıracak fonksiyonu tanımlamanız beklenmektedir.
###############################################

kume1 = set(["data", "python"])
kume2 = set(["data", "function", "qcut", "lambda", "python", "miuul"])

def kume(set1, set2):
    if set1.issuperset(set2):
        print(set1.intersection(set2))
    else:
        print(set2.difference(set1))

kume(kume1, kume2)


# set2.difference(set1) # => eleman set2'de olacak, set1'de olmayacak (set2 - set1)
# set1.difference(set2) # => eleman set1'de olacak, set2'de olmayacak (set1 - set2)

kume1.difference(kume2)
kume2.difference(kume1)



