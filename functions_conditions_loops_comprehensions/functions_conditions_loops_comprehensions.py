####################################################
# FONKSİYONLAR, KOŞULLAR, DÖNGÜLER, COMPREHENSIONS
####################################################
# - Fonksiyonlar (Functions)
# - Koşullar (Conditions)
# - Döngüler (Loops)
# - Comprehesions

##########################
# Fonksiyon Okuryazarlığı
##########################

print("a")
# ?print # Print bir fonksiyondur. Bu şekilde sorgularken konsola yazmamız gerekir. Kod tarafına yazarsak kod akışını
# bozacağı için python uyarı verir.

# Argüman ne demek ? Argümanlar bir özellik belirttiği gibi fonksiyonun genel amacını biçimlendirmek üzere kullanılan
# alt görevlerdir.

print("a", "b", sep="__")


# help(print) ile de görebiliriz.


#########################
# Fonksiyon Tanımlama
#########################

def calculate(x):  # def fonksiyon oluşturmak için kullanılır. calculate yazdığımız kısım fonksiyona vermek istediğimizi adı, x ise parametreyi tanımlar.
    print(x * 2)


calculate(3)

# İki argümanlı/parametreli bir fonksiyon tanımlayalım.

def sum(arg1, arg2):
    print(arg1 + arg2)

sum(7,8)
sum(8,7) # Sıralamanında bir önemi vardır. İlk girilen değer ilk parametreye eş değer olur.

sum(arg2=8, arg1=7) # Parametre adlarını biliyor isek bu şekilde sırası önemli olmadan atama yapabiliriz.


################
# Docstring
################

# Fonksiyonlara herkesin anlayabileceği ortak bir dil ile bilgi notu ekleme yoludur.

def sum2(arg1, arg2): # Bu şekilde kimse bu fonksiyonun ne yaptığını, fonksiyonu hazırlayan kişi haricinde bilemez.
    print(arg1 + arg2)


def summer(arg1, arg2):
    """
    Sum of two numbers

    Parameters
    ----------
    arg1 : int or float
    arg2 : int or float

    Returns
    -------
    int or float
        The sum of arg1 and arg2.
    """
    print(arg1 + arg2)

# Fonksiyon içerisine 3 tırnak atıyarak içerisine, fonksiyon hakkında bilgi hazırlayabiliriz.
# İlk kısımda fonkisyonun ne yaptığı bilgisi yazılır.
# 2.bölümde parameters (param) ya da argümanlar da (args) ifadesini de kullanabiliriz.
# Bu parametrelerin tiplerini yazabiliriz
# Parametreleri yazarken görevinin ne olduğunu altına yazabiliriz.
# Bu fonksiyonun bir şeyi return edecek ise çıktı değil bunun bilgisini de verebiliriz.
# Help menüsünden fonksiyonumuzu incelediğimiz de hazırlanan bilgi notu gelecektir.

# 1 - Fonksiyon ne yapar ? yazılır.
# 2 - Argümanlar ne yapar ? yazılır.
# 3 - Çıktısı ne olacak ? yazılır.


#########################
# Fonksiyonların Statement/Body Bölümü
#########################

# def function_name(parameters/arguments): Bir fonksiyon parametre veya argümansız da olabilir.
#     statements(function body)

def say_hi(string):
    print(string)
    print("Hi")
    print("Hello")


say_hi("Merhaba")


def multiplication(a, b):
    c = a * b
    print(c)


multiplication(10, 9)


# Girilen değerleri bir liste içinde saklayacak fonksiyon

list_store = []


def add_element(a,b):
    c = a + b
    list_store.append(c)
    print(list_store)


add_element(1,8)
add_element(18,8)
add_element(180,10)


#########################
# Ön Tanımlı Argümanlar/Parametreler (Default Parametres/Arguments)
#########################

def divide(a, b):
    print(a / b)

divide(1,2)

def divide(a, b=1):
    print(a / b)

divide(1)

# Varsayalım ki bir fonksiyonda çok fazla argüman var ya da fonksiyonun özelinde de düşünebiliriz.
# Bir şekilde kullanıcıyla etkileşim sağlayacak olan bu fonksiyonu kullanıcı daha rahat kullanabilsin diye
# varsaydığımızı düşünelim. Bu durumda bazı argümanlara ön tanımlı değerler ekleyerek, kullanıcılar her argümanı
# girmemiş olsa dahi fonksiyonların çalışmasını sürdürmesini sağlayabiliriz.


def say_hi(string="Hello"):
    print(string)
    print("Hi")
    print("Hello")


say_hi()
# Burada parametre almasa dahi argümanda ön tanımlı default bir değer olduğu için hata vermeyecek bunun yerine default değeri verecek.
# Eğer kullanıcı parametre girerse o parametreyi dönecektir.


###########################################
# Ne Zaman Fonksiyon Yazma İhtiyacımız Olur?
###########################################

# Diyelim ki bir belediyede işe girdik ve belediyenin çeşitli alanlarda kurmuş olduğu akıllı sokak lambaları var.
# Bu sokak lambalarından gelen bazı sinyaller olduğunu, bilgiler olduğunu düşünelim.
# Nasıl bilgiler? Isı, nem ve pil durumu gibi bilgiler geldiğini düşünelim. Diyelim ki bu akıllı sokak lambalarından
# gelen bilgileri kullanarak bazı hesaplamalar yapmak istiyoruz.
# Diyelim ki ısı bilgisini, nem bilgisini ve pil durumu bilgisini kullanarak çeşitli matematiksel işlemler yapacağız.
# Örneğin ikisini toplayıp diğerine bölmek gibi, bu bize bir skor verecek. Buna göre çeşitli aksiyonlar alınacak
# diye düşünelim.

# Örneğin birinci sokak lambası için bunu hesaplayalım.
# Örnek; (56 + 15) / 80'i hesapladı. Değer geldi.
(56 + 15) / 80

# İkinci sokak lambası için bu lazım olduğunda bilgi geldi. Yine yazdık, hesapladık.
(17 + 45) / 70

# Üçüncü de gelsin onu da hesaplayalım.
(52 + 45) / 80

# Bunu kendimizi tekrar ederek yapmamızın çok işlevsel olmadığı açıktır. Bu sebeple temel programlama prensiplerinden
# birisi der ki, "Don't Repeat Yoursel" yani DRY prensibi der ki "Kendini Tekrar Etme". Bu sebeple birbirini tekrar
# eden görevler söz konusu olduğunda fonksiyon yazma ihtiyacı kendini hissettirecektir.

def calculate(varm, moisture, charge):
    print((varm + moisture) / charge)

calculate(98,12,78)


###########################################
# Return: Fonksiyon Çıktılarını Girdi Olarak Kullanmak
###########################################

def calculate(varm, moisture, charge):
    print((varm + moisture) / charge)

# calculate(98,12,78) * 10
# Hata verecektir. Çünkü fonksiyon parametreleri biz int atasak dahi NoneType'dır. NoneType ile int veya diğer
# değişken türleri ile işlem yapılamaz. Bunun için bize return gerekir.


def calculate(varm, moisture, charge):
    return((varm + moisture) / charge)

a = calculate(98,12,78) * 10
# Burada return ifadesi çıktıyı dışarıya yönlerdireceği için 10 ile çarpabiliyoruz.


def calculate(varm, moisture, charge):
    varm = varm * 2
    moisture = moisture * 2
    charge = charge * 2
    output = (varm + moisture) / charge
    return varm, moisture, charge, output


calculate(98,12,78)
# 4 tane değer dönecek

varm, moisture, charge, output = calculate(98,12,78)
# Bu şekilde çıktıları saklayabiliriz. Verdiğimiz değerler fonksiyon içerisinde işlem görür ve return yeni değerleri gönderir.
# Burada verdiğimiz değişken adları return ile farklı olabilir. Return sırasına göre değer alırlar.


###########################################
# Fonksiyon İçerisinden Fonksiyon Çağırmak
###########################################

def calculate(varm, moisture, charge):
    return int((varm + moisture) / charge)

calculate(98,12,78) * 10

def standartdization(a, p):
    return a * 10 / 100 * p * p

standartdization(45, 1)

# DİKKAT: Eğer bir fonksiyon tanımlıyorsak ve bu fonksiyon içerisinde başka fonksiyonlar yer alacaksa yani çağrılacaksa
# çağırmış olduğunuz fonksiyonların argümanların da eğer bu tanımladığımız ana fonksiyondan biçimlendirmek istersek,
# bu durumda ana fonksiyonu tanımlarken fonksiyonun misafir edeceği fonksiyonların argümanlarını da girmemiz gerekecektir.

def all_calculation(varm, moisture, charge, p):
    a = calculate(varm, moisture, charge)
    b = standartdization(a, p)
    print(b * 10)

all_calculation(1,3,5, 12)


###########################################
# Local & Global Değişkenler
###########################################

list_store = [1, 2] # Bu değişkene her yerden ulaşabileceğimiz için Global Scope aralığında olan bir değişkendir.

def add_element(a, b):
    c = a * b # Burada ki c değişkeni dışarıdan erişilemez çünkü Local aralıkdadır. Sadece add_element fonksiyonunda bulunur.
    list_store.append(c)
    print(list_store)

add_element(1,9)


###########################################
# KOŞULLAR (CONDITIONS)
###########################################

# Koşullar bir program yazımı esnasında akış kontrolü sağlayan ve programların belirli kurallara göre ya da koşullara
# göre nasıl hareket etmesi gerektiğini biz kullanıcılar tarafından programa bildirme imkanı sağlayan yapılardır.
# Örneğin; eğer bu koşul sağlanıyorsa şunu yap, sağlanmıyorsa bunu yap şeklinde akışlar sağlamaya yarar.

# True-False'u hatırlayalım.
1 == 1
1 == 2

# if = Eğer bu koşul sağlanıyorsa şunu yap, bu koşul sağlanmıyorsa şunu yap şeklinde ya da bir şey yapma şeklindedir.
# Python a bir akış ifade etmek istediğimizde kullandığımız araçlardan birisidir.

if 1 == 1:
    print("something")

if 1 == 2:
    print("something")

number = 11

if number == 10:
    print("number is 10")

number = 10

if number == 10:
    print("number is 10")

def number_check(number):
    if number == 10:
        print("number is 10")

number_check(12)

# Eğer if False gelirse o zaman ne olacak burada Else & Elif devreye girer.


##########
# else = Eğer koşul sağlanmıyor ise else içerisinde ki kodlar çalışır.
###########

def number_check(number):
    if number == 10:
        print("number is 10")
    else:
        print("number is not 10")

number_check(12)


##########
# elif
###########

# Eğer 3 lü sorgu yapacak olursak, Örnek; 10 dan büyük ise bunu, küçük ise bunu, eşit ise bunu yap gibi

def number_check(number):
    if number > 10:
        print("greater than 10")
    elif number < 10:
        print("less than 10")
    else:
        print("equal to 10")

number_check(10)
number_check(12)
number_check(8)


######################
# DÖNGÜLER (LOOPS)
######################

# for loop

students = ["John", "Mark", "Venessa", "Mariam"]

students[0]
students[1]
students[2]

# Bütün elemanlara ulaşmak istediğim zahmetli bir iş çıkıyor. Döngüler ise bu zahmeti kurtarıyor.

for student in students:
    print(student)

for student in students:
    print(student.upper())

salaries = [1000, 2000, 3000, 4000, 5000]

for salary in salaries:
    print(salary)

for salary in salaries:
    print(int(salary * 20 / 100 + salary))

for salary in salaries:
    print(int(salary * 30 / 100 + salary))

def new_salary(salary, rate):
    return int(salary * rate / 100 + salary)

new_salary(1500, 10)
new_salary(2000, 20)

for salary in salaries:
    print(new_salary(salary, 10))

# 3000'den büyük olanlara farklı bir zam altında olanlara farklı bir zam yapalım

for salary in salaries:
    if salary >= 3000:
        print(new_salary(salary, 10))
    else:
        print(new_salary(salary, 20))


######################
# UYGULAMA - Mülakat Sorusu
######################

# Amaç: Aşağıdaki şekilde string değiştiren fonksiyon yazmak istiyoruz.

# before : "hi my name is john and i am learning python"
# after : "Hi mY NaMe iS JoHn aNd i aM LeArNiNg pYtHoN"

# range(len("miuul")) range bize 2 değer arasında sayı üretme imkanı sağlar.
range(0, 5)

for i in range(len("miuul")):
    print(i)

def alternating(string):
    new_string = ""
    # girilen string'in index'lerinde gez
    for string_index in range(len(string)):
        # index çift ise büyük harfe çevir
        if string_index % 2 == 0:
            new_string += string[string_index].upper()
        # index tek ise küçük harfe çevir
        else:
            new_string += string[string_index].lower()
    print(new_string)

alternating("hi my name is john and i am learning python")


######################
# break & continue & while
######################

# Döngülerle birlikte kullanılan bu yapılar bir program akışında, akışı kesmeye ya da ilgili şart gözlemlendiğinde o
# şartı atlayarak devam etmeye, bir koşul sağlandığı sürece çalışmayı sürdürmeye yarayan ifadelerdir.

salaries = [1000, 2000, 3000, 4000, 5000]

# break ifadesi aranan bir koşul yakalandığında döngünün durmasını sağlar.

for salary in salaries:
    if salary == 3000: # burada 3000 ne geldiğini döngü durur.
        break
    print(salary)

# continue

for salary in salaries:
    if salary == 3000: # burada 3000 ne geldiğini onu atlayarak devam eder.
        continue
    print(salary)

# while - dı sürece demektir.

number = 1

while number < 5: # sayı 5 olana kadar devam et
    print(number) # sayıyı yazdır.
    number += 1 # sayıya 1 ekleyerek güncelle, sayı güncellendikden sonra yeni değeri sorguda sorgulanır ve ona göre işlem devam eder.


######################
# Enumerate: Otomatik Counter/Indexer ile for loop
######################

# Otomatik index üretici ya da otomatik counter diye geçen ve for loop ile birlikte kullanılabilen bir yapıdır.
# Bir iteratif nesne içinde gezip elemanlarına bir şey yaparken aynı zamanda o elemanların index bilgilerini de takip
# etmek istediğimizde kullanılır. Genellikle, derinlemesine bir projenin içerisinde bu problemle karşılaşıp nasıl
# çözülebileceği bile anlaşılamadan uzun zamanlar harcanmasına sebep olabilecek bir problemdir.

# Örneğin; bir liste içerisinde gezerken bu elemanlara belirli bir işlem uygularken aynı zamanda işlem uygulanan
# elemanların index bilgisini de tutup gerekirse bu index bilgisine göre de bir işlem yapmak istediğimizde kullanılan
# bir yapıdır.

students = ["John", "Mark", "Vanessa", "Mariam"]

# Index'i çift olanlara bir işlem, tek olanlara başka bir işlem yapmak istediğimizi düşünelim.
# Bunu, örneğin; range ve len gibi ifadeler kullanarak index üretebileceğim gibi, buradaki her bir elemanla eşleşmiş
# index'ler kullanarak da bu takip etme işlemini gerçekleştirebiliriz.

for student in students:
    print(student)

for index, student in enumerate(students):
    print(index, student)
# enumerate() ile hem listedeki sırayı (index) hem de elemanın kendisini (student) aynı anda alıyoruz. Varsayılan olarak index 0'dan başlar.

for index, student in enumerate(students, 1):
    print(index, student)
# enumerate()'in ikinci parametresi, index’in hangi sayıdan başlayacağını belirler. Burada 1 yazdığımız için index 0 yerine 1'den başlar.

A = []
B = []

for index, student in enumerate(students):
    if index % 2 == 0:
        A.append(student)
    else:
        B.append(student)


######################
# Uygulama - Mülakat Sorusu
######################

# divide_students fonksiyonu yazınız
# çift indexte yer alan öğrencileri bir listeye alınız
# Tek indexte yer alan öğrencileri başka bir listeye alınız.
# Fakat bu iki liste tek bir liste olarak return olsun

students = ["John", "Mark", "Venessa", "Mariam"]

def divide_students(students):
    groups = [[], []] # İki ayrı boş liste içeren bir liste tanımlıyoruz.
    # Bu yapı, örneğin iki farklı gruptaki verileri (grup1 ve grup2) ayrı ayrı saklamak için kullanılabilir.
    for index, student in enumerate(students):
        if index % 2 == 0:
            groups[0].append(student)
        else:
            groups[1].append(student)
    print(groups)
    return groups

st = divide_students(students)
st[0] # 1.grubun elemanlarını döner
st[1] # 2.grubun elemanlarını döner


######################
# alternating fonksiyonunun enumerate ile yazılması
######################

# Öyle bir fonksiyon yazmalıyız ki kendisine girilen string ifadelerin çift index'lerinde yer alan karakterleri
# büyültsün, tek index'lerinde yer alan karakterlerini küçültsün.

# Dolayısıyla yapılması gereken şey girilen string'lerin içerisinde gezmemiz gerektiğidir.

def alternating_with_enumerate(string):
    new_string = ""
    for i, letter in enumerate(string):
        if i % 2 == 0:
            new_string += letter.upper()
        else:
            new_string += letter.lower()
    print(new_string)


alternating_with_enumerate("hi my name is john and i am learning python")

# enumerate yerine alternatif olarak range ve len kullanabiliriz fakat kod okunabilirliği ve takibi açısından daha kolay
# takip etme imkanı sağlamaktadır.


######################
# Zip
######################

students = ["John", "Mark", "Vanessa", "Mariam"]

departments = ["Computer Science", "Science", "statistics", "astronomy"]

ages = [25, 35, 45, 50]

# Elimizde 3 farklı liste olsun. Biri öğrencileri, biri departmanları, bir diğeri de yaşları tutsun.
# Diyelim ki şöyle bir amacımız var; burada ki 3 farklı listenin elemanlarını birlikte kullanmak istiyoruz.
# Zip, birbirinden farkı şekilde olan listeleri bir arada değerlendirme imkanı sağlar.


list(zip(students, departments, ages))

# 3 listeyi birleştirerek getirdi. Zipleme şekli ise her elemana denk gelen index numarasına göre.


######################
# Lambda, Map, Filter & Reduce (azaltmak)
######################

# Python'ın fonksiyonel programlama yönüne hitap eden vektörel seviyede işlemler yapma yönüne hitap eden bazı araçlardır.

def summer(a, b):
    return a + b

summer(1, 3) * 9

# Lambda nedir ? Lambda kullan-at fonksiyondur. Yani bir tanımlama atama işlemi yapılmaksızın kullanılabilir.

new_sun = lambda a, b: a + b # hatalı kullanım

# Lambda da bir fonksiyon tanımlama şeklidir.
# def'ten farkı bunlar kullan at şeklinde çalışır. apply gibi, map gibi diğer bazı araçlarda kullanıldığında asıl
# kullanılma amacını yerine getirmektedir. Yani yukarıda ki yazım şekli ile kullan at özelliğini taşımamaktadır.

# Kullan-at demek değişkenler bölümünde yer tutmadan ihtiyaç duyulan bir noktada kullanılıp atılması anlamına gelmektedir.


# map

# map nedir ? map fonksiyonu bizi döngü yazmaktan kurtarır. İçerisinde gezebileceğimiz, iteratif bir nesne veririz ve
# bu nesneye uygulamak istediğimiz fonksiyonu veririz. Bize bu işlemi otomatik olarak yapar.

salaries = [1000, 2000, 3000, 4000, 5000]

# diyelim ki burada ki maaşlara bir zam uygulamak istiyoruz

def new_salary(x):
    return x * 30 / 100 + x

new_salary(1000)

for salary in salaries:
    print(new_salary(salary))


# şimdi map'in işlevselliğini göreceğiz. Öncelikle fonksiyonları map e veriyoruz.

list(map(new_salary, salaries))

# Döngü yazmadan for yazmadan bütün elemanlara gitti ve bir map işlemi yaptı.
# Şimdi lambda ile map ilişkisine gelelim.

list(map(lambda x: x * 20 / 100 + x, salaries))
# map, salaries listesindeki her elemanı tek tek lambda fonksiyonuna gönderir;
# lambda ise gelen elemanın %20 fazlasını hesaplayıp geri döndürür.


# FILTER -----> Filtreleme işlemlerinde kullanılır.

list_store = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Örneğin; elimizde böyle bir liste var. Bu listede belirli bir koşulu sağlayanları seçmek istediğimizi, belirli bir
# koşulu sağlamayanları seçmek istemediğimizi düşünelim.

list(filter(lambda x: x % 2 == 0, list_store))


# REDUCE ( İNDİRGEMEK )

from functools import reduce

# Bu yapıyı kullanbilmek için functool içerisinden reduce metodunu, fonksiyonunu import ediyoruz.

list_store = [1, 2, 3, 4]

# Diyelim ki elimizde böyle bir liste var. Bu sefer yapmak istediğimiz şey iteratif bir şekilde ilgili elemanlara
# tek tek belirli bir işlemi uygulamak.

reduce(lambda a, b: a + b, list_store)
