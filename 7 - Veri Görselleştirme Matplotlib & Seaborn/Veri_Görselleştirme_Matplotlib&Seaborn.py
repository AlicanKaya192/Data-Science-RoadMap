###########################################
# VERİ GÖRSELLEŞTİRME: MATPLOTLIB & SEABORN
###########################################

################
# MATPLOTLIB
################

# Düşük seviye bir veri görselleştirme aracıdır. "Bu ne anlama gelir?" Seaborn'a kıyas ile ifade edecek olursak daha fazla
# çabayla veri görselleştirme yapmak demek anlamına gelir. Yani örneğin; bir görevimiz var ve bu görev içerisinde birden
# fazla alt görevi barındırıyor. Çizilecek olan grafiğin özellikleri, veri ifade etmek vesaire gibi. Bu iki kütüphane
# karşılaştırıldığında seaborn high level'dır, yani yüksek seviyedir. Yani daha az çabayla daha fazla şey yapma imkanı sağlar
# ama matplotlib python'daki veri görselleştirme araçlarının atası olması sebebiyle yine diğer birçok veri görselleştirme
# kütüphanesiyle bir şekilde ilişkilidir ve bir şekilde matplotlib'in özellikleri kullanılmaktadır.
# "Hangi noktalarda?" Denilecek olursa, özellikle grafik biçimlendirme noktalarında diğer seaborn'un dışındaki de birçok
# kütüphaneye yine matplotlib'in katkısı vardır.


# Kategorik değişken: sütun grafik. seaborn içerisinde ki countplot ya da  matplotlib içerisinde ki bar
# Sayısal değişken: histogram, boxplot


################################
# Kategorik Değişken Görselleştirme
################################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()

df["sex"].value_counts().plot(kind="bar")
# Kategorik değişkenlerde sınıf (kategori) frekanslarını görmek için value_counts() kullanılır.
# plot(kind="bar") ifadesi bu frekansları çubuk grafik (bar chart) olarak görselleştirir.

plt.show()
# Grafiği ekranda görüntüler.


################################
# Sayısal Değişken Görselleştirme
################################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()

plt.hist(df["age"])
plt.show()

plt.boxplot(df["fare"])
plt.show()


################################
# Matplotlib'in Özellikleri
################################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option("display.width", 500)
df =sns.load_dataset("titanic")

# Matplotlib yapısı itibariyle katmanlı şekilde veri görselleştirme imkanı sağlar. Bu şu anlama gelir; bir katmanda bir görsel,
# diğer katmanda ayrı bir görsel, diğer katmanda bit title, bir isimlendirme, örneğin diğer bir katmanda ise eksenlere bilgi vermek gibi
# diğer çeşitli bazı başlıklarda çalışma imkanı sağlar. Bu yapısını kullanarak bazı ihtiyaçlarımızı giderebiliriz.


################################
# plot
################################
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


x = np.array([1, 8])
y = np.array([0, 150])

plt.plot(x, y)
plt.show()

plt.plot(x, y, 'o') # Köşelere 2 nokta koyar.
plt.show()

# Eğer hali hazırda çık olan görselleştirme grafiği kapatılmadan yeni bir tanesi çalıştırılırsa eğer yeni olan açık olanın üzerine koyar.

x = np.array([2, 4, 6, 8, 10])
y = np.array([1, 3, 5, 7, 9])

plt.plot(x, y)
plt.show()

plt.plot(x, y, 'o') # Burada noktalar daha fazla sayıda olacak. X array'indeki her bir değere nokta koyar.
plt.show()


################################
# marker
################################

y = np.array([13, 28, 11, 100])

plt.plot(y, marker='o') # y'ye market eklemeke istersek eğer.
plt.show()

plt.plot(y, marker='*') # yıldız da koyabiliriz.
plt.show()

marker = ['o', '*', '.', ',', 'x', 'X', '+', 'P', 's', 'D', 'd', 'p', 'H', 'h']


################################
# line
################################

# matplotlib kütüphanesini kullanarak çizgiyi oluşturma işlemlerine olanak sağlar.

y = np.array([13, 28, 11, 100])
plt.plot(y, linestyle='dashed', color="r") # dashed ile kesikli gelir, color ile kesikli çizgilere renk verebiliriz.
plt.show()

# dotted (noktalar), dashdot(hem nokta hem kesikli çizgi)


################################
# Multiple Lines
################################

x = np.array([23, 18, 31, 10])
y = np.array([13, 28, 11, 100])

plt.plot(x)
plt.plot(y)
plt.show()
# Bu şekilde peş peşe bunları gösterebiliriz.


################################
# Labels (Etiketler)
################################

x = np.array([80, 85, 90, 95, 100, 110, 120])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.plot(x, y)
# Başlık
plt.title("Bu ana başlık")

# X eksenini isimlendirme
plt.xlabel("X ekseni isimlendirmesi")

# Y eksenini isimlendirme
plt.ylabel("Y ekseni isimlendirmesi")

plt.grid() # Tablo nun arkasına ızgara ekler.
plt.show()


################################
# Subplots
################################

# Birden fazla görselin gösterilmesi

# plot 1
x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.subplot(1, 3, 1)
plt.title("1")
plt.plot(x, y)

# plot 2
x = np.array([8, 8, 9, 9, 10, 15, 11, 15, 12, 15])
y = np.array([24, 25, 26, 27, 28, 29, 30, 31, 32, 33])
plt.subplot(1, 3, 2)
plt.title("2")
plt.plot(x, y)

# plot 3
x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.subplot(1, 3, 3)
plt.title("3")
plt.plot(x, y)

plt.show()


################################
# SEABORN
################################
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


# "Seabron nedir?" Seaborn bir veri görselleştirme kütüphanesidir. Veri görselleştirme işlemleri için kullanılan
# yüksek seviyede bir kütüphanedir. Buradaki yüksek seviye ifadesi daha az çabayla daha çok iş yapabilmek anlamına gelmektedir.
# Dolayısıyla seaborn matplotlib ile kıyaslandığında daha az çabayla bize daha fazla, ya da daha kolay bir şekilde grafikler
# oluşturma imkanı sağlamaktadır.

df = sns.load_dataset("tips") #Örnek veri seti
df.head()

df["sex"].value_counts()
sns.countplot(x=df["sex"], data=df)
plt.show()

df["sex"].value_counts().plot(kind="bar")
plt.show()


################################
# Sayısal Değişken Görselleştirme
################################

sns.boxplot(x=df["total_bill"])
plt.show()

df["total_bill"].hist()
plt.show()