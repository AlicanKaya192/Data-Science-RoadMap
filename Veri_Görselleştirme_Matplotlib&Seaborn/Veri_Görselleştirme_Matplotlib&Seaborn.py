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
















