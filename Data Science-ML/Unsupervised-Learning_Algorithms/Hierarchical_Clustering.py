"""
Hiyerarşik kümeleme, bir veri setini adım adım daha küçük kümeler halinde bölen veya daha büyük kümeler halinde birleştiren bir gözetimsiz öğrenme algoritmasıdır.
Bu yöntem, ağaç benzeri bir yapı olan bir dendrogram üzerinden görselleştirilebilir ve bu yapı verilerin nasıl gruplandırıldığını gösterir.

Hiyerarşik Kümelemenin İki Temel Yaklaşımı
Agglomerative (Birleştirici) Hiyerarşik Kümeleme: Bu yaklaşım "alttan yukarıya" (bottom-up) bir yöntemdir. Başlangıçta, veri setindeki her veri noktası bir küme olarak
kabul edilir. Algoritma, birbirine en yakın kümeleri adım adım birleştirerek devam eder ve bu süreç, tüm veriler tek bir kümede toplanana kadar devam eder.
Divisive (Bölücü) Hiyerarşik Kümeleme: "üstten aşağıya" (top-down) bir yaklaşımdır. Başlangıçta tüm veriler tek bir büyük küme olarak kabul edilir ve algoritma, bu
kümeyi adım adım daha küçük kümeler halinde bölerek ilerler. Bu süreç, her veri noktası kendi başına bir küme olana kadar devam eder.


Hiyerarşik Kümelemenin Özellikleri
Dendrogram: Kümeleme işleminin sonuçları genellikle bir dendrogram üzerinde görselleştirilir, böylece hangi veri noktalarının birbirine benzer olduğunu ve kümeleme
yapısını kolayca gözlemleyebiliriz.
Küme Sayısına Karar Vermek: Dendrogram üzerinden, bir kesme çizgisi çizerek istediğimiz sayıda kümeye bölme işlemi yapılabilir. Bu çizgi farklı yüksekliklerde
çizilerek farklı sayıda kümeye ulaşılabilir.
Uzaklık Ölçütleri: Kümelerin birleştirilmesi veya bölünmesi sırasında kullanılan uzaklık ölçütleri (Euclidean, Manhattan, Cosine vb.) algoritmanın sonuçlarını etkiler.
Bağlantı Metodları: Kümeleme sırasında, kümeler arasındaki benzerlikleri ölçmek için farklı bağlantı metodları (single linkage, complete linkage, average linkage, vb.) kullanılabilir.


Hiyerarşik kümeleme, özellikle veri setinin doğal hiyerarşik yapısını anlamak ve görselleştirmek istediğimizde kullanışlıdır. Ayrıca, küme sayısını önceden belirlemeye
gerek olmaması ve değişik küme sayıları ile esneklik sunması nedeniyle tercih edilen bir yöntemdir. Ancak, büyük veri setlerinde hesaplama yoğunluğu nedeniyle zaman
alabilir ve bu yüzden genellikle orta veya küçük ölçekli veri setleri için tercih edilir.
"""


from warnings import filterwarnings
filterwarnings('ignore')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import spicy as sp
import mpl_toolkits



# Veri işleme ve görselleştirme
df = pd.read_csv("Data Science-ML/Unsupervised-Learning_Algorithms/USArrests.csv")
df.head()
df.index = df.iloc[:,0]
df.head()
df = df.iloc[:,1:5]
df.index.name = None

from scipy.cluster.hierarchy import linkage,dendrogram
hc_complete = linkage(df,"complete")
hc_average = linkage(df,"average")
hc_single = linkage(df,"single")


plt.figure(figsize=(15,10))
plt.title("Hiyerarşik Kümeleme - Dendogram")
plt.xlabel("Indexler")
plt.ylabel("Uzaklık")
dendrogram(hc_complete,leaf_font_size=10)
dendrogram(hc_complete,truncate_mode="lastp", p=12, show_contracted=True) # p küme sayısını ifade ediyor.


# Optimum Küme Sayısı

from sklearn.cluster import AgglomerativeClustering
print(AgglomerativeClustering.__doc__)

cluster = AgglomerativeClustering(n_clusters=4, metric="euclidean",linkage = "ward")
cluster.fit_predict(df)
pd.DataFrame({"Eyaletler" : df.index, "Kumeler" : cluster.fit_predict(df)})[0:10]

df["kume_no"] = cluster.fit_predict(df)
df.head()






