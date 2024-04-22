"""

Unsupervised learning (gözetimsiz öğrenme), etiketlenmemiş veriler üzerinde çalışan ve veri setindeki yapıyı veya desenleri otomatik olarak keşfetmeye
çalışan makine öğrenmesi yaklaşımıdır. Gözetimsiz öğrenme, veri setindeki girdilere ait çıktılar veya etiketler olmadan, veriler arasındaki ilişkileri,
benzerlikleri veya farklılıkları bulmaya yöneliktir. Bu yaklaşım, genellikle veri keşfi, desen tanıma ve veri setinin içsel yapısının anlaşılmasında kullanılır.

Unsupervised Learning Örnekleri
Kümeleme (Clustering): Veri noktalarını benzer özelliklere sahip gruplara ayırmak için kullanılır. Kümeleme algoritmaları, veri içindeki doğal gruplamaları
veya segmentleri bulmaya çalışır.
    K-Means: En yaygın kümeleme tekniklerinden biridir. Veriyi 'K' sayıda kümeye ayırır. Her küme, kümedeki noktaların ortalaması olan bir merkez etrafında
    gruplanır.
    Hiyerarşik Kümeleme: Veri noktalarını birbirine benzerliklerine göre adım adım birleştirerek bir hiyerarşi oluşturur. Sonuçta, bir ağaç benzeri yapı
    elde edilir.
Boyut Azaltma (Dimensionality Reduction): Veri setindeki değişken sayısını azaltırken, veri setinin önemli özelliklerini mümkün olduğunca korumaya çalışır.
Bu, hem veriyi daha anlaşılır hale getirir hem de makine öğrenmesi modellerinin eğitim süresini kısaltabilir.
    Principal Component Analysis (PCA): Veri setindeki değişkenler arasındaki korelasyonları analiz eder ve bu değişkenleri daha az sayıda, birbiriyle
    ilişkisiz bileşenlere dönüştürür.
    t-Distributed Stochastic Neighbor Embedding (t-SNE): Yüksek boyutlu veri setlerini, özellikle görselleştirme amacıyla, iki veya üç boyuta indirgemek
    için kullanılır.
Derinlemesine Öğrenme (Deep Learning): Gözetimsiz öğrenme, derin öğrenme modellerinde de kullanılır. Bu modeller, genellikle veri setindeki karmaşık yapıları
ve ilişkileri otomatik olarak keşfetmek için tasarlanmıştır.
    Otoenkoderler (Autoencoders): Girdi verisini sıkıştırıp sonra tekrar oluşturarak verinin daha düşük boyutlu bir temsilini öğrenir. Bu süreç, verinin
    önemli özelliklerini yakalamayı amaçlar.
    Generative Adversarial Networks (GAN'ler): İki ağın (bir jeneratör ve bir diskriminatör) birbirine karşı eğitilmesiyle çalışır. Jeneratör, gerçek veriye
    benzeyen yeni örnekler üretmeye çalışırken, diskriminatör bu örneklerin gerçek mi yoksa sahte mi olduğunu ayırt etmeye çalışır.


Unsupervised learning, veri biliminde temel bir yaklaşımdır ve veri setlerindeki gizli yapıları keşfetmek, anormali tespiti, öneri sistemleri ve daha pek çok alanda kullanılır. Bu teknikler, özellikle etiketlenmiş verinin az olduğu veya hiç olmadığı durumlarda, veri hakkında önemli içgörüler elde etmek için değerlidir.

"""
# K - MEANS ALGORITHM

"""
K-Means, veriyi benzer özelliklere sahip gruplara (kümeler) ayırmak için kullanılan popüler bir kümeleme algoritmasıdır. Bu algoritma, veri içindeki doğal 
gruplamaları veya segmentleri keşfetmeyi amaçlar ve gözetimsiz öğrenme (unsupervised learning) tekniklerinden biridir.

K-Means Algoritmasının Çalışma Prensibi
Başlangıç Noktalarının Seçilmesi: Algoritma başlangıçta, veri seti içinden rastgele 'K' sayıda noktayı merkez (centroid) olarak seçer. 'K', oluşturulacak 
kümelerin sayısını ifade eder ve kullanıcı tarafından belirlenir.
Kümeleme: Her veri noktası, kendisine en yakın merkeze (en düşük uzaklık) sahip olan kümeye atanır. Uzaklık genellikle Öklid uzaklığı olarak hesaplanır.
Merkezlerin Güncellenmesi: Her atamadan sonra, her kümenin merkezi, kümeye ait noktaların ortalaması alınarak yeniden hesaplanır. Bu, kümelerin merkezini 
daha temsil edici hale getirir.
İterasyon: Adım 2 ve 3, kümelerdeki atamalar değişmeyene kadar veya belirlenen iterasyon sayısına ulaşana kadar tekrar edilir. Bu süreç, algoritmanın 
kümeleri iyileştirmesine ve veri setindeki doğal gruplamaları bulmasına olanak tanır.


Özellikleri ve Kullanım Alanları
Hız ve Basitlik: K-Means, basitliği ve hızı sayesinde büyük veri setleri üzerinde bile etkili bir şekilde çalışabilir.
Küme Sayısının Belirlenmesi: 'K' değeri, algoritmanın önemli bir parametresidir ve doğru küme sayısını belirlemek kritik öneme sahiptir. Çeşitli yöntemler 
(örneğin, dirsek yöntemi) bu seçimde yardımcı olabilir.
Kısıtlamalar: K-Means, küre şeklindeki kümelerle en iyi sonuçları verir ve farklı şekil ve yoğunluktaki kümelerde iyi performans göstermeyebilir. 
Ayrıca, başlangıç merkezlerinin seçimi sonuçları etkileyebilir.

Uygulama Alanları
Müşteri Segmentasyonu: Müşterileri satın alma davranışlarına, tercihlerine veya demografik özelliklerine göre gruplara ayırmak için kullanılır.
Belge Kümeleme: Benzer konulara veya kelimelere sahip belgeleri gruplamak için kullanılabilir, böylece benzer içerikler kolaylıkla sınıflandırılabilir.
Anomali Tespiti: Normalden sapma gösteren veri noktalarını tespit etmek için kümeleme kullanılabilir. Bu, özellikle finans ve ağ güvenliği gibi alanlarda önemlidir.


K-Means, veri bilimi ve makine öğrenmesi projelerinde yaygın olarak kullanılan güçlü ve esnek bir araçtır. Ancak, en iyi sonuçları elde etmek için veri 
setinin doğası ve algoritmanın kısıtlamaları dikkate alınmalıdır.


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

df.isnull().sum()
df.info()

df.describe().T
df.hist(figsize= (10,10))

# Model ve Görselleştirme

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 4)
kmeans.get_params()

k_fit = kmeans.fit(df)
k_fit.n_clusters
k_fit.cluster_centers_
k_fit.labels_

# Görselleştirme
kmeans = KMeans(n_clusters=2)
k_fit = kmeans.fit(df)

kumeler = k_fit.labels_
plt.scatter(df.iloc[:,0],df.iloc[:,1], c = kumeler, s = 50, cmap ="viridis")
merkezler = k_fit.cluster_centers_
plt.scatter(merkezler[:,0], merkezler[:,1], c = "black" , s=200, alpha = 0.5);

# 3 boyutlu gösterim için :
from mpl_toolkits.mplot3d import Axes3D
kmeans = KMeans(n_clusters=3)
k_fit = kmeans.fit(df)
kumeler = k_fit.labels_
merkezler = k_fit.cluster_centers_


plt.rcParams['figure.figsize'] = (16, 9)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2]);

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], c=kumeler)
ax.scatter(merkezler[:, 0], merkezler[:, 1], merkezler[:, 2],
           marker='*',
           c='#050505',
           s=1000);


# Kümeler ve Gözlem Birimleri

kmeans = KMeans(n_clusters = 4)
k_fit = kmeans.fit(df)
kumeler = k_fit.labels_

pd.DataFrame({"Eyaletler" : df.index, "Kumeler": kumeler})[0:10]

df["kume_no"] = kumeler
df.head()
df["kume_no"] = df["kume_no"] + 1



# Optimum küme sayısının belirlenmesi

from yellowbrick.cluster import KElbowVisualizer
import matplotlib.pyplot as plt

# KMeans modeli
model = KMeans()

# Matplotlib subplot tanımı
fig, ax = plt.subplots()

# KElbowVisualizer başlatılırken 'ax' parametresi ile subplot axes'i veriliyor
visualizer = KElbowVisualizer(model, k=(2,20), ax=ax)

# Veriyi sığdır ve görselleştiriciyi göster
visualizer.fit(df)
visualizer.show()














