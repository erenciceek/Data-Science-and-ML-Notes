"""
PCA (Principal Component Analysis - Temel Bileşen Analizi), boyut azaltma ve veri analizi için sıkça kullanılan bir unsupervised (denetimsiz) öğrenme algoritmasıdır.

PCA'nın amacı, veri setindeki değişken sayısını azaltmak ve veri setindeki karmaşıklığı anlamak için veri setindeki değişkenler arasındaki ilişkileri belirlemektir.
Özellikle büyük boyutlu veri setlerinde, veri setindeki gereksiz bilgiyi azaltmak ve veri setinin temel özelliklerini korumak için kullanılır.

PCA'nın temel amacı, veri setindeki değişkenler arasındaki korelasyonu kullanarak yeni değişkenler (temel bileşenler) oluşturmaktır. Bu temel bileşenler, orijinal
veri setinin değişkenlerinin bir kombinasyonu olarak oluşturulur ve orijinal veri setinin varyansını en iyi şekilde açıklamak için seçilir.

PCA'nın çalışma adımları şunlardır:
Veri Standardizasyonu (Normalize Etme): PCA, veri setindeki değişkenlerin ölçeklerinin birbirine benzer olmasını gerektirir. Bu nedenle, veri setindeki her değişkenin
ortalamasını sıfır ve standart sapmasını bir yaparak veriyi normalize ederiz. (veya standardizasyon)
Kovaryans Matrisinin Hesaplanması: Veri setindeki değişkenler arasındaki kovaryansı (ilişkiyi) belirlemek için kovaryans matrisi hesaplanır.
Özdeğer ve Özvektörlerin Hesaplanması: Kovaryans matrisinin özdeğerleri ve özvektörleri hesaplanır. Özvektörler, orijinal değişkenlerin temel bileşenlerini temsil eder.
Temel Bileşenlerin Seçilmesi: En yüksek varyansı açıklayan özvektörler temel bileşenler olarak seçilir.
Yeni Veri Kümesinin Oluşturulması: Seçilen temel bileşenler kullanılarak, orijinal veri setindeki boyut azaltılarak yeni bir veri kümesi oluşturulur.


PCA'nın uygulama alanlarından bazıları şunlardır:
Veri Görselleştirme: Yüksek boyutlu veri setlerinin iki veya üç boyutlu grafiklerle görselleştirilmesi için kullanılır.
Gürültü Azaltma: Verilerdeki gürültüyü azaltmak için kullanılabilir, çünkü düşük varyans gösteren bileşenler genellikle gürültüyü içerir ve bu bileşenler atılarak
verinin daha temiz bir versiyonu elde edilebilir.
Özellik Çıkarımı ve Veri Sıkıştırma: Büyük veri setlerindeki özellik sayısını azaltmak ve veri saklama/muhafaza etme maliyetlerini düşürmek için kullanılır.
Modelleme: Makine öğrenmesi modellerinde, gereksiz boyutlardan kaynaklanabilecek boyutun laneti (curse of dimensionality) sorununu hafifletmek için kullanılır ve
modelin daha iyi genelleme yapmasına yardımcı olur.

PCA'nın başlıca bir dezavantajı, temel bileşenlerin yorumlanmasının zor olabilmesidir. Bu nedenle, PCA'nın uygulanmasıyla elde edilen temel bileşenlerin orijinal
değişkenlerle anlamlı bir şekilde ilişkilendirilmesi gerekir.

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

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df = scaler.fit_transform(df)
df[0:5,0:5]

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_fit = pca.fit_transform(df)
component_df = pd.DataFrame(data = pca_fit, columns = ["first_component","second_component"])
component_df.head()

pca.explained_variance_ratio_


pca = PCA().fit(df)
plt.plot(np.cumsum(pca.explained_variance_ratio_))


