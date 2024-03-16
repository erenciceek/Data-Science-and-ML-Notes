"""
Artifical Neural Networks

Yapay Sinir Ağları (YSA), insan beyninin bilgi işleme şeklini taklit etmeye çalışan bilgisayar sistemleridir.
Biyolojik sinir ağlarının temel yapı taşları olan nöronlardan esinlenerek geliştirilmiştir. YSA'lar, veri
madenciliği ve desen tanıma gibi görevlerde sıkça kullanılır ve makine öğreniminin temel yapıtaşlarından biridir.

Yapı
YSA'lar, birbiriyle bağlantılı düğümler veya nöronlar ağından oluşur. Bu düğümler genellikle katmanlar halinde düzenlenir:

Giriş Katmanı (Input Layer): Verinin modele girdiği ilk katmandır. Her bir düğüm, bir veri özelliğine (feature) karşılık gelir.
Gizli Katmanlar (Hidden Layers): Giriş ve çıkış katmanları arasında bir veya daha fazla gizli katman bulunabilir.
                                Bu katmanlardaki nöronlar, giriş verisini karmaşık işlevler aracılığıyla işler ve modelin öğrenmesini sağlar.
Çıkış Katmanı (Output Layer): Modelin tahmin veya sınıflandırma sonuçlarını verir.

Her bir nöron, bir önceki katmandan gelen sinyalleri toplar, bir ağırlıklandırma işlemi uygular ve genellikle bir
aktivasyon fonksiyonu (örn. sigmoid, ReLU) geçirerek sonraki katmana iletilmek üzere bir çıkış üretir.

Öğrenme Süreci
YSA'lar, genellikle geri yayılım (backpropagation) ve gradyan inişi (gradient descent) gibi öğrenme algoritmaları kullanarak eğitilir:

İleri Yayılım (Forward Propagation): Giriş verisi ağ üzerinden ileri doğru yayılır ve çıkış katmanında bir tahmin üretir.
Kayıp Fonksiyonu (Loss Function): Modelin tahmininin gerçek değerlerden ne kadar farklı olduğunu ölçer. Örneğin, regresyon problemleri için
                                  ortalama kare hata (mean squared error), sınıflandırma için ise çapraz entropi (cross-entropy) kullanılır.
Geri Yayılım (Backpropagation): Hatanın türevleri hesaplanarak, ağdaki her bir ağırlığın hatayı azaltacak şekilde nasıl güncellenmesi gerektiği belirlenir.
Ağırlıkların Güncellenmesi: Gradyan inişi veya benzeri bir optimizasyon algoritması kullanarak, ağırlıklar hatayı azaltacak şekilde güncellenir.
Bu işlem, belirlenen bir iterasyon sayısı boyunca veya modelin performansı tatmin edici bir seviyeye ulaşana kadar tekrar edilir.

Avantajları ve Kullanım Alanları
Esneklik: YSA'lar lineer olmayan ve karmaşık ilişkileri modelleyebilir.
Genelleştirme Kabiliyeti: Doğru şekilde eğitildiğinde, YSA'lar görülmemiş veriler üzerinde iyi tahminler yapabilir.
Uyarlanabilirlik: YSA'lar farklı problemlere ve veri türlerine kolaylıkla adapte edilebilir.


YSA'lar, görüntü ve ses tanıma, metin işleme, robot kontrol sistemleri, tıbbi teşhis ve finansal tahminler gibi çok çeşitli alanlarda kullanılmaktadır.
Ancak, genellikle büyük miktarda eğitim verisi gerektirir ve aşırı uyum (overfitting) gibi problemlere karşı dikkatli bir şekilde düzenlenmeleri gerekir.
YSA'lar ayrıca, modelin nasıl karar verdiğini yorumlamakta zorluklara neden olabilir, bu nedenle "kara kutu" olarak adlandırılan modeller arasındadır.


"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn import model_selection
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor




hit = pd.read_csv("Data Science-ML/Regressions/Hitters.csv")
df = hit.copy()
df = df.dropna()
dms = pd.get_dummies(df[['League', 'Division', 'NewLeague']])
y = df["Salary"]
X_ = df.drop(['Salary', 'League', 'Division', 'NewLeague'], axis=1).astype('float64')
X = pd.concat([X_, dms[['League_N', 'Division_W', 'NewLeague_N']]], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.25,
                                                    random_state=42)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.neural_network import MLPRegressor
mlp_model = MLPRegressor(hidden_layer_sizes=(100,20)).fit(X_train_scaled,y_train)
mlp_model.get_params()

mlp_model.n_layers_
mlp_model.hidden_layer_sizes


#  Tahmin

mlp_model.predict(X_train_scaled)[0:5]
y_pred = mlp_model.predict(X_test_scaled)
np.sqrt(mean_squared_error(y_test,y_pred))
# 447.8142732174795


# Model Tuning

mlp_params = {'alpha': [0.1, 0.01, 0.02, 0.005], 'hidden_layer_sizes': [(20,20),(100,50,150),(300,200,150)],
                'activation': ['relu','logistic']}

mlp_cv_model = GridSearchCV(mlp_model, mlp_params, cv=10)
mlp_cv_model.fit(X_train_scaled,y_train)
mlp_cv_model.best_params_


# activation default olarak 'relu' zaten
mlp_tuned = MLPRegressor(alpha = 0.01, hidden_layer_sizes = (100,50,150))
mlp_tuned.fit(X_train_scaled,y_train)
y_pred = mlp_tuned.predict(X_test_scaled)
np.sqrt(mean_squared_error(y_test,y_pred))
# 359.5811849497426 Şuana kadarki modeller arasında en düşük test hatasını veren model oldu.




