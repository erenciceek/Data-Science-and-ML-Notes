"""
Support Vector Classification (SVC), destek vektör makineleri (Support Vector Machines - SVM) konseptini kullanarak sınıflandırma görevleri için geliştirilmiş
bir algoritmadır. SVC, özellikle iki sınıflı (ikili) sınıflandırma problemlerinde etkili bir yöntem olarak bilinir, ancak tek bir SVC modeli kullanarak çok
sınıflı sınıflandırma da yapılabilir.

SVC'nin temel fikri, veri noktaları arasında bir ayrım hiper-düzlemi (hyperplane) bulmaktır. Bu düzlem, farklı sınıflara ait veri noktalarını birbirinden ayırırken,
en büyük marjinal boşluğa (margin) sahip olanı seçmeyi hedefler. Bu maksimum marjinal boşluk, modelin yeni veri noktalarını sınıflandırırken daha iyi bir
genelleştirme yapmasına olanak tanır.

SVC'nin anahtar konseptleri şunlardır:
Hiper-Düzlem: Veri noktalarını iki sınıfa ayırmak için kullanılan n-boyutlu düzlem.
Marjinal Boşluk: Hiper-düzleme en yakın olan veri noktaları arasındaki boşluktur. SVC, bu boşluğu maksimize etmeye çalışır.
Destek Vektörleri: Hiper-düzleme en yakın olan ve marjinal boşluğu belirleyen veri noktalarıdır.

SVC algoritması şu adımları izler:
Model Oluşturma: SVC modeli, verileri en iyi ayıran hiper-düzlemi bulmak için eğitilir.
Özgün Karakteristikler: Veri doğrusal olarak ayrılabilir değilse, SVC kernel trick adı verilen bir yöntemle verileri daha yüksek boyutlu bir uzaya eşler.
Bu sayede lineer olmayan sınırlar bulunabilir.
Optimizasyon: SVC, Quadratic Programming (Kuadratik Programlama) adı verilen bir optimizasyon problemi çözerek en uygun hiper-düzlem parametrelerini bulur.
Tahmin: Model eğitildikten sonra, yeni veri noktaları hiper-düzlem kullanılarak sınıflandırılır.

SVC'nin avantajları şunlardır:
Etkili: İyi ayarlandığında ve uygun kernel seçildiğinde, çeşitli veri setlerinde güçlü sınıflandırma sonuçları üretir.
Genelleştirme Kabiliyeti: Maksimum marj prensibi, modelin genelleştirme kabiliyetini artırır.
Farklı Kernel Seçenekleri: Lineer olmayan sınıflandırma için çeşitli kernel fonksiyonları kullanılabilir (örneğin, RBF, polinomiyel, sigmoid).

Dezavantajları ise şunlardır:
Parametre Seçimi: Kernel tipi ve diğer hiperparametrelerin (C parametresi gibi) ayarlanması zor olabilir.
Ölçeklenebilirlik: Büyük veri setlerinde eğitim süresi uzun olabilir.
Sonuçların Yorumlanması: Bulunan hiper-düzlem ve destek vektörleri genellikle yorumlaması zor olan karmaşık modellerdir.


Özetle, SVC, veri biliminde yaygın olarak kullanılan güçlü ve esnek bir sınıflandırma yöntemidir. Ancak, veri setinin doğasına ve karmaşıklığına göre dikkatli
bir şekilde hiperparametre ayarlaması gerektirir.
"""


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report



diabetes = pd.read_csv("Data Science-ML/Classifications/diabetes.csv")
df = diabetes.copy()
df = df.dropna()
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,
                                                 random_state=42)

# Model ve Tahmin
from sklearn.svm import SVC
svm_model = SVC(kernel ="linear").fit(X_train,y_train)

y_pred = svm_model.predict(X_test)
accuracy_score(y_test,y_pred)

# Model Tuning
svc_params = {"C": np.arange(1,10)}
svc = SVC(kernel="linear")
svc_cv_model = GridSearchCV(svc,svc_params, cv=10, n_jobs=-1,verbose=2)
svc_cv_model.fit(X_train,y_train)
svc_cv_model.best_params_# C = 1


svc = SVC(kernel="linear", C=1)
svc_tuned = svc.fit(X_train,y_train)
y_pred = svc_tuned.predict(X_test)
accuracy_score(y_test,y_pred)





