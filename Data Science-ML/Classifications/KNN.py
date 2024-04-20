"""
K-En Yakın Komşu (K-Nearest Neighbors ya da KNN), hem sınıflandırma hem de regresyon problemleri için kullanılan, temel ama güçlü bir makine öğrenimi
algoritmasıdır. KNN, veri noktaları arasındaki benzerliğe dayalı çalışan tembel bir öğrenme (lazy learning) ve örnek tabanlı öğrenme (instance-based learning)
yöntemidir. Tembel bir öğrenme algoritması olduğu için, model eğitme aşamasında verileri model olarak doğrudan bellekte tutar ve asıl hesaplamayı tahmin
sırasında yapar.

KNN algoritmasının çalışma prensibi oldukça basittir:
Uzaklık Hesaplama: Tahmin edilmek istenen nokta ile eğitim veri setindeki tüm noktalar arasındaki uzaklık hesaplanır. Uzaklık genellikle Öklid uzaklığı olarak
hesaplanır ancak Manhattan, Minkowski veya özel bir uzaklık fonksiyonu da kullanılabilir.
En Yakın Komşuların Seçimi: Hesaplanan uzaklıklara göre en yakın 'k' adet komşu seçilir. 'k', algoritmanın temel parametresidir ve kullanıcının seçtiği bir değerdir.
Çoğunluk Oyu veya Ortalama: Sınıflandırma için, bu 'k' en yakın komşunun çoğunluk sınıfı, tahmin edilmek istenen noktanın sınıfı olarak atanır. Regresyon için,
komşuların çıktı değerlerinin ortalaması alınır.

KNN algoritmasının avantajları şunlardır:
Anlaşılması ve Uygulanması Kolay: Basit bir mantığa sahip olması nedeniyle yeni başlayanlar için bile kolaylıkla anlaşılabilir ve uygulanabilir.
Parametre Sayısı Az: Temelde sadece bir parametre ayarı gerektirir: komşu sayısı 'k'.
Özellik Ölçeğine Duyarsız Versiyonları Mevcut: Veriler normalize edildiğinde veya uygun bir uzaklık ölçütü seçildiğinde, farklı ölçeklerdeki özelliklerden etkilenmez.

Dezavantajları ise şunlardır:
Ölçeklenebilirlik Sorunu: Büyük veri setleriyle çalışırken, her tahmin için veri setindeki tüm noktalarla karşılaştırma yapmak hesaplama açısından çok maliyetlidir.
Yüksek Boyutlu Veriler için Lanet: Boyutluluk laneti (curse of dimensionality) nedeniyle, çok sayıda özelliği olan veri setlerinde performansı düşebilir.
Dengesiz Veri Setleri: Azınlık sınıfına ait örneklerin komşu olarak seçilme olasılığı düşük olduğunda yanlı tahminler yapabilir.

Sonuç olarak, KNN özellikle küçük ve orta ölçekli veri setleri için uygun olup, karmaşık öğrenme modellerine gerek duyulmayan durumlarda pratik ve etkili bir
çözüm sunar.
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
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn_model = knn.fit(X_train,y_train)

knn_model.get_params()
y_pred = knn_model.predict(X_test)
accuracy_score(y_test,y_pred)
print(classification_report(y_test,y_pred))


# Model Tuning
knn_params = {"n_neighbors" : np.arange(1,50)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn,knn_params,cv = 10)
knn_cv.fit(X_train,y_train)

print("En iyi skor :"  + str(knn_cv.best_score_))
print("En iyi parametreler : " + str(knn_cv.best_params_)) # 11

knn = KNeighborsClassifier(n_neighbors=11)
knn_tuned = knn.fit(X_train,y_train)
knn_tuned.score(X_test,y_test)  #accuracy_score 'a ulaşmanın alternatif bir yolu

y_pred = knn_tuned.predict(X_test)  # bu da klasik yöntemimiz
accuracy_score(y_test,y_pred)



