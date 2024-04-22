"""
Extreme Gradient Boosting (XGBoost), Gradient Boosting algoritmasının yüksek performanslı ve ölçeklenebilir bir uygulamasıdır.
XGBoost, hem sınıflandırma hem de regresyon görevleri için geniş çapta kullanılmakta ve makine öğrenimi yarışmalarında sıkça
tercih edilmektedir. Algoritma, yüksek hız ve performans sağlamak için optimize edilmiş olup, aynı zamanda üzerinde çalıştığı
donanımı verimli kullanacak şekilde tasarlanmıştır.

XGBoost'un ana özellikleri şunlardır:
Düzenlileştirme (Regularization): XGBoost, modelin karmaşıklığını kontrol ederek aşırı uydurma (overfitting) riskini azaltmak
için L1 (Lasso regresyonu) ve L2 (Ridge regresyonu) düzenlileştirme tekniklerini kullanır.
Eğitim Sürecinde Kesme (Pruning): Karar ağaçlarının büyümesini sınırlamak için kesme işlemi uygular, bu da gereksiz dallanmaları
önler ve modelin genelleştirme kabiliyetini artırır.
Hesaplama Verimliliği: Eğitim sürecinde çoklu çekirdek kullanımı ve donanım kaynaklarını etkin bir şekilde kullanma yeteneği
ile bilgisayar kaynaklarından maksimum fayda sağlar.
Ölçeklenebilirlik: Büyük veri setleri üzerinde bile etkili çalışabilen, ölçeklenebilir bir yapıya sahiptir.
Esnek Kayıp Fonksiyonları: Farklı kayıp fonksiyonları tanımlanabilir, bu sayede algoritma çeşitli veri setleri ve farklı problemlere
uyarlanabilir.
Eksik Değerleri Otomatik Olarak İşleme: Eksik veri değerlerini model içerisinde otomatik olarak işleyebilme yeteneğine sahiptir.


XGBoost algoritması aşağıdaki adımları takip eder:
İlk Tahmin: Basit bir model ile başlar ve ilk tahmini üretir.
Hata Hesaplama: İlk tahmin ile gerçek değerler arasındaki farkı hesaplar.
Yeni Model Eğitimi: Bu hataları hedef alarak yeni bir model eğitir.
Modeli Güncelleme: Yeni eğitilen modeli, öğrenme hızı ile ölçeklendirerek mevcut modele ekler.
İterasyon: Yukarıdaki işlemleri belirli bir sayıda iterasyon için veya durdurma kriterine kadar tekrarlar.
Son Model: Tüm iterasyonlardan sonra, elde edilen model topluluğu, verilen sınıflandırma görevini yerine getirecek final model
olarak kullanılır.


XGBoost'un avantajları, yüksek tahmin performansı ve esnekliğinin yanı sıra, çeşitli donanımlarda ve veri setlerinde etkili bir
şekilde çalışabilmesidir. Dezavantaj olarak ise, hiperparametrelerin (öğrenme hızı, ağaç sayısı, derinlik vb.) dikkatli bir şekilde
ayarlanması gerektiğini ve bazen hesaplama maliyetinin yüksek olabileceğini söyleyebiliriz. Bununla birlikte, XGBoost genellikle
varsayılan ayarlarla bile iyi sonuçlar üretebilen bir algoritmadır.
"""


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt


diabetes = pd.read_csv("Data Science-ML/Classifications/diabetes.csv")
df = diabetes.copy()
df = df.dropna()
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)
X = pd.DataFrame(X)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,
                                                 random_state=42)


# Model ve Tahmin
# Fakat burada oluşturulan xgb_model'in default parametreleri videodakinden çok farklı. Bu nedenle farklı skorlar elde ediliyor.
from xgboost import XGBClassifier

xgb_model = XGBClassifier().fit(X_train,y_train)
y_pred = xgb_model.predict(X_test)
accuracy_score(y_test,y_pred)

# Model Tuning
xgb_model

xgb = XGBClassifier()
xgb_params = {
    "n_estimators" : [100,500,1000,2000],
    "subsample" : [0.6,0.8,1.0],
    "max_depth" : [3,4,5,6],
    "learning_rate" : [0.1,0.01,0.02,0.05],
    "min_samples_split" : [2,5,10]
}

xgb_cv = GridSearchCV(xgb,xgb_params,cv = 10, n_jobs=-1 , verbose=2)
xgb_cv_model = xgb_cv.fit(X_train,y_train)

xgb_cv_model.best_params_

# Final Model
xgb = XGBClassifier(learning_rate = 0.01, max_depth = 3, min_samples_split = 2,
                    n_estimators = 100, subsample = 0.8)
xgb_tuned = xgb.fit(X_train,y_train)


y_pred = xgb_tuned.predict(X_test)
accuracy_score(y_test,y_pred)