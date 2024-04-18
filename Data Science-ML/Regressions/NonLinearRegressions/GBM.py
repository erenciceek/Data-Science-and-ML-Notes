"""
Gradient Boosting Machines (GBM), hem sınıflandırma hem de regresyon problemleri için kullanılan başka bir güçlü ve popüler topluluk öğrenme tekniğidir.
GBM, zayıf öğrenicileri (genellikle basit karar ağaçları) aşamalı olarak iyileştirerek güçlü bir model oluşturur. Boosting, zayıf öğrenicileri
sıralı bir şekilde bir araya getirerek, her bir yeni modelin önceki modellerin hatalarını düzeltmeye odaklanmasına dayanır.

GBM'in çalışma prensibi şu adımları takip eder:

İlk Tahmin: Veri seti üzerinde basit bir model (genellikle bir karar ağacı) oluşturulur ve bir tahmin yapılır. Bu, topluluğun ilk öğrenicisi
            olarak kabul edilir.
Hata Türevleri: Modelin yaptığı tahminler ile gerçek değerler arasındaki fark (rezidü) hesaplanır. Bu fark, gerçek değerler ile tahmin edilen
                değerler arasındaki kayıp fonksiyonunun gradyanı (türevi) olarak düşünülebilir.
Yeni Öğrenicilerin Eğitimi: Rezidüler üzerine yeni bir zayıf öğrenici eğitilir. Bu öğrenici, önceki toplam tahmini iyileştirmek için var olan
                            modelin hatalarını düzeltmeye yöneliktir.
Ağırlıklandırma ve Birleştirme: Her bir zayıf öğrenici, önceki adımdaki kayıpları azaltacak şekilde bir ağırlık ile birleştirilir. Yeni öğrenici,
                                toplam modele bir katsayı ile çarpılarak eklenir. Bu katsayı, genellikle öğrenme hızı olarak adlandırılan ve modelin
                                her adımda ne kadar "öğreneceğini" kontrol eden bir hiperparametredir.
İterasyon: Bu süreç, belirli bir sayıda iterasyon için veya bir durdurma kriteri karşılanana kadar devam eder (örneğin, kayıp fonksiyonunda bir
           iyileşme olmadığında).


GBM'in temel avantajları şunlardır:
Yüksek Performans: GBM, birçok standart veri setinde mükemmel performans gösterir ve genellikle makine öğrenimi yarışmalarında yüksek sıralamalar elde eder.
Esneklik: Farklı kayıp fonksiyonları kullanılabilir, bu da GBM'i birçok farklı istatistiksel modelleme problemi için uygun hale getirir.
Özellik Önemi: GBM, özelliklerin modeldeki önemini sıralayabilir, bu da özellik seçimi ve veri yorumlamada yardımcı olur.

Bununla birlikte, GBM'in dezavantajları şunlardır:
Aşırı Uydurma Riski: Eğer düzgün bir şekilde ayarlanmaz ve kontrol edilmezse, özellikle gürültülü veri setlerinde aşırı uydurma (overfitting) olabilir.
Parametre Ayarlaması: GBM, ayarlanması gereken birkaç kritik hiperparametreye sahiptir, bu da modelin başarısı için deneyim ve dikkat gerektirir.
Eğitim Zamanı: GBM modelleri sıralı olarak eğitildiğinden, büyük veri setlerinde eğitim zamanı uzun olabilir ve bazen paralelleştirme için uygun olmayabilir.

XGBoost, LightGBM ve CatBoost gibi GBM'in geliştirilmiş versiyonları, performansı artırma ve eğitim süresini azaltma gibi ek avantajlar sağlayarak bu dezavantajları azaltmaya çalışır.
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


from sklearn.ensemble import GradientBoostingRegressor

gbm_model = GradientBoostingRegressor()
gbm_model.fit(X_train,y_train)
gbm_model.get_params()

# TAHMİN
y_pred = gbm_model.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))

# MODEL TUNING

gbm_params = {'learning_rate' : [0.001,0.01,0.1,0.2],
              'max_depth' : [3,5,8,50,100],
              'n_estimators': [200,500,1000,2000],
              'subsample' : [1,0.5,0.75],
}

gbm = GradientBoostingRegressor()
gbm_cv_model = GridSearchCV(gbm, gbm_params, cv=10, n_jobs = -1, verbose = 2)
gbm_cv_model.fit(X_train,y_train)
gbm_cv_model.best_params_

gbm_tuned = GradientBoostingRegressor(learning_rate=0.1, max_depth=3, n_estimators=500, subsample=0.75)
gbm_tuned = gbm_tuned.fit(X_train,y_train)
y_pred = gbm_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))




# Tune edilmiş GBM modelinde değişkenlerin bağımlı değişkene olan etkilerine göre anlamlılıklarını göstermek istiyoruz.
Importance = pd.DataFrame({"Importance" : gbm_tuned.feature_importances_*100},
                          index= X_train.columns)

Importance.sort_values(by = "Importance",
                       axis=0 ,
                       ascending=True).plot(kind="barh",color ="r")

plt.xlabel("Değişken Önem Düzeyleri")

