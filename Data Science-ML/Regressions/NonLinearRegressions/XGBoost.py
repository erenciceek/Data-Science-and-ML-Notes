"""
XGBoost, GBM'in hız ve tahmin performansını arttırmak üzere optimize edilmiş; ölçeklenebilir ve farklı platformlara entegre edilebilir halidir.

Extreme Gradient Boosting (XGBoost), Gradient Boosting'in ölçeklenebilir ve optimize edilmiş bir uygulamasıdır. XGBoost, hem hız hem de model
performansında geliştirmeler sunar. Tianqi Chen ve Carlos Guestrin tarafından geliştirilen bu algoritma, veri bilimi yarışmalarında ve endüstriyel
uygulamalarda oldukça popülerdir.


XGBoost'un ana özelliklerinden ve avantajlarından bazıları şunlardır:

Düzenlileştirme (Regularization): XGBoost, hem L1 (lasso regresyonu) hem de L2 (ridge regresyonu) düzenlileştirmesini içerir. Bu, aşırı uydurmanın
                                  (overfitting) önlenmesine yardımcı olur ve modelin genelleştirme kabiliyetini artırır.
Ağaç Kesme (Pruning): XGBoost, karar ağaçlarının büyümesini, kayıp fonksiyonuna göre negatif kazanç veren dalları kırparak kontrol eder. Bu,
                      gereksiz karmaşıklıkların önlenmesine ve hızlı model eğitimine olanak tanır.
Esnek Kayıp Fonksiyonları: XGBoost, özelleştirilebilir kayıp fonksiyonlarına izin verir ve bu nedenle çeşitli farklı tahmin problemlerine uygulanabilir.
Eksik Değerlerle Başa Çıkma: XGBoost, eksik verileri işleyebilir ve bir özelliğin en iyi bölünmesini hesaplarken eksik değerleri göz ardı edebilir veya
                             eksik değerleri öğrenme sürecine dahil edebilir.
Ağaç Yapılarını Öğrenme: XGBoost, veri setindeki özelliklerin etkileşimlerini daha iyi yakalayabilen ağaç yapısını öğrenmek için bir çerçeve sunar.
Yüksek Performans ve Hız: XGBoost, paralel işlemeyi ve ağaç yükseltmeyi optimize ederek, diğer gradient boosting algoritmalarına göre çok daha hızlı
                          eğitim süreleri sunar.
Çapraz Platform Desteği ve Ölçeklenebilirlik: XGBoost, çoklu işletim sistemleri üzerinde çalışabilir ve Hadoop, Spark ve Flink gibi büyük veri işlem
                                              platformları ile entegrasyonu destekler.
Etkili Özellik Önemi Hesaplaması: XGBoost, her özelliğin modelin tahmin performansına katkısını değerlendirebilir, bu da özellik seçimi ve model yorumlaması
                                  için faydalıdır.


Bu avantajlar, XGBoost'u karmaşık veri bilimi problemleri için birçok durumda tercih edilen bir araç haline getirmiştir. Bununla birlikte, XGBoost'un da
öğrenme hızı gibi dikkatle seçilmesi gereken hiperparametreleri vardır ve modelin başarısını en üst düzeye çıkarmak için deneyimli bir ayar gerekebilir.

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


import xgboost as xgb
# xgboost dökümanlarına göre bu kütüphanede oluşturulmuş olan veri yapılarının daha iyi sonuçlar verebilceği aktarılmış. Bu nedenle bu şekilde bir
# dönüşüm yapılabilir. Fakat biz modelimizde önceki örneklerde kullanmış olduğumuz veri yapılarını kullanacağız.
DM_train = xgb.DMatrix(data = X_train, label = y_train)
DM_test = xgb.DMatrix(data = X_test, label = y_test)

from xgboost import XGBRegressor
xgb_model = XGBRegressor().fit(X_train,y_train)

# TAHMİN
y_pred = xgb_model.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))


# MODEL TUNING

xgb_grid = {'colsample_bytree' : [0.4,0.5,0.6,0.9,1],
            'n_estimators' : [100,200,500,1000],
            'max_depth' : [2,3,4,5,6],
            'learning_rate' : [0.1,0.01,0.5]}

xgb = XGBRegressor()
xgb_cv = GridSearchCV(xgb, param_grid = xgb_grid, cv = 10, n_jobs = -1, verbose = 2)
xgb_cv.fit(X_train,y_train)
xgb_cv.best_params_

xgb_tuned = XGBRegressor(colsample_bytree = 0.4,
                         learning_rate = 0.1,
                         max_depth = 6,
                         n_estimators = 100)
xgb_tuned = xgb_tuned.fit(X_train,y_train)
y_pred = xgb_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))