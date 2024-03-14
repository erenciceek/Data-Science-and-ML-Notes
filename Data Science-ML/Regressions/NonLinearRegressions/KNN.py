# Doğrusal Olmayan Regresyon Modelleri :

"""
1- K-En yakın Komşu (KNN)
2- Destek Vektör Regresyonu(SVR)
3- Çok Katmanlı Algılayıcılar(ANN)
4- Classification and Regression Tress(CART)
5- Bagging(Bootstrap Aggregation)
6- Random Forests (RF)
7- Gradient Boosting Machines (GBM)
8- Extreme Gradient Boosting (XGBoost)
9- LightGBM
10- CatBoost

"""

"""
KNN (K-En Yakın Komşu) regresyonu, hem sınıflandırma hem de regresyon problemlerinde kullanılan basit, ancak güçlü bir makine 
öğrenmesi algoritmasıdır. KNN regresyonunda, bir noktanın değeri, ona en yakın k komşunun değerlerinin basit bir ortalaması alınarak 
tahmin edilir. Bu yöntem, veriler arasındaki yerel ilişkileri yakalamak için özellikle yararlıdır ve modelin eğitimi esnasında 
varsayılan bir fonksiyonel form gerektirmez.

KNN Regresyonunun Çalışma Prensibi
Komşu Sayısının Belirlenmesi (k): KNN algoritmasının en önemli parametresi, komşu sayısı olan k'dır. 
k, bir noktanın değerini tahmin ederken dikkate alınacak en yakın komşu sayısını belirler.

Mesafe Ölçütü: Her bir veri noktası arasındaki mesafe hesaplanır. Bu mesafe genellikle Öklid mesafesi olarak hesaplanır, ancak diğer mesafe ölçütleri de kullanılabilir.

En Yakın k Komşunun Bulunması: Tahmin edilecek noktanın, tüm eğitim seti içindeki noktalara olan mesafeleri hesaplanır ve en yakın k komşu seçilir.
Tahmin: Seçilen k komşunun çıktı değerlerinin (regresyon durumunda genellikle aritmetik ortalaması) hesaplanmasıyla tahmin yapılır.

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

import os
os.getcwd()
from warnings import filterwarnings
filterwarnings('ignore')

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



knn_model = KNeighborsRegressor().fit(X_train,y_train)
knn_model.get_params()  # modelin default parametlerini görmek için
knn_model.n_neighbors
knn_model.effective_metric_

# Tahmin
y_pred = knn_model.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))

# Farklı k değerleri için hata değerleri hesaplama
RMSE = []

for k in range(10):
    k = k+1
    knn_model = KNeighborsRegressor(n_neighbors = k).fit(X_train, y_train)
    y_pred = knn_model.predict(X_train)
    current_rmse = np.sqrt(mean_squared_error(y_train,y_pred))
    RMSE.append(current_rmse)
    print("k =" , k , "için RMSE değeri: ", current_rmse)

# Model Tuning

"""
GridSearchCV, scikit-learn kütüphanesinde bulunan, belirli bir model için en iyi hiperparametrelerin kombinasyonunu sistemli bir şekilde aramak
ve bulmak için kullanılan güçlü bir araçtır. Çapraz doğrulama (Cross-Validation, CV) yöntemini kullanarak, verilen hiperparametrelerin 
tüm olası kombinasyonlarını deneyerek modelin performansını değerlendirir ve en iyi sonucu veren hiperparametreleri seçer.
"""


knn_params = {'n_neighbors' : np.arange(1,30,1)}
knn = KNeighborsRegressor()
knn_cv_model = GridSearchCV(knn,knn_params,cv=10)
knn_cv_model.fit(X_train,y_train)

knn_cv_model.best_params_["n_neighbors"]
# optimum parametre değerimiz 8

# validasyon yapılmış ve yapılmamış hallerini kıyaslıyoruz.
RMSE = []
RMSE_CV = []

for k in range(10):
    k = k+1
    knn_model = KNeighborsRegressor(n_neighbors = k).fit(X_train, y_train)
    y_pred = knn_model.predict(X_train)
    rmse = np.sqrt(mean_squared_error(y_train,y_pred))
    rmse_cv = np.sqrt(-1*cross_val_score(knn_model, X_train, y_train, cv=10,
                                         scoring = "neg_mean_squared_error").mean())
    RMSE.append(rmse)
    RMSE_CV.append(rmse_cv)
    print("k =" , k , "için RMSE değeri: ", rmse, "RMSE_CV değeri: ", rmse_cv )


#  Son modeli kurma
knn_tuned = KNeighborsRegressor(n_neighbors=knn_cv_model.best_params_["n_neighbors"])
knn_tuned.fit(X_train,y_train)
y_pred = knn_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))




















