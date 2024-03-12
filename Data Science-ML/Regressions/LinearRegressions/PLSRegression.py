# Değişkenlerin daha az sayıda ve aralarında çoklu doğrusal bağlantı problemi
# olmayan bileşenlere indirgenip regresyon modeli kurulması fikrine dayanır.

# PLS NIPALS'in özel bir halidir, iteratif olarak bağımlı değişken ile yüksek korelasyona sahip
# değişenler arasındaki gizil (latent) ilişkiyi bulmaya çalışır.

# PCR'da doğrusal kombinasyonlar yani bileşenler bağımsız değişken uzayındaki değişkenliği
# maksimum şekilde özetleyecek şekilde oluşturulur.
# Bu durum bağımlı değişkeni açıklama yeteneği olmamasına sebep olmakta.

# PLS'te ise bileşenler bağımlı değişken ile olan kovaryansı maksimum şekilde özetleyecek
# şekilde oluşturulur.

# Değişkenler atılmak istenmiyorsa ve açıklanabilirlik aranıyorsa : PLS

# İki yönteminde bir tunning parametresi vardır o da bileşen sayısıdır.


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

hit = pd.read_csv("../Hitters.csv")
df = hit.copy()
df.head()
df = df.dropna()
dms = pd.get_dummies(df[['League', 'Division', 'NewLeague']])
dms.head()
y = df["Salary"]
X_ = df.drop(['Salary','League','Division','NewLeague'], axis=1).astype('float64')
X = pd.concat([X_, dms[['League_N','Division_W','NewLeague_N']]], axis=1)
X_.head()

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=42)

from sklearn.cross_decomposition import PLSRegression, PLSSVD
pls_model = PLSRegression(n_components=2).fit(X_train,y_train)  # default olarak denendiğinde n_components girmeye gerek yok.
pls_model.coef_


# TAHMİN
pls_model.predict(X_train)[0:10]
y_pred = pls_model.predict(X_train)
np.sqrt(mean_squared_error(y_train,y_pred))
r2_score(y_train,y_pred)

y_pred_test = pls_model.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred_test))


# MODEL TUNING
import matplotlib.pyplot as plt
from sklearn import model_selection

cv_10 = model_selection.KFold(n_splits=10, shuffle=True, random_state=1)

# Hata hesaplamak için döngü
RMSE = []

for i in np.arange(1, X_train.shape[1] + 1):
    pls = PLSRegression(n_components=i)
    score = np.sqrt(-1*cross_val_score(pls, X_train, y_train, cv=cv_10, scoring='neg_mean_squared_error').mean())
    RMSE.append(score)

# Sonuçların Görselleştirilmesi
plt.plot(np.arange(1, X_train.shape[1] + 1), np.array(RMSE), '-v', c = "r")
plt.xlabel('Bileşen Sayısı')
plt.ylabel('RMSE')
plt.title('Salary')
plt.show()

# grafiğe göre optimum bileşen sayımız 2 olmalı.
# n_components = 2

pls_model = PLSRegression(n_components=2).fit(X_train,y_train)
y_pred = pls_model.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))











# Aralarındaki Temel Farklar:
# Bileşen Seçimi: PLS, bağımlı değişkenle ilişkili varyansı dikkate alarak bileşenleri seçerken, PCR sadece bağımsız değişkenlerin varyansına bakar.
# Model Tahmini: PLS, tahmin edilecek değişkeni direkt olarak hedef alırken, PCR önce veriyi indirger ve sonra bir regresyon modeli kurar.
# Bileşen Sayısı: PLS genellikle tahmin için daha az bileşen kullanır, çünkü her bir bileşen özellikle tahminle ilgili bilgi içerir;
# PCR ise daha fazla bileşen kullanabilir çünkü sadece varyansı maksimize etmeye çalışır.
# Veri Setinin Yapısı: PLS, özellikle çoklu doğrusal bağlantı durumlarında ve bağımlı değişkenin yapısının da önemli olduğu durumlarda daha uygundur. PCR ise,
# bağımsız değişkenlerin çok boyutlu yapısını anlamak ve indirgemek amacıyla kullanılır.