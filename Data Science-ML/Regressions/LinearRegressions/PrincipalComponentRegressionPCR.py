# Değişkenlere boyut indirgeme uygulandıktan sonra çıkan bileşenlere regresyon modeli kurulması fikrine dayanır

import pandas as pd

hit = pd.read_csv("../Hitters.csv")
df = hit.copy()
df.dropna(inplace=True)

df.info()
df.describe().T


dms = pd.get_dummies(df[['League','Division','NewLeague']])
dms.head()

for column in dms.columns:
    if ("League" in column) or ("Division" in column) or ("NewLeague" in column):
        dms[column] = dms[column].astype(int)

dms.head()

y = df["Salary"]
X_ = df.drop(["Salary","League","Division","NewLeague"], axis=1).astype("float64")
X_.head()

X = pd.concat([X_,dms[["League_N","Division_W","NewLeague_N"]]],axis=1)
X.head()

from sklearn.model_selection import train_test_split, cross_val_score,cross_val_predict

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, random_state =42)

print("X_train", X_train.shape)
print("y_train",y_train.shape)
print("X_test",X_test.shape)
print("y_test",y_test.shape)

training = df.copy()
print("training", training.shape)

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

pca = PCA()
X_reduced_train = pca.fit_transform(scale(X_train))
X_reduced_train[0:1,:]

np.cumsum(np.round(pca.explained_variance_ratio_, decimals = 4)*100)[0:10]

# Bunun sonucunda her değişkenin varyansı açıklayabilme oranı sırasıyla kümülatif olarak belirtiliyor
# 1. bileşenin veri setinde bulunan toplam varyansın %38ini açıkladığı görülüyor
# 2. bileşenin 1. bileşen ile birlikte toplam varyansın %59.88ini açıkladığı görülüyor.
# Bu şekilde çıktıdaki arrayde bileşen sayısı arttıkça toplam varyansı açıklama oranının giderek arttığı gözükmekte.
# Amaç veri setini daha az sayıda bileşene indirgemek.

lm = LinearRegression()
pcr_model = lm.fit(X_reduced_train,y_train)
pcr_model.intercept_
pcr_model.coef_

# TAHMİN
train_y_pred = pcr_model.predict(X_reduced_train)
train_y_pred[0:5]

np.sqrt(mean_squared_error(y_train,train_y_pred))
df["Salary"].mean()
r2_score(y_train,train_y_pred)

pca2 = PCA()
X_reduced_test = pca2.fit_transform(scale(X_test))

test_y_pred = pcr_model.predict(X_reduced_test)
np.sqrt(mean_squared_error(y_test,test_y_pred))



#  MODEL TUNING / Model Doğrulama

# bileşen sayısı ile şuana kadar oynamadık, veri setindeki her bileşen hataları farklı oranda etkileyebiliyor
# en optimal sayıda bileşeni bulmak için :

from sklearn import model_selection
cv_10 = model_selection.KFold(n_splits = 10, shuffle= True, random_state=1)
lm = LinearRegression()
RMSE = []

# Buradaki kod bloğu her bir bileşen sayısı için k katlı cross validation uygulayıp
# hata değerlerini elde edip daha sonra bize hangi bileşen sayısının daha az hata oluşturduğu bilgisini sunacak.
for i in np.arange(1, X_reduced_train.shape[1] + 1):
    score = np.sqrt(-1 * model_selection.cross_val_score(lm,
                                                         X_reduced_train[:, :i],
                                                         y_train.to_numpy(),
                                                         cv=cv_10,
                                                         scoring='neg_mean_squared_error').mean())
    RMSE.append(score)


import matplotlib.pyplot as plt

plt.plot(RMSE, '-v')
plt.xlabel('Bileşen Sayısı')
plt.ylabel('RMSE')
plt.title('Maaş Tahmin Modeli İçin PCR Model Tuning')
plt.show()
# optimum değeri grafiğe göre 6 olarak belirledik.

lm = LinearRegression()
pcr_model = lm.fit(X_reduced_train[:,0:6],y_train)
y_pred_train = pcr_model.predict(X_reduced_train[:,0:6])
print(np.sqrt(mean_squared_error(y_train,y_pred_train)))

y_pred_test = pcr_model.predict(X_reduced_test[:,0:6])
print(np.sqrt(mean_squared_error(y_test,y_pred_test)))

# Öncelikle her zaman ilkel bir test hatası ve train hatası hesaplıyoruz,
# Bu elde ettiğimiz test hatası ve train hatasını daha doğru değerlendirebilmenin yolu
# cross validation yöntemi ile bunları incelemektir.
# Tune ettiğimiz bir model demek model için uygun olan hiperparametre değerini bulduğumuz anlamına gelir
# Bu en uygun parametreyi bulduktan sonra train seti üzerinden final modelini oluşturuyoruz.
# Bu final modelleri üzerinden de test hatalarını hesaplayacağız.
#
