"""
Destek Vektör Regresyonu (Support Vector Regression, SVR), destek vektör makineleri (SVM) temelinde geliştirilen ve
hem lineer hem de non-lineer regresyon problemlerini çözmek için kullanılan güçlü bir makine öğrenmesi modelidir.
SVR, veri noktalarını en iyi şekilde ayıran ve belirli bir marj içinde kalan maksimum sayıda veri noktasını içeren
bir hiper düzlem (veya eğri) bulmaya çalışır. Bu yöntem, veriler arasındaki ilişkiyi modellemek için esnek bir yaklaşım
sunar ve aşırı uyuma (overfitting) karşı iyi bir direnç gösterir.


SVR, finansal zaman serisi tahmini, enerji tüketimi tahmini, iklim değişikliği ile ilgili tahminler ve daha pek çok alanda
başarıyla kullanılmıştır. SVR modelinin performansı, seçilen kernel tipi ve hiperparametrelerin (C, ϵ, kernel parametreleri)
doğru ayarlanmasına büyük ölçüde bağlıdır. Bu hiperparametrelerin optimizasyonu için çapraz doğrulama ve GridSearchCV
gibi teknikler yaygın olarak kullanılır.

SVR'ın Temel Prensipleri
Epsilon Marjı (ϵ): SVR, belirlenen bir hata marjı (ϵ) içindeki tahminleri hata olarak kabul etmez. Bu, modelin hafif tahmin
hatalarını göz ardı etmesine ve yalnızca önemli hatalara odaklanmasına olanak tanır.

Cezalandırma Parametresi (C): C parametresi, modelin hata marjı dışında kalan noktalara ne kadar ceza uygulayacağını belirler.
C değeri arttıkça model daha fazla hata noktasını düzeltmeye çalışır, bu da modelin aşırı uyuma meyilli olmasına yol açabilir.

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

# Amaç bir marjin aralığına maksimum noktayı en küçük hata ile alabilecek şekilde doğru ya da eğriyi belirlemektir.

X_train = pd.DataFrame(X_train["Hits"])
X_test = pd.DataFrame(X_test["Hits"])


# hem linear hem non-linear regresyon uygulayacağız.

from sklearn.svm import SVR
svr_model = SVR(kernel="linear",gamma='auto').fit(X_train,y_train)
svr_model.predict(X_train)[0:5]

print ("y = {0} + {1} x".format(svr_model.intercept_[0],
                                svr_model.coef_[0][0]))

X_train["Hits"][0:1]

-48.69756097561513 + 91*4.969512195122093
# 403.5280487804953 sonucu geliyor ve gerçek sonucumuz da bu.
"""
 # eğer doğru çıkmazsa :
bir SVR modelinin coef_ ve intercept_ özelliklerini kullanarak bir tahmin yapmaya çalışıyorsunuz ve bunu doğrudan bir lineer denklem gibi
hesaplıyorsunuz. Ancak SVR modeli, tahminleri hesaplarken tüm destek vektörlerini ve karşılık gelen çift kat sayıları dikkate alır. 
Bu, modelin tahminlerini hesaplarken sadece katsayıları ve kesme terimini kullanmanızın doğru olmayacağı anlamına gelir.
"""
y_pred = svr_model.predict(X_train)
plt.scatter(X_train,y_train)
plt.plot(X_train,y_pred)
plt.show()


# Linear Regression
from sklearn.linear_model import LinearRegression
lm_model = LinearRegression().fit(X_train,y_train)
lm_pred = lm_model.predict(X_train)
print("y = {0} + {1} x".format(lm_model.intercept_, lm_model.coef_[0]))
-8.814095480334345 + 5.172456135470686*91
# 461.87941284749803 değeri çıkıyor sonuç olarak.

# iki modeli birlikte gözlemlemek için
plt.scatter(X_train, y_train, alpha=0.5, s=23)
plt.plot(X_train, lm_pred, 'g')  # linear regression tahmin değerleri
plt.plot(X_train, y_pred, color='r')  # svr tahmin değerleri

plt.xlabel("Atış Sayısı(Hits)")
plt.ylabel("Maaş (Salary)")
plt.plot()


# Tahmin
print ("y = {0} + {1} x".format(svr_model.intercept_[0], svr_model.coef_[0][0]))
svr_model.predict([[91]])
y_pred = svr_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

# Model Tuning

svr_params = {"C": np.arange(0.1,2,0.1)}
svr_cv_model = GridSearchCV(svr_model, svr_params,cv=10).fit(X_train,y_train)

pd.Series(svr_cv_model.best_params_)[0]
svr_tuned = SVR(kernel="linear",gamma="auto",C=pd.Series(svr_cv_model.best_params_)[0]).fit(X_train,y_train)

y_pred = svr_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))


# Şimdiye kadar dataframe'deki yalnızca bir değişkeni ele alıp modeli eğitip tahminde bulunmuştuk.
# Şimdi işe bütünüyle deneyeceğiz.



# NON-LINEAR SVR

np.random.seed(3)

x_sim = np.random.uniform(2, 10, 145)
y_sim = np.sin(x_sim) + np.random.normal(0, 0.4, 145)

x_outliers = np.arange(2.5, 5, 0.5)
y_outliers = -5*np.ones(5)

x_sim_idx = np.argsort(np.concatenate([x_sim, x_outliers]))
x_sim = np.concatenate([x_sim, x_outliers])[x_sim_idx]
y_sim = np.concatenate([y_sim, y_outliers])[x_sim_idx]



from sklearn.linear_model import LinearRegression
ols = LinearRegression()
ols.fit(np.sin(x_sim[:, np.newaxis]), y_sim)
ols_pred = ols.predict(np.sin(x_sim[:, np.newaxis]))

from sklearn.svm import SVR
eps = 0.1
svr = SVR(kernel = 'rbf', epsilon = eps)
svr.fit(x_sim[:, np.newaxis], y_sim)
svr_pred = svr.predict(x_sim[:, np.newaxis])


plt.scatter(x_sim, y_sim, alpha=0.5, s=26)
plt_ols, = plt.plot(x_sim, ols_pred, 'g')
plt_svr, = plt.plot(x_sim, svr_pred, color='r')
plt.xlabel("Bağımsız Değişken")
plt.ylabel("Bağımlı Değişken")
plt.ylim(-5.2, 2.2)
plt.legend([plt_ols, plt_svr], ['EKK', 'SVR'], loc = 4);
plt.show()


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

svr_rbf = SVR(kernel = "rbf",gamma="auto").fit(X_train,y_train)
y_pred = svr_rbf.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

# MODEL TUNING
svr_params = {"C": [0.01, 0.1,0.4,5,10,20,30,40,50]}
svr_cv_model = GridSearchCV(svr_rbf,svr_params, cv = 10)
svr_cv_model.fit(X_train, y_train)

pd.Series(svr_cv_model.best_params_)[0]

# final modelimiz
svr_tuned = SVR(kernel = "rbf",gamma="auto", C = pd.Series(svr_cv_model.best_params_)[0]).fit(X_train, y_train)
y_pred = svr_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))
# 472.20974524750574  çıktı , iyi bir değer değil. Parametre tercihlerinden kaynaklı olabilir.








