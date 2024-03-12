# Amaç hata kareler toplamını minimize eden katsayıları bu katsayılara bir ceza uygulayarak bulmaktır.
# Overfitting'e karşı dirençli
# bias'ı yüksek fakat varyansı düşüktür. (Bazen bias'ı yüksek modelleri daha çok tercih ederiz)
# Tüm değişkenler ile model kurar. İlgisiz değişkenleri modelden çıkarmaz, katsayılarını sıfıra yaklaştırır
# lambda(veya bazı kaynaklarda alpha) kritik roldedir. İyi bir değer bulunması önemlidir. Bunun için CV yöntemi kullanılır

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

hit = pd.read_csv("../Hitters.csv")
df = hit.copy()
df = df.dropna()
dms = pd.get_dummies(df[['League', 'Division', 'NewLeague']])
y = df["Salary"]
X_ = df.drop(['Salary','League','Division','NewLeague'], axis=1).astype('float64')
X = pd.concat([X_, dms[['League_N','Division_W','NewLeague_N']]], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=42)

from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt


ridge_model = Ridge(alpha=0.1).fit(X_train,y_train)
ridge_model.coef_

lambdalar = 10**np.linspace(10,-2,100)*0.5
ridge_model = Ridge()
katsayilar = []

for i in lambdalar:
    ridge_model.set_params(alpha = i)
    ridge_model.fit(X_train,y_train)
    katsayilar.append(ridge_model.coef_)


ax = plt.gca()
ax.plot(lambdalar,katsayilar)
ax.set_xscale('log')

plt.xlabel('Lambda(alpha) Değerleri')
plt.ylabel('Katsayılar/Ağırlıklar')
plt.title('Düzenlileştirmenin Bir Fonksiyonu Olarak Ridge Katsayıları')
plt.show()


# TAHMİN

from sklearn.metrics import mean_squared_error

y_pred = ridge_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

# MODEL TUNING
lambdalar = 10**np.linspace(10,-2,100)*0.5
lambdalar[0:5]

from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

scaler = StandardScaler()

# RidgeCV modelini oluştur
ridge_cv = RidgeCV(alphas=lambdalar, scoring="neg_mean_squared_error") #normalize=True parametresi bu sürümde olmadığı için aşağıdaki yönteme başvuruyoruz.


# Bir pipeline oluşturarak normalizasyonu ve modeli birleştir
pipeline = make_pipeline(scaler, ridge_cv)

# Pipeline'ı eğitim verileri ile eğit
pipeline.fit(X_train, y_train)
# En iyi alpha parametresini al
best_alpha = pipeline.named_steps['ridgecv'].alpha_
best_alpha
# videodaki alpha değeri 0.75... , bizimkisi ise çok daha büyük bir değer . Bu nedenle sonuç farklı
ridge_tuned = Ridge(alpha=best_alpha).fit(X_train,y_train)
ridge_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test,ridge_tuned.predict(X_test)))
# videoda 386 çıktı test hatası , bizde 358.208