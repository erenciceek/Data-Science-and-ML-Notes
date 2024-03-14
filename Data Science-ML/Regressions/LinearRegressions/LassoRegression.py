# Amaç hata kareler toplamını minimize eden katsayıları bu katsayılara bir ceza uygulayarak bulmaktır.
# Ridge Regresyona göre tek farkı λ değeri katsayıları 0 yapabilir. Yani bazı bağımsız değişkenleri denklemden çıkarabilir.
"""
Lasso regresyonu (Least Absolute Shrinkage and Selection Operator), hem değişken seçimi yapabilen hem de
    regresyon katsayılarını küçültebilen bir lineer regresyon türüdür. Ridge regresyonuna benzer şekilde,
    Lasso da katsayılara bir ceza uygular, ancak bu ceza terimi L1 normunu (katsayıların mutlak değerlerinin toplamı)
    kullanır. Lasso'nun temel amacı, modelin karmaşıklığını azaltarak aşırı uyumu (overfitting) önlemek
    ve modelin genelleştirme kabiliyetini artırmaktır.

Değişken Seçimi: Lasso, bazı katsayıları tam olarak sıfıra indirebilir. Bu, modeldeki önemsiz değişkenlerin otomatik
olarak çıkarılmasına olanak tanır, böylece daha yalın ve yorumlanabilir bir model elde edilir.
Aşırı Uyumu Önleme: Lasso, modelin eğitim verilerine fazla uyum sağlamasını engelleyerek,
modelin yeni verilere genelleştirme kabiliyetini artırır.
Parametre Tuning:
λ parametresi, modelin ne kadar cezalandırılacağını belirler.
λ değeri arttıkça, daha fazla katsayı sıfıra indirgenir. Bu parametre genellikle "Cross Validation" yoluyla seçilir.
Yüksek Boyutlu Veriler: Lasso, değişken sayısının gözlem sayısından fazla olduğu durumlarda da etkili bir şekilde çalışabilir.

"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score

import os
os.getcwd()

# VERİNİN HAZIRLANIŞI VE TRAIN,TEST DİZİLERİNİN OLUŞTURULMASI
hit = pd.read_csv("Data Science-ML/Regressions/Hitters.csv")
df = hit.copy()
df = df.dropna()
df.head()
dms = pd.get_dummies(df[['League', 'Division', 'NewLeague']])
y = df["Salary"]
X_ = df.drop(['Salary','League','Division','NewLeague'], axis=1).astype('float64')
X = pd.concat([X_, dms[['League_N','Division_W','NewLeague_N']]], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=42)

from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

lasso_model = Lasso(alpha=0.1).fit(X_train,y_train)
lasso_model.coef_


lasso = Lasso()
lambdalar = 10**np.linspace(10,-2,100)*0.5
katsayilar = []

for i in lambdalar:
    lasso.set_params(alpha=i)
    lasso.fit(X_train,y_train)
    katsayilar.append(lasso.coef_)

ax = plt.gca()
ax.plot(lambdalar*2,katsayilar)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')

# TAHMİN
y_pred = lasso_model.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))

# MODEL TUNING
from sklearn.linear_model import LassoCV
lasso_cv_model = LassoCV(alphas=None , cv = 10, max_iter = 10000)  # Normalize parametresi bunda da yok.  #cross validation ile en uygun lambda seçiliyor.
lasso_cv_model.fit(X_train,y_train)
lasso_cv_model.alpha_   #videoda 0.3940.. gibi bir değer çıkıyor.

lasso_tuned = Lasso(alpha = 0.3940612)
lasso_tuned.fit(X_train,y_train)
y_pred_new = lasso_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred_new))

# 356 çıkıyor, Ridge Regresyonda 386 gibi bir değer vardı. Yani bu model bir tık daha başarılı oldu.