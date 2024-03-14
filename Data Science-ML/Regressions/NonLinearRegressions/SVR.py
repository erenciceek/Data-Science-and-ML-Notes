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
