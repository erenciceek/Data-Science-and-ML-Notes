"""
CatBoost (Categorical Boosting), Yandex tarafından geliştirilen ve özellikle kategorik özelliklerin işlenmesine odaklanmış bir
Gradient Boosting algoritmasıdır. Makine öğreniminde sıkça karşılaşılan kategorik veri türleriyle doğrudan ve etkin bir şekilde
çalışabilmek üzere tasarlanmıştır.

CatBoost'un bazı dikkate değer özellikleri şunlardır:

Kategorik Özelliklerin Otomatik İşlenmesi: CatBoost, kategorik özellikleri one-hot encoding yapmadan doğrudan işleyebilir. Bu,
özellik mühendisliği sürecini basitleştirir ve kullanıcıların kategorik veri üzerinde zaman harcamadan model eğitimine başlamasını sağlar.
Sıralı (Ordered) ve Tekrarlı (Plain) Boosting: CatBoost, hedef sızıntısını (target leakage) ve aşırı öğrenmeyi (overfitting) önlemek
için veri setini rastgele permütasyonlar kullanarak parçalara ayırır ve her bir permütasyon için sıralı bir şekilde model eğitir.
Bu yöntemle, modelin daha önce görmediği veriler üzerinde doğrulama yapılması sağlanır.
Sınırlı Büyüme Algoritması (Symmetric Tree): CatBoost, dengeli ağaçlar oluşturur, yani her seviyede tüm yapraklar ya bölünür ya
da hiçbiri bölünmez. Bu, hızlı tahmin yapılmasını sağlar ve modelin interpretasyonunu kolaylaştırır.
Model Şeffaflığı: CatBoost, özelliklerin model üzerindeki etkilerini açıklamak için özellik önem skorlarını ve SHAP değerlerini
hesaplama yeteneğine sahiptir.
Etkin Kayıp Minimizasyonu: CatBoost, heteroscedastic regresyon ve sınıflandırma gibi farklı problemler için çeşitli kayıp fonksiyonları sunar.
Yüksek Performans: CatBoost, paralel hesaplama ve GPU desteği ile yüksek performans sunar, böylece büyük veri setleri üzerinde hızlı
eğitim ve tahmin yapılabilir.
Robust (Sağlam) Sonuçlar: CatBoost, varsayılan parametreleri ile dahi robust sonuçlar üretebilir, bu yüzden hiperparametre ayarlaması
için çok fazla zaman harcamaya gerek kalmaz.
Python, R, Java ve diğer diller için API Desteği: CatBoost, popüler programlama dilleri için API'ler sunar ve böylece farklı uygulama
senaryolarında kolayca kullanılabilir.

CatBoost, veri bilimcileri ve analistler arasında, özellikle kategorik verilerle çalışırken tercih edilen bir seçenektir.
Bu algoritma, kategorik özelliklerin getirdiği zorlukları ele almak ve bu tür verilere sahip veri setleri üzerinde yüksek
doğrulukta modeller oluşturmak için güçlü bir araçtır.

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
from catboost import CatBoostRegressor

catb = CatBoostRegressor()
catb_model = catb.fit(X_train,y_train)

# TAHMİN
y_pred = catb_model.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))

# MODEL TUNING
catb_grid = {
    'iterations' : [200,500,1000,2000],
    'learning_rate' : [0.01,0.03,0.05,0.1],
    'depth' : [3,4,5,6,7,8],
}

catb = CatBoostRegressor()
catb_cv_model = GridSearchCV(catb, catb_grid, cv = 5 ,n_jobs=-1, verbose=2)
catb_cv_model.fit(X_train,y_train)
catb_cv_model.best_params_

catb_tuned = CatBoostRegressor(depth=5, iterations=1000,learning_rate=0.1)
catb_tuned.fit(X_train,y_train)

y_pred = catb_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))
