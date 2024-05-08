"""

LightGBM, Microsoft tarafından geliştirilen ve Gradient Boosting Machine (GBM) algoritmasının hızlı ve verimli bir uygulaması olan bir
makine öğrenimi kütüphanesidir. XGBoost'un bir türü olan ve özellikle büyük veri setleri üzerinde çalışırken verimlilik ve hız açısından
avantajlar sağlayan LightGBM, GBM'in optimize edilmiş versiyonlarından biridir. LightGBM, özellikle yüksek boyutlu veri setleri ve çok
sayıda özellik içeren veriler için uygundur.

LightGBM'in bazı önemli özellikleri şunlardır:

Histogram Tabanlı Öğrenme: LightGBM, sürekli özellik değerlerini sabit boyutlu kovalara (bins) yerleştirerek histogram tabanlı bölme yöntemi kullanır.
                           Bu, daha az bellek kullanımı ve daha hızlı eğitim süresi anlamına gelir.
Leaf-wise (Yaprak Temelli) Büyüme: Geleneksel GBM algoritmaları genellikle level-wise (seviye temelli) büyüme kullanırken, LightGBM leaf-wise büyüme
                                   yaklaşımını benimser. Bu, verimliliği artırır ve modelin daha iyi sonuçlar vermesini sağlayabilir, ancak aşırı öğrenme
                                   (overfitting) riski artabilir.
Hızlı ve Ölçeklenebilir: LightGBM, paralel ve GPU öğrenmeyi destekler, bu da büyük veri setleri üzerinde çalışırken bile hızlı eğitim süreleri sağlar.
Kategorik Özelliklerin Otomatik İşlenmesi: LightGBM, kategorik özellikleri otomatik olarak işleyebilir ve one-hot encoding yapılmasına gerek kalmadan
                                           bu özellikleri doğrudan modele dahil edebilir.
Daha Az Hiperparametre Ayarı: LightGBM, diğer GBM algoritmalarına göre daha az hiperparametre ayarı gerektirebilir ve genellikle varsayılan parametrelerle bile iyi sonuçlar verir.
Eksik Değerlerle Başa Çıkma: LightGBM, eksik değerlerle etkili bir şekilde başa çıkabilir ve bu değerleri modelin eğitim sürecine dahil edebilir.

LightGBM, büyük veri setleri üzerinde çalışırken dikkate alınması gereken güçlü bir seçenektir. Ancak, veri setinin büyüklüğüne ve karmaşıklığına göre aşırı
öğrenmeyi önlemek için parametre ayarlamalarına ihtiyaç duyulabilir. LightGBM ayrıca, XGBoost gibi diğer GBM uygulamalarına kıyasla daha az kaynak kullanımı
avantajına sahip olduğundan, özellikle kaynak sınırlamaları olan ortamlarda tercih edilebilir.
"""



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error



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

from lightgbm import LGBMRegressor

# conda install -c conda-forge lightgbm

lgbm = LGBMRegressor()
lgbm_model = lgbm.fit(X_train,y_train)

# TAHMİN
y_pred = lgbm_model.predict(X_test,num_iteration = lgbm_model.best_iteration_)
np.sqrt(mean_squared_error(y_test,y_pred))

# MODEL TUNING
lgbm_model.get_params()
lgbm_grid = {
    'learning_rate' : [0.01,0.1,0.5,1],
    'n_estimators' : [20,40,100,200,500,1000],
    'max_depth' : [1,2,3,4,5,6,7,8],
    'colsample_bytree' : [0.4,0.5,0.6,0.9,1]
}
lgbm = LGBMRegressor()
lgbm_cv_model = GridSearchCV(lgbm, lgbm_grid, cv = 10, n_jobs =-1, verbose = 2)
lgbm_cv_model.fit(X_train,y_train)
lgbm_cv_model.best_params_

lgbm_tuned = LGBMRegressor(learning_rate=0.1, colsample_bytree= 0.5, max_depth=6,n_estimators=20)
lgbm_tuned = lgbm_tuned.fit(X_train,y_train)

y_pred = lgbm_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))

























