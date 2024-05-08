"""
Bagging(Breiman, 1996) ile Random Subspace (Ho, 1998) yöntemlerinin birleşimi ile oluşmuştur.
Ağaçlar için gözlemler bootstrap rastgele örnek seçim yöntemi ile değişkenler random subspace yöntemi ile seçilir.



Random Forests, sınıflandırma ve regresyon görevleri için kullanılan popüler ve güçlü bir makine öğrenimi algoritmasıdır. Bir topluluk öğrenme (ensemble learning)
modeli olan Random Forests, birden fazla karar ağacının tahminlerini birleştirerek çalışır. Bu birleştirme işlemi, genellikle her ağacın tahmini üzerinde bir
"oylama" veya ortalama alma işlemi yaparak gerçekleştirilir. Temel fikir, bireysel modellerin çeşitliliğinden güç alarak genel bir modelin hatasını azaltmaktır.


İşte Random Forests algoritmasının temel özellikleri ve avantajları:

Çoklu Karar Ağaçları: Random Forests, rastgele seçilen özellikler kullanarak eğitilen çok sayıda karar ağacından oluşur. Her bir karar ağacı, veri setinin
                      rastgele seçilmiş alt kümeleri üzerinde eğitilir (bootstrap örneklemesi).

Azalan Varyans: Tek karar ağaçları yüksek varyansa sahip olabilir (yani, farklı veri setlerine aşırı duyarlıdırlar), ancak Random Forests bir topluluk olarak
                hareket ederek bu varyansı azaltır. Ağaçlar arasındaki karşılıklı bağımsızlık, topluluğun aşırı uydurma (overfitting) sorununa karşı daha dirençli olmasını sağlar.
Özellik Seçimi: Her bölme noktasında, özelliklerin rastgele bir alt kümesi seçilir ve en iyi bölme, yalnızca bu alt kümedeki özellikler arasından seçilir.
                Bu, özellikler arası korelasyonu azaltır ve modelin her bir özelliği değerlendirme şansını artırır.
Genel Kullanım: Hem sınıflandırma hem de regresyon görevleri için kullanılabilir ve genellikle iyi sonuçlar verir. Bu, veri setindeki gürültüye ve eksik verilere
                karşı dayanıklı olmasıyla da ilgilidir.
Hiperparametreler: Random Forests modelinin performansını etkileyebilecek birkaç önemli hiperparametre vardır. Bunlar arasında ağaç sayısı,
                   maksimum derinlik, min_samples_split ve min_samples_leaf bulunur.
Özellik Önemi: Algoritma, her bir özelliğin modeldeki önemini değerlendirebilir, bu da özellik seçimi ve veri anlayışı için yararlı olabilir.

Random Forests algoritması, stabil ve sağlam olduğu için genellikle çok az ayarlamayla iyi performans gösterir ve birçok farklı veri türü üzerinde etkilidir.
Bununla birlikte, çok büyük veri setlerinde eğitim zamanı uzun olabilir ve oluşturulan model, tek bir karar ağacına göre daha az yorumlanabilir olabilir.


"""


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt




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

from sklearn.ensemble import RandomForestRegressor

# Default parametre değerleri kütüphanenin versiyonlarına göre değişebilir. Örneğin kursta max_features = 'auto' olarak belirtilmiş
# Fakat bu modelde max_features yalnızca integer veya float değerler alabiliyor. Bunlara ek olarak sqrt gibi 2 adet string değer alabilir.
rf_model = RandomForestRegressor(random_state = 42, n_estimators = 10)
rf_model.fit(X_train,y_train)
rf_model.get_params()

# TAHMİN
rf_model.predict(X_test)[0:5]
y_pred = rf_model.predict(X_test)
np.sqrt(mean_squared_error(y_pred,y_test))


# MODEL TUNING
rf_params = {'max_depth' : list(range(1,10)),
             'max_features' : [3,5,10,15],
             'n_estimators' : [100,200,500,1000,2000]}
rf_model = RandomForestRegressor(random_state = 42)
rf_cv_model = GridSearchCV(rf_model, rf_params, cv = 10, n_jobs = -1)
rf_cv_model.fit(X_train,y_train)
rf_cv_model.best_params_

rf_tuned = RandomForestRegressor(max_depth = 8, max_features = 3, n_estimators = 200)
rf_tuned = rf_tuned.fit(X_train,y_train)

y_pred = rf_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_pred,y_test))


# Tune edilmiş RF modelinde değişkenlerin bağımlı değişkene olan etkilerine göre anlamlılıklarını göstermek istiyoruz.
Importance = pd.DataFrame({"Importance" : rf_tuned.feature_importances_*100},
                          index= X_train.columns)

Importance.sort_values(by = "Importance",
                       axis=0 ,
                       ascending=True).plot(kind="barh",color ="r")

plt.xlabel("Değişken Önem Düzeyleri")