# Temeli bootstrap yöntemi ile oluşturulan birden fazla karar ağacının ürettiği tahminlrein bir araya getirilerek değerlendirilmesine dayanır

"""

Bagged Trees, ya da tam adıyla Bagging (Bootstrap Aggregating) yöntemi, karar ağaçlarının varyansını azaltmak ve genel performanslarını iyileştirmek
için kullanılan bir topluluk öğrenme tekniğidir. Bu yöntem, Leo Breiman tarafından 1996 yılında geliştirilmiştir ve aşırı uyum (overfitting)
ile mücadele ederek modellerin genelleştirme kabiliyetini artırmayı hedefler.

Bagging’in Çalışma Prensibi:
Bootstrap Örneklemeleri: Orijinal eğitim veri setinden rastgele, tekrarlı örnekleme ile birçok bootstrap örneği oluşturulur. Her bir örnekleme,
orijinal veri setinin büyüklüğüne eşit sayıda örnek içerir, ancak örnekler rastgele seçildiği için bazıları örneklemede birden fazla kez yer
alabilirken bazıları hiç yer almayabilir.

Karar Ağaçları Oluşturma: Her bir bootstrap örneği üzerinde ayrı ayrı karar ağaçları eğitilir. Bu ağaçlar, genellikle bütün veri seti ve özellikler
üzerinde değil, rastgele seçilmiş alt kümeler üzerinde eğitilir.

Aggregating (Birleştirme): Her bir ağaçtan elde edilen tahminler, bir sınıflandırma problemi için çoğunluk oylaması veya bir regresyon problemi için
alınarak birleştirilir. Bu süreç, topluluğun tahminlerinin daha istikrarlı ve doğru olmasını sağlar.

Bagged Trees'in Özellikleri:
Düşük Varyans: Tek bir karar ağacının aksine, bagging tekniği kullanıldığında ağaçlar arasındaki bağımsızlık, modelin varyansını önemli ölçüde azaltır.
Aşırı Uyuma Karşı Direnç: Ağaçların her birinin eğitim veri setinin farklı alt kümeleri üzerinde eğitilmesi, her bir ağacın veriye aşırı uyum sağlamasını engeller.
Genelleştirme: Bagging, modelin genel veri seti üzerindeki performansını iyileştirerek daha iyi genelleştirme yapmasını sağlar.
Paralel İşleme: Karar ağaçları bağımsız olarak eğitilebilir, bu da bagging yönteminin paralel işlemeye çok uygun olmasını sağlar.

Uygulamaları:
Bagging tekniği, başta karar ağaçları olmak üzere farklı algoritmalarla kullanılabilir. Ancak, karar ağaçları ile kullanıldığında en popüler bagging
uygulaması "Random Forest" algoritmasıdır. Random Forest, her bir karar ağacını yalnızca bir alt özellik kümesi üzerinde eğitir, bu da özellik
uzayının her bölgesini farklı ağaçların öğrenmesine olanak tanır.


Avantajları ve Dezavantajları:

Avantajları:
Modellerin genelleştirme yeteneğini iyileştirir.
Aşırı uyuma karşı oldukça dirençlidir.
Farklı problemlere kolayca uyarlanabilir ve paralel işleme imkanı sunar.

Dezavantajları:
Model yorumlaması zorlaşır, çünkü toplulukta çok sayıda model bir araya gelir.
Tek bir karar ağacına göre daha fazla hesaplama gücü gerektirebilir.
Modelin boyutu artar, depolama ve işleme daha fazla kaynak gerekir.

Sonuç olarak, bagged trees yöntemi, karmaşık ve büyük veri setleri üzerinde güçlü ve güvenilir tahminler yapabilen modeller oluşturmak için kullanılır.

"""



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
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

bag_model = BaggingRegressor(bootstrap_features=True)
bag_model.fit(X_train,y_train)
bag_model.get_params()
# n_estimators parametresi tree sayısını belirtiyor.
bag_model.n_estimators
bag_model.estimators_
bag_model.estimators_samples_
bag_model.estimators_features_
bag_model.estimators_[0].get_params()


# Tahmin
y_pred = bag_model.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))

iki_y_pred = bag_model.estimators_[1].fit(X_train,y_train).predict(X_test)
np.sqrt(mean_squared_error(y_test,iki_y_pred))

yedi_y_pred = bag_model.estimators_[6].fit(X_train,y_train).predict(X_test)
np.sqrt(mean_squared_error(y_test,yedi_y_pred))

# MODEL TUNING
bag_params = {"n_estimators" : range(2,20)}

bag_cv_model = GridSearchCV(bag_model,bag_params,cv=10)
bag_cv_model.fit(X_train,y_train)
bag_cv_model.get_params()
bag_cv_model.best_params_

bag_tuned = BaggingRegressor(n_estimators=14 , random_state=45)
bag_tuned.fit(X_train,y_train)
y_pred = bag_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))
# 346.457987188104  şuana kadarki en iyi değer.
