"""
CatBoost (Categorical Boosting), kategorik verileri doğrudan işleyebilen, yüksek performanslı ve hızlı bir gradyan boosting
kütüphanesidir. Yandex tarafından geliştirilmiş ve hem sınıflandırma (classification) hem de regresyon (regression) problemleri
için kullanılabilir. CatBoost, özellikle kategorik verilerin bulunduğu veri kümelerinde diğer popüler gradyan boosting
kütüphanelerine (örneğin XGBoost, LightGBM) göre avantajlar sunar.

Özellikleri
Kategorik Veri Desteği: CatBoost, kategorik özelliklerle doğrudan çalışabilme yeteneği ile öne çıkar. Diğer birçok modelin
aksine, kategorik verileri önceden işleme veya dönüştürme gerektirmez. Bu, modelin veriler üzerinde daha etkili bir şekilde
öğrenmesini ve kullanımı daha kolay hale getirir.
Hız ve Ölçeklenebilirlik: CatBoost, CPU ve GPU desteği sayesinde yüksek hızda eğitim ve tahmin yapabilir. Büyük veri kümeleri
üzerinde bile etkili çalışabilir.
Düşük Aşırı Uydurma Riski: CatBoost, özel düzenleme teknikleri kullanarak aşırı uydurmayı (overfitting) önleme konusunda iyidir.
Özellikle Ordered Boosting ve bayraklı sınıflandırma gibi teknikler bu konuda yardımcı olur.
Esneklik: Farklı kayıp fonksiyonları ve özelleştirilmiş metrikler kullanabilir, böylece farklı türdeki problemlere kolayca uyarlanabilir.


Temel Kavramlar
Ordered Boosting: CatBoost'un aşırı uydurmayı azaltma yöntemidir. Eğitim veri setini rastgele alt kümelerine ayırır ve modeli
bu alt kümeler üzerinde sırayla eğitir. Bu yaklaşım, modelin kendi tahminlerine aşırı uyum sağlamasını önler.
Bayraklı Sınıflandırma (Target Statistics): Kategorik özelliklerin işlenmesi için kullanılan bir yöntemdir. Bu teknik,
kategorik değişkenlerin değerlerini hedef değişkenin istatistiklerine dönüştürür.

CatBoost, kullanım kolaylığı, kategorik verilerle doğrudan çalışabilme yeteneği ve aşırı uydurmaya karşı direnç gibi özellikleri
sayesinde, çeşitli sınıflandırma ve regresyon problemleri için güçlü ve etkili bir araçtır.

"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt


diabetes = pd.read_csv("Data Science-ML/Classifications/diabetes.csv")
df = diabetes.copy()
df = df.dropna()
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)
X = pd.DataFrame(X)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,
                                                 random_state=42)

# Model ve Tahmin
from catboost import CatBoostClassifier
cat_model = CatBoostClassifier().fit(X_train,y_train)

y_pred = cat_model.predict(X_test)
accuracy_score(y_test,y_pred)


# Model Tuning
catb_params = {"iterations" : [200,500],
               "learning_rate" : [0.01,0.05,0.1],
               "depth" : [3,5,8]}
catb = CatBoostClassifier()
catb_cv_model = GridSearchCV(catb,catb_params,cv=5,n_jobs=-1,verbose=2)
catb_cv_model.fit(X_train,y_train)
catb_cv_model.best_params_

# Final Model
catb = CatBoostClassifier(iterations = 200, depth = 5, learning_rate = 0.01)
catb_tuned = catb.fit(X_train,y_train)

y_pred = catb_tuned.predict(X_test)
accuracy_score(y_test,y_pred)





