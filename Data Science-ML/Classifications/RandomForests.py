"""
Random Forests, karar ağaçlarının topluluk öğrenme (ensemble learning) yaklaşımıyla birleştirildiği güçlü bir sınıflandırma algoritmasıdır. Bu yöntem,
birden çok karar ağacını bir araya getirerek her bir ağacın güçlü yönlerinden yararlanır ve zayıf yönlerini azaltır, böylece genel olarak daha stabil
ve güvenilir bir sınıflandırma performansı sunar.


Random Forests algoritması şu adımları takip eder:
Bootstrap Örnekleri: Veri setinden rastgele seçilmiş örneklerle (bootstrap sampling) birçok farklı eğitim veri seti oluşturulur. Her bir karar ağacı,
bu örneklerden oluşturulan farklı veri setleri üzerinde eğitilir.
Özellik Alt Kümeleri: Her bir bölünme (split) işleminde, tüm özelliklerin yerine rastgele seçilmiş özellik alt kümeleri kullanılır. Bu yaklaşım, ağaçlar
arasındaki korelasyonu azaltarak modelin varyansını düşürmeyi amaçlar.
Ağaçların Bağımsız Eğitimi: Oluşturulan her bir karar ağacı bağımsız olarak eğitilir ve kesin budama (pruning) yapılmaz. Her bir ağacın fazla uydurma
(overfitting) yapması muhtemeldir, ancak topluluk olarak bu etki azalır.
Tahminlerin Birleştirilmesi: Sınıflandırma görevinde, bir örneğin sınıfı her bir ağacın tahminiyle "oylama" yaparak belirlenir. En çok oy alan sınıf, o
örneğin sınıfı olarak atanır. Bu, "çoğunluk oylaması" olarak bilinir.
Sonuçların Agregasyonu: Regresyon görevinde, bireysel ağaçların tahminleri ortalaması alınarak nihai tahmin yapılır.


Random Forests'ın avantajları şunlardır:
Yüksek Doğruluk: Topluluk öğrenme yaklaşımı, genellikle tek bir karar ağacına göre daha yüksek doğruluk sağlar.
Aşırı Öğrenmeye Karşı Direnç: Rastgele özellik seçimi ve birden fazla ağaç kullanımı, modelin veriye aşırı uydurma riskini azaltır.
Özellik Önemi: Random Forests, her bir özelliğin sınıflandırma üzerindeki etkisini belirlemek için kullanılabilir, bu da özellik seçimi için yararlıdır.
Esneklik: Hem sınıflandırma hem de regresyon görevleri için etkili bir şekilde kullanılabilir.
Kullanım Kolaylığı: Hiperparametre ayarlaması nispeten basittir ve genellikle iyi varsayılan ayarlarla güçlü sonuçlar üretir.

Dezavantajları ise şunlardır:
Yorumlanabilirlik: Tek bir karar ağacına göre daha az yorumlanabilir, çünkü yüzlerce veya binlerce ağaçtan oluşan bir ormanın bireysel kararlarını izlemek zordur.
Eğitim Süresi: Çok fazla sayıda ağaç ve büyük veri setleri kullanıldığında eğitim süresi uzayabilir.
Hesaplama ve Bellek Kullanımı: Büyük ölçekli veri setleri üzerinde çok sayıda ağaç eğitmek hesaplama açısından maliyetli olabilir ve daha fazla bellek kullanımı gerektirebilir.


Random Forests, sınıflandırma ve regresyon problemlerinin çözümünde sıkça tercih edilen, sağlam ve etkili bir yöntemdir.
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
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier().fit(X_train,y_train)
print(rf_model.__doc__)

y_pred = rf_model.predict(X_test)
accuracy_score(y_test,y_pred)


# Model Tuning
rf_params = {"max_depth" : [2,5,8,10],
             "max_features" : [2,5,8],
             "n_estimators" : [10,500,1000],
             "min_samples_split" : [2,5,10]
             }
rf_cv_model = GridSearchCV(rf_model,rf_params, cv=10, n_jobs=-1, verbose=2)
rf_cv_model.fit(X_train,y_train)
print("En iyi parametreler : " + str(rf_cv_model.best_params_))

rf = RandomForestClassifier(max_depth=10, max_features=8, min_samples_split=10, n_estimators=1000)
rf_tuned = rf.fit(X_train,y_train)

y_pred = rf_tuned.predict(X_test)
accuracy_score(y_test,y_pred)


# Değişkenlerin önem derecelerini görüntülemek için :
Importance = pd.DataFrame({"Importance": rf_tuned.feature_importances_*100},
                         index = X_train.columns)

Importance.sort_values(by = "Importance",
                       axis = 0,
                       ascending = True).plot(kind ="barh", color = "r");
plt.xlabel("Değişken Önem Düzeyleri")








