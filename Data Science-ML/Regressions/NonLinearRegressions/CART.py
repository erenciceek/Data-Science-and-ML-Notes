# Classification and Regression Trees
"""
Amaç veri seti içerisindeki karmaşık yapıları basit karar yapılarına dönüştürmektir.
Heterojen veri setleri belirlenmiş bir hedef değişkene göre homojen alt gruplara ayrılır.


CART (Classification and Regression Trees), hem sınıflandırma hem de regresyon problemleri için kullanılan ve bir veri setini en iyi
şekilde temsil eden karar ağaçları oluşturan bir makine öğrenmesi algoritmasıdır. 1984 yılında Leo Breiman, Jerome Friedman,
Charles J. Stone ve R.A. Olshen tarafından geliştirilmiştir.

Anahtar Özellikler:
İkili Ağaç Yapısı: CART, her düğümü iki alt düğüme (evet/hayır veya doğru/yanlış) ayırarak ikili bir ağaç oluşturur.
Gini Impurity veya Ortalama Kare Hata (MSE): Sınıflandırma için Gini impurity ve regresyon için Ortalama Kare Hata (MSE) ölçütlerini kullanır.
Bu ölçütler, her bir bölme işleminde homojenliği artırarak en iyi bölünmeyi seçmek için kullanılır.

Özellik Seçimi: En iyi bölünmeyi belirlemek için her adımda tüm özellikler ve bunların tüm mümkün bölme noktaları değerlendirilir.
Budama (Pruning): Overfitting'i önlemek için, ağaç oluşturulduktan sonra budama işlemi uygulanır. Bu, modelin genelleştirme yeteneğini artırır.
Değişken Önem Derecelendirmesi: Hangi özelliklerin tahminde en etkili olduğunu belirler ve bunları önem derecesine göre sıralar.
Kayıp Fonksiyonları: Farklı kayıp fonksiyonları, belirli veri setlerine veya problemlere özel olarak uyarlanabilir.
Sayısal ve Kategorik Değişkenler: Hem sayısal hem de kategorik değişkenlerle çalışabilir.

Çalışma Mekanizması:
Bölme (Splitting): Veri seti, seçilen bir özelliğe göre iki alt küme oluşturacak şekilde bölünür. Bu özellik ve bölme noktası, Gini impurity veya MSE'yi en çok azaltacak şekilde seçilir.
Büyütme (Growing): Bu süreç, ağaç yeterli bir detay seviyesine ulaşana veya kullanıcı tarafından belirlenen bir durdurma kriteri sağlanana kadar tekrarlanır.
Budama (Pruning): Daha basit bir ağaç oluşturmak ve overfitting'i önlemek için gerçekleştirilir. Doğrulama veri seti üzerindeki performansına göre ağacın dalları kırpılır.
Sonuç: Budanmış ağaç, sınıflandırma veya regresyon için son model olarak kullanılır.


Avantajları ve Dezavantajları:

Avantajları:
Anlaşılması ve yorumlanması kolaydır.
Veri ön işleme gereksinimi azdır.
Hem sınıflandırma hem de regresyon görevlerinde kullanılabilir.

Dezavantajları:
Diğer bazı algoritmalara göre daha düşük tahmin performansı gösterebilir.
Veri değişikliklerine ve aykırı değerlere karşı hassastır.
Aşırı uyma (overfitting) eğilimindedir, ancak budama ile bu sorun hafifletilebilir.

CART genellikle, topluluk yöntemleri olan Bagging, Random Forest ve Boosting algoritmalarının temelini oluşturur.
Bu yöntemler, birçok karar ağacının tahminlerini birleştirerek daha güçlü ve kararlı modeller oluşturur.

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

X_train = pd.DataFrame(X_train["Hits"])
X_test = pd.DataFrame(X_test["Hits"])

cart_model = DecisionTreeRegressor(max_leaf_nodes=10)
cart_model.get_params()
cart_model.set_params(criterion = "friedman_mse")
cart_model.fit(X_train,y_train)


X_grid = np.arange(min(np.array(X_train)),max(np.array(X_train)),0.01)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X_train,y_train,color="red")
plt.plot(X_grid, cart_model.predict(X_grid),color='blue')
plt.title("CART Regression Tree")
plt.xlabel("Atış sayısı(Hits)")
plt.ylabel("Maaş (Salary)")
plt.show()




#pip install skompiler
from skompiler import skompile

print(skompile(cart_model.predict).to('python/code'))

# Tahmin

x = [91]
(345.2011551724138 if x[0] <= 117.5 else ((((1300.0 if x[0] <= 118.5 else
    641.0) if x[0] <= 122.5 else 1468.5236666666667) if x[0] <= 125.5 else
    621.9679230769232) if x[0] <= 143.0 else (958.6111111111111 if x[0] <=
    150.5 else 2460.0) if x[0] <= 151.5 else 499.1666666666667 if x[0] <=
    157.5 else 892.5402413793104) if x[0] <= 225.5 else 1975.0)

cart_model.predict(X_test)[0:5]
cart_model.predict([[91]])

y_pred = cart_model.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))



#   MODEL TUNING

cart_model = DecisionTreeRegressor()
cart_model.set_params(criterion = "friedman_mse")
cart_model.fit(X_train,y_train)
y_pred = cart_model.predict(X_test)

np.sqrt(mean_squared_error(y_test,y_pred))

cart_params = {"min_samples_split": range(2,100),
          "max_leaf_nodes" : range(2,10)}
cart_cv_model = GridSearchCV(cart_model,cart_params,cv=10)
cart_cv_model.fit(X_train,y_train)
cart_cv_model.best_params_


cart_tuned = DecisionTreeRegressor(max_leaf_nodes=9, min_samples_split=76)
cart_tuned.fit(X_train,y_train)
y_pred = cart_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))


# tüm değişkenleri kullanıp dataframe'e aldığımız veri seti ile modeli eğittiğimizde, tune edilmiş modelin hata değeri 376 geliyor.
# Yani değişken sayısı arttığında ve uygun hiperparametreler gridsearch ile bulunduğunda tahmin edebilme başarısı artıyor.






