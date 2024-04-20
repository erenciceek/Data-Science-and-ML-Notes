"""
CART (Classification and Regression Trees) hem sınıflandırma (Classification Trees) hem de regresyon (Regression Trees) problemleri için kullanılan bir
karar ağacı algoritmasıdır. CART modeli, veri setini en iyi ayıran özellik ve eşik değerini bulmak için ikili ağaç yapısını kullanarak veri setini alt
kümelerine ayırır.

CART sınıflandırma algoritmasının temel adımları şunlardır:
Özellik Seçimi: Veri setindeki her bir özellik ve bu özelliklere ait potansiyel tüm eşik değerleri üzerinde, en iyi bölünmeyi (split) verecek özellik-eşik
değeri çiftini belirlemek için bir ölçüt kullanılır. Sınıflandırma problemlerinde, genellikle Gini impurity veya entropy kullanılır.
En İyi Bölünme Noktasını Bulma: Seçilen ölçüte göre en düşük maliyeti (en düşük Gini impurity veya entropy değerini) veren özellik ve eşik değeri, o düğüm
için en iyi bölünme noktası olarak seçilir.
Ağaç Oluşturma: En iyi bölünme noktası belirlendikten sonra, veri seti iki alt kümeye ayrılır. Bu işlem, her bir alt küme homojen bir sınıf dağılımına sahip
olana kadar ya da durdurma kriterleri karşılanana kadar devam eder. Durdurma kriterleri, ağacın maksimum derinliği, düğümdeki minimum örnek sayısı veya bir
düğümün bölünmesi için gerekli minimum örnek sayısı olabilir.
Budama: Ağacın aşırı öğrenmesini önlemek için, oluşturulan ağaç üzerinde budama (pruning) işlemi gerçekleştirilebilir. Budama, ağacın karmaşıklığını azaltarak
modelin genelleştirme kabiliyetini artırmayı hedefler.
Sınıflandırma: Eğitilmiş karar ağacı, yeni bir örneğin sınıfını tahmin etmek için kullanılır. Örnek, ağaçtaki karar düğümlerini takip ederek uygun yaprak
düğüme ulaşana kadar ilerler. Yaprak düğüm, örneğin ait olduğu sınıfı temsil eder.


CART algoritmasının avantajları şunlardır:
Anlaşılması Kolay: Karar ağaçları, kuralların açık bir şekilde ifade edilmesini sağlar ve modellerin yorumlanabilirliği yüksektir.
Veri Ön İşlemeye İhtiyaç Az: Normalleştirme veya ölçeklendirme gerektirmez ve hem sayısal hem de kategorik verilerle çalışabilir.
Hızlı ve Ölçeklenebilir: Büyük veri setleri üzerinde bile nispeten hızlı çalışır.


Dezavantajları ise şunlardır:
Aşırı Öğrenme (Overfitting): Eğer düzgün bir şekilde budama yapılmazsa, karar ağaçları veriye aşırı uydurma eğilimindedir.
Dengesiz Veri Setlerine Duyarlılık: Azınlık sınıfına ait örneklerin komşu olarak seçilme olasılığı düşük olduğunda yanlı tahminler yapabilir.
Karar Sınırları Sınırlıdır: Karar ağaçları doğrusal olmayan karmaşık sınırları modellemekte sınırlı olabilir, özellikle düzlemsel bölünmelerle kısıtlıdır.


CART algoritması, karmaşık olmayan sınıflandırma ve regresyon problemlerinde kullanışlı bir başlangıç noktasıdır ve özellikle veri keşfi ve yorumlanabilir
modeller oluşturmak için tercih edilir.
"""



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report



diabetes = pd.read_csv("Data Science-ML/Classifications/diabetes.csv")
df = diabetes.copy()
df = df.dropna()
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)
#X = df["Pregnancies"]
X = pd.DataFrame(X)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,
                                                 random_state=42)


from sklearn.tree import DecisionTreeClassifier
cart = DecisionTreeClassifier()
cart_model = cart.fit(X_train,y_train)
cart_model.get_params()



# ağaçtaki kuralı yazdırıyoruz.
from skompiler import skompile
print(skompile(cart_model.predict).to("python/code"))

x = [3]
((0 if x[0] <= 2.5 else 0) if x[0] <= 6.5 else 1 if x[0] <= 13.5 else 1)


y_pred = cart_model.predict(X_test)
accuracy_score(y_test,y_pred)


# Model Tuning
cart_grid = {"max_depth": range(1,10),
            "min_samples_split" : list(range(2,50)) }

cart = DecisionTreeClassifier()
cart_cv = GridSearchCV(cart, cart_grid, cv = 10, n_jobs = -1, verbose = 2)
cart_cv_model = cart_cv.fit(X_train, y_train)

print("En iyi parametreler: " + str(cart_cv_model.best_params_))

cart = DecisionTreeClassifier(max_depth=5, min_samples_split=19)
cart_tuned = cart.fit(X_train,y_train)

y_pred = cart_tuned.predict(X_test)
accuracy_score(y_test,y_pred)





