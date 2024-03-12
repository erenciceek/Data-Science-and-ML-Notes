# Temel amaç bağımlı ve bağımsız değişkenler arasındaki ilişkiyi ifade eden doğrusal fonksiyonu bulmaktır.

import pandas as pd
ad = pd.read_csv("../Advertising.csv", usecols = [1, 2, 3, 4])
df = ad.copy()
df.head()

X = df.drop("sales",axis=1) # SALES değişkenini silip geri kalanını atar
y = df["sales"]

from sklearn.model_selection import train_test_split, cross_val_score,cross_val_predict

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, random_state =42)

X_train.shape
y_train.shape
X_test.shape
y_test.shape

training = df.copy()
training.shape

# yorumlamaya ihtiyacımız varsa statsmodels kütüphanesini kullanmamız gerekiyor , eğer gerek yoksa sklearn kullan
import statsmodels.api as sm
lm = sm.OLS(y_train,X_train)
model = lm.fit()
model.summary()

print(model.summary().tables[1])

# R-squared modelin açıklanabilirliğini belirtir oran olarak (bağımlı değişkenlerin varyansının vs bağımsız değişkenler tarafından açıklanabilirliği)
# F-statistic değeri ise modelin anlamlılığını değerlendirir.
# varyans açıklama başarımız da F-statistic değeridir.


# scikit-learn ile model
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
model = lm.fit(X_train, y_train)
model.intercept_  # sabit katsayı değeri
model.coef_  # bağımsız değişkenlerin katsayılarını ifade ediyor.

## TAHMİN
# Model denklemi : Sales = 2.97 + TV*0.04 + radio*0.18 + newspaper*0.002
# Örneğin 30 birim TV harcaması, 10 birim radio harcaması, 40 birim de gazete harcaması olduğunda satışların tahmini değeri ne olur ?

yeni_veri = [[30],[10],[40]]
yeni_veri = pd.DataFrame(yeni_veri).T

model.predict(yeni_veri)

from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
train_rmse = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
train_rmse # train error
test_rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
test_rmse



# MODEL TUNING / Model Doğrulama

X = df.drop("sales",axis=1) # SALES değişkenini silip geri kalanını atar
y = df["sales"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, random_state =144)

lm = LinearRegression()
model = lm.fit(X_train, y_train)
np.sqrt(mean_squared_error(y_train, model.predict(X_train)))

model.score(X_train, y_train)
cross_val_score(model,X_train,y_train, cv=10,scoring="r2").mean()
np.sqrt(-cross_val_score(model,
                X_train,
                y_train,
                cv=10,
                scoring="neg_mean_squared_error")).mean()


np.sqrt(mean_squared_error(y_test, model.predict(X_test)))

np.sqrt(-cross_val_score(model,
                X_test,
                y_test,
                cv=10,
                scoring="neg_mean_squared_error")).mean()

"""
model.score(X_train, y_train): Eğitim verileri üzerinde modelin R^2 skorunu hesaplar. R^2 skoru, modelin verileri 
ne kadar iyi açıkladığının bir ölçüsüdür.

cross_val_score: Modelin çapraz doğrulama skorlarını hesaplar. 
cv=10 parametresi 10 katlı çapraz doğrulama yapılacağını belirtir.
scoring="r2" ile R^2 skorunun kullanılacağını belirtir. .mean() ile çapraz doğrulama skorlarının ortalaması alınır.


cross_val_score fonksiyonu ile 10 katlı (10-fold) çapraz doğrulama yapmak, veri setinizin rastgele olarak on eşit parçaya bölünüp, 
modelinizi her bir parça üzerinde test etmek ve geri kalan dokuz parça üzerinde eğitmek anlamına gelir. Bu işlem her parça için tekrarlanır. 
Yani, her parça bir kez test seti olarak kullanılırken, diğer dokuz parça eğitim seti olarak kullanılır.

İşte 10 katlı çapraz doğrulamanın adım adım açıklaması:

Veri Bölünmesi: Veri seti 10 eşit (veya yaklaşık eşit) bölüme ayrılır.

Model Eğitimi ve Değerlendirme: Model, ilk olarak 1. parçayı hariç tutarak geri kalan 9 parça üzerinde eğitilir. 
Daha sonra, eğitilen model 1. parçayı kullanarak test edilir ve performansı kaydedilir.

İterasyon: Bu işlem, her bir parçanın tam olarak bir kez test seti olarak kullanılacağı şekilde 10 kez tekrarlanır.

Sonuçların Toplanması: Her iterasyondan elde edilen performans skorları (örneğin, doğruluk, R^2, MSE vb.) toplanır.

Ortalama Skor: Elde edilen skorlar ortalama alınarak modelin genel performansının bir ölçüsü olarak sunulur.

Bu süreç, modelinizin farklı veri alt kümesinde nasıl performans göstereceğine dair güvenilir ve genelleştirilmiş bir tahmin sağlar.
Çapraz doğrulama, özellikle veri setinizin boyutu küçükse veya modelinizin genelleme yeteneğini sağlam bir şekilde değerlendirmek istediğinizde önemlidir.

10 katlı çapraz doğrulama, modelinizin veri setinizin farklı yönlerini öğrenme kapasitesini ve veri setinizin tamamına genelleme 
yapma yeteneğini test etmenin iyi bir yoludur. Ayrıca, modelinizin aşırı öğrenip öğrenmediğini (overfitting) kontrol etmenin de bir yoludur, 
çünkü model her seferinde farklı bir test seti üzerinde değerlendirilir. Bu, modelinizin yalnızca eğitim verilerine değil, 
görmediği verilere de iyi genelleme yapabilme yeteneğini gösterir.




R-kare (R²)
R-kare, model tarafından açıklanan toplam varyansın oranını ölçer. Diğer bir deyişle, bağımsız değişkenlerin bağımlı değişkendeki varyansı ne kadar iyi açıkladığını gösterir.
R-kare değeri 0 ile 1 arasında değişir. Değer 1'e yaklaştıkça, modelin verileri daha iyi açıkladığı anlamına gelir. 
R-kare değeri şu şekilde yorumlanabilir:

0 Yakını: Model, bağımlı değişkendeki varyansın çok azını veya hiçbirini açıklamıyor.
1 Yakını: Model, bağımlı değişkendeki varyansın büyük bir kısmını açıklıyor.
R-kare aynı zamanda "açıklanan varyans oranı" olarak da bilinir.

F İstatistiği
F istatistiği, genel olarak modelin anlamlılığını test etmek için kullanılır. Özellikle, modeldeki tüm regresyon katsayılarının sıfıra 
eşit olup olmadığını test eder. Bu, modelin bağımlı değişkeni açıklamada bağımsız değişkenlerden herhangi birinin faydalı olup olmadığını 
belirlemek için kullanılır. F istatistiği, genel model uygunluğunun bir ölçüsüdür ve şu şekilde yorumlanabilir:

F istatistiği büyük ve ilgili p-değeri küçükse (genellikle 0.05 veya daha az): Modelin bağımlı değişkeni açıklamada anlamlı olduğu sonucuna varılır. 
Bu, modeldeki en az bir bağımsız değişkenin bağımlı değişken üzerinde anlamlı bir etkisi olduğu anlamına gelir.
F istatistiği küçük ve ilgili p-değeri büyükse: Modelin anlamlı olmadığı ve bağımsız değişkenlerin bağımlı değişken üzerindeki etkisinin 
istatistiksel olarak önemli olmadığı sonucuna varılır.
R-kare ve F istatistiği birlikte, modelin verileri ne kadar iyi açıkladığını ve modelin anlamlılığını değerlendirmek için kullanılır.
Ancak, bu istatistiklerin sınırlılıklarının farkında olmak önemlidir. Örneğin, R-kare değeri, modelin doğruluğunu veya tahmin edici değişkenlerin bireysel önemini değil, sadece modelin açıkladığı varyans oranını ölçer. Benzer şekilde, F istatistiği modelin genel anlamlılığını değerlendirir, ancak bireysel değişkenlerin önemini doğrudan test etmez.
"""



