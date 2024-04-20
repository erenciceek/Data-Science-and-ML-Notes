"""
Naive Bayes, sınıflandırma görevleri için kullanılan basit ama etkili bir olasılıksal makine öğrenimi algoritmasıdır. Adını, özellikler arasındaki
koşullu bağımsızlık varsayımından alır, bu da her özelliğin sınıf üzerindeki etkisinin diğer özelliklerden bağımsız olduğu anlamına gelir (bu nedenle
"naive" yani "saf").

Naive Bayes sınıflandırıcısının temel prensibi, Bayes Teoremi'ne dayanır. Bayes Teoremi, bir olayın gerçekleşme olasılığını ve ilgili koşullu
olasılıkları bağlantılandırır.Matematiksel olarak, Bayes Teoremi şu şekilde ifade edilir:
P(A∣B) = (P(B∣A)⋅ P(A)) / P(B)

Bu formülde:
->  P(A∣B): B olayının gerçekleştiği durumda A olayının olasılığıdır (koşullu olasılık).
->  P(B∣A): A olayının gerçekleştiği durumda B olayının olasılığıdır (ters koşullu olasılık).
->  P(A): A olayının marjinal olasılığıdır.
->  P(B): B olayının marjinal olasılığıdır.

Naive Bayes sınıflandırıcısında, A olayı belirli bir sınıfın (etiketin) olasılığını ve B olayı verilen özellikler (veri) setini temsil eder. Bu şekilde,
verilen bir özellikler seti için her sınıfın koşullu olasılığını hesaplar ve en yüksek koşullu olasılığa sahip sınıfı tahmin olarak seçer.


Naive Bayes algoritması çeşitli türlerde olabilir, bunlar:
Gaussian Naive Bayes: Sürekli veriler için ve özelliklerin normal (Gaussian) dağılımı takip ettiği varsayılır.
Multinomial Naive Bayes: Metin sınıflandırma gibi frekans verilerinde kullanılır, özelliklerin multinomial dağılımı takip ettiği varsayılır.
Bernoulli Naive Bayes: İkili veya Boole veriler için kullanılır ve özelliklerin Bernoulli dağılımı takip ettiği varsayılır.

Naive Bayes'in avantajları:
Hızlı ve Verimli: Modelin eğitilmesi ve tahmin yapması diğer birçok sınıflandırma algoritmasına göre çok daha hızlıdır.
Az Veri ile Çalışabilme: Naive Bayes, diğer karmaşık modellere kıyasla daha az miktarda eğitim verisiyle bile makul sonuçlar verebilir.
Anlaşılması Kolay: Modellerinin anlaşılması ve yorumlanması nispeten kolaydır.

Ancak bazı dezavantajları da vardır:
Özellik Bağımsızlığı Varsayımı: Gerçek dünyada özellikler sıkça birbirine bağlıdır ve bu bağımsızlık varsayımı yanıltıcı olabilir.
Sınırlı Uygulama: Özelliklerin bağımlılığının güçlü olduğu durumlarda, performansı azalabilir.

Naive Bayes, basitliği ve hızı nedeniyle, özellikle metin sınıflandırma (spam filtreleme, duyarlılık analizi vb.) gibi uygulamalarda yaygın olarak kullanılır.

"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report



diabetes = pd.read_csv("Data Science-ML/Classifications/diabetes.csv")
df = diabetes.copy()
df = df.dropna()
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,
                                                 random_state=42)

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb_model = nb.fit(X_train,y_train)
nb_model.__doc__

#TAHMİN
nb_model.predict(X_test)[0:10]
y_pred = nb_model.predict(X_test)
accuracy_score(y_test,y_pred)

cross_val_score(nb_model, X_test,y_test, cv=10).mean()









