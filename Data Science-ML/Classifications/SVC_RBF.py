"""
SVC (Support Vector Classification) ile RBF (Radial Basis Function) kernel kullanımı, destek vektör makinesi sınıflandırma algoritmalarında en yaygın
kullanılan yaklaşımlardan biridir. RBF kerneli, özellikle doğrusal olarak ayrılamayan veri setleri için kullanışlıdır çünkü bu kernel fonksiyonu, verileri
daha yüksek boyutlu bir özellik uzayına dönüştürerek sınıflar arasında doğrusal bir ayrım hiper-düzlemini bulmayı mümkün kılar.

RBF kernelinin temel özelliği, her bir veri noktasının etkisinin, mesafe arttıkça azalan 'radyal' bir şekilde yayılmasıdır. Yani, bir noktanın bir diğer nokta
üzerindeki etkisi, aralarındaki mesafenin bir fonksiyonudur. Bu özellik, veri noktaları arasında pürüzsüz ve esnek sınırlar oluşturulmasını sağlar.

SVC RBF kerneli şu adımları takip eder:
1. Uzaklık Hesaplama: Veri noktaları arasındaki Öklid uzaklığını hesaplar.
2. RBF Kernel Fonksiyonu: Uzaklıkları bir kernel fonksiyonu aracılığıyla yeni bir özellik uzayına dönüştürür. RBF kernel fonksiyonu genellikle şu şekildedir:
    K(xi,xj)=exp(−γ∣∣xi − xj∣∣2)
Burada K kernel fonksiyonudur, xi ve xj veri noktalarıdır, ve γ pozitif bir öğrenme parametresidir. γ değeri, etki alanının ne kadar geniş olduğunu kontrol eder;
yani γ yüksekse, her bir noktanın etki alanı daralır ve model daha fazla ayrıntıya duyarlı hale gelir.
3. Optimizasyon ve Model Eğitimi: Modelin eğitimi sırasında, yüksek boyutlu özellik uzayında maksimum marjinal hiper-düzlemi bulmak için bir optimizasyon
problemi çözülür.
4. Destek Vektörleri: RBF kerneli, destek vektörlerini seçer ve bu vektörler modelin nasıl karar vereceğini belirler. Bunlar, hiper-düzlemi destekleyen ve
marjinal boşluğun sınırlarını çizen kritik veri noktalarıdır.
5. Sınıflandırma: Model eğitildikten sonra, herhangi bir yeni veri noktasının sınıfını tahmin etmek için, o noktanın destek vektörlerine olan uzaklığı ve
ilgili kernel değerleri hesaplanır ve sınıflandırma kararı verilir.

SVC RBF'nin avantajları ve dezavantajları genel SVC yorumlarında bahsedilenlerle benzerdir, ancak RBF özelinde önemli bir nokta, kernel parametresi γ'nın ayarının
modelin performansı üzerinde büyük etkisi olmasıdır. γ ve düzenlileştirme parametresi C'nin doğru bir şekilde ayarlanması, modelin hem doğruluk hem de
genelleştirme yeteneği açısından kritik önem taşır. Yüksek bir γ değeri, modelin aşırı uydurma (overfitting) yapmasına yol açabilirken, çok düşük bir γ değeri
modelin yetersiz uydurma (underfitting) yapmasına sebep olabilir. Bu parametrelerin çapraz doğrulama gibi tekniklerle titizlikle ayarlanması tavsiye edilir.

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

# Model ve Tahmin
from sklearn.svm import SVC
svc_model = SVC(kernel ="rbf").fit(X_train,y_train)
svc_model.__doc__
y_pred = svc_model.predict(X_test)
accuracy_score(y_test,y_pred)

# Model Tuning
svc_params = {"C": [0.0001, 0.001, 0.1, 1, 5, 10 ,50 ,100],
             "gamma": [0.0001, 0.001, 0.1, 1, 5, 10 ,50 ,100]}

svc = SVC()
svc_cv_model = GridSearchCV(svc, svc_params,
                         cv = 10,
                         n_jobs = -1,
                         verbose = 2)

svc_cv_model.fit(X_train, y_train)
print("En iyi parametreler: " + str(svc_cv_model.best_params_))
svc_tuned = SVC(C = 10, gamma = 0.0001).fit(X_train, y_train)
y_pred = svc_tuned.predict(X_test)
accuracy_score(y_test, y_pred)