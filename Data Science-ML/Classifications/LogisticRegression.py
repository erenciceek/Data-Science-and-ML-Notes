"""
Logistik regresyon, özellikle ikili sınıflandırma problemlerinde kullanılan istatistiksel bir modelleme tekniğidir.
Temel amacı, bağımlı değişkenin (genellikle 0 ve 1 arasında kodlanan bir kategori) olasılığını, bağımsız değişken(ler)
üzerinden tahmin etmektir. Lineer regresyonun bir çeşidi olmasına rağmen, logistik regresyon bir çıktıyı 0 ve 1 arasında
sıkıştırarak olasılık olarak yorumlanabilecek bir değer üretir, yani bir olayın olma ya da olmama olasılığını modeller.

Logistik regresyonun çalışma prensibi şu adımlardan oluşur:

Olasılık Tahmini: Öncelikle, bağımsız değişkenlerin lineer bir kombinasyonu kullanılarak bir log-odds ya da logit değeri
hesaplanır. Matematiksel olarak bu ...... olarak ifade edilir, burada p olayın olasılığını temsil eder.

Logistic Fonksiyonu (Sigmoid): Elde edilen logit değeri, 0 ile 1 arasında bir değere dönüştürmek için bir lojistik fonksiyon
(veya sigmoid fonksiyon) uygulanır. Bu fonksiyon .... şeklinde ifade edilir, burada z logit değeridir.

Parametre Tahmini: Logistik regresyon modelinin parametreleri (ağırlıklar ve bias), genellikle maksimum olabilirlik tahmini
(maximum likelihood estimation - MLE) yöntemiyle tahmin edilir. Bu, gözlemlenen verilerin olasılığını maksimize edecek
parametre değerlerini bulmayı amaçlar.

Sınıflandırma Kararı: Model tarafından üretilen tahmin edilen olasılık, bir eşik değere (genellikle 0.5) göre değerlendirilir.
Eğer tahmin edilen olasılık eşik değerden yüksekse, olayın gerçekleşeceği (1 olarak sınıflandırılacak), eşik değerden düşükse
gerçekleşmeyeceği (0 olarak sınıflandırılacak) sonucuna varılır.

Logistik regresyonun avantajları arasında modelin anlaşılabilirliği, hesaplama açısından verimliliği ve çıktısının olasılık olarak
yorumlanabilir olması bulunur. Ayrıca, çeşitli alanlarda, örneğin tıp (hastalık varlığı), finans (iflas tahmini), pazarlama
(müşterinin ürünü satın alma olasılığı) gibi yerlerde yaygın olarak kullanılır.

Ancak, logistik regresyon doğrusal bir sınıflayıcıdır ve karmaşık ilişkileri modellemekte sınırlı olabilir. Ayrıca, verilerin
doğrusal olarak ayrılabilir olduğu varsayımına dayanır ve aykırı değerlere duyarlıdır. Karmaşık problemler için karar ağaçları,
random forest veya derin öğrenme gibi diğer algoritmalar tercih edilebilir.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score,roc_curve
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt


from warnings import filterwarnings
filterwarnings('ignore')


# VERİYİ HAZIRLAMA VE İNCELEME BÖLÜMÜ
diabetes = pd.read_csv("Data Science-ML/Classifications/diabetes.csv")
df = diabetes.copy()
df = df.dropna()
df.head()

df.info()
df["Outcome"].value_counts()
df["Outcome"].value_counts().plot.barh();

df.describe().T

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

# statsmodels ile
loj = sm.Logit(y,X)
loj_model = loj.fit()
loj_model.summary()


# scikit-learn
from sklearn.linear_model import LogisticRegression


loj = LogisticRegression(solver="liblinear")
loj_model = loj.fit(X, y)

# Modelin parametrelerini görüntüleme
print("Coefficients: ", loj_model.coef_)
print("Intercept: ", loj_model.intercept_)



# TAHMİN
y_pred = loj_model.predict(X)
confusion_matrix(y,y_pred)
accuracy_score(y,y_pred)

print(classification_report(y,y_pred))

loj_model.predict(X)[0:5] # 0 ve 1 olarak tahmin üretir
loj_model.predict_proba(X)[0:10] # 0 ve 1 olma olasılıklarını gösterir
loj_model.predict_proba(X)[0:10][:,0:1]
y[0:10]

y_probs = loj_model.predict_proba(X)
y_probs = y_probs[:,1]
y_pred = [1 if i > 0.5 else 0 for i in y_probs]


confusion_matrix(y,y_pred)
accuracy_score(y,y_pred)

logit_roc_auc = roc_auc_score(y, loj_model.predict(X))

fpr, tpr, thresholds = roc_curve(y, loj_model.predict_proba(X)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='AUC (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Oranı')
plt.ylabel('True Positive Oranı')
plt.title('ROC')
plt.show()



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,
                                                 random_state=42)


loj = LogisticRegression(solver="liblinear")
loj_model = loj.fit(X_train, y_train)
accuracy_score(y_test,loj_model.predict(X_test))

cross_val_score(loj_model,X_test,y_test,cv = 10).mean()






