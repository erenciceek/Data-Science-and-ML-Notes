"""
LightGBM, Microsoft tarafından geliştirilen ve Gradient Boosting framework'ünü kullanan hafif bir makine öğrenimi algoritmasıdır. LightGBM, verimlilik,
hız ve performans açısından optimize edilmiştir ve özellikle büyük veri setleri üzerinde çalışırken diğer Gradient Boosting algoritmalarına göre bazı
avantajlar sunar.

LightGBM'in temel özellikleri ve işleyişi şunlardır:
Hafif: Daha az bellek kullanımı ve daha hızlı eğitim/tahmin süreleri ile büyük veri setlerinde bile yüksek hız sağlar.
Leaf-wise Büyüme: Geleneksel Gradient Boosting algoritmaları genellikle level-wise ağaç büyümesi kullanırken, LightGBM leaf-wise büyüme kullanır. Bu yaklaşım,
hata azalmasını en çok sağlayan yaprağı (yani en büyük gradyanı olan yaprağı) seçerek ağacın büyümesini sürdürür.
Histogram Tabanlı Öğrenme: LightGBM, sürekli değerleri sabit sayıda kovaya (bin) yerleştirerek işler. Bu yaklaşım, veri bölme noktalarını daha hızlı
hesaplamayı ve aynı zamanda modelin bellek kullanımını azaltmayı sağlar.
Kategorik Özelliklerin Doğrudan İşlenmesi: Kategorik özellikleri doğrudan işleyebilme yeteneği, one-hot encoding gibi dönüşümlere gerek kalmadan etkili
sınıflandırmalar yapılmasını sağlar.
Paralel ve GPU Optimizasyonu: Paralel işlemeye ve GPU kullanımına olanak tanıyan optimizasyonlar sayesinde, LightGBM çok hızlı bir şekilde eğitilebilir.


LightGBM sınıflandırma algoritması aşağıdaki adımları içerir:
İlk Tahmin: Genellikle veri setinin hedef değişkeninin ortalaması veya modu gibi basit bir tahminle başlar.
Gradient Hesaplama: Her iterasyonda, gerçek değerler ile tahmin edilen değerler arasındaki farkın gradyanı hesaplanır.
Yeni Ağaç Eğitimi: Hesaplanan gradyanları hedef alarak yeni bir ağaç eğitilir.
Ağacı Ekleme ve Öğrenme Hızı: Yeni eğitilen ağaç, önceden belirlenen bir öğrenme hızıyla mevcut modele eklenir.
İterasyon: Belirlenen sayıda iterasyon veya erken durdurma kriterine ulaşılana kadar süreç tekrarlanır.
Son Model: Tüm iterasyonlardan sonra, oluşturulan model topluluğu sınıflandırma görevini yerine getirecek final model olarak kullanılır.


LightGBM algoritmasının avantajları arasında hızlı eğitim süreleri, büyük veri setleri ile çalışabilme ve düşük hafıza kullanımı yer alır.
Dezavantajları ise, aşırı uydurma (overfitting) riskinin olması ve modelin karmaşıklığı nedeniyle yorumlanabilirliğinin zor olabilmesidir. Parametre
ayarlaması önemli olduğundan, doğru bir model elde etmek için çapraz doğrulama ve hiperparametre optimizasyonu kullanılması önerilir.

"""


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report



diabetes = pd.read_csv("Data Science-ML/Classifications/diabetes.csv")
df = diabetes.copy()
df = df.dropna()
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)
X = pd.DataFrame(X)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,
                                                 random_state=42)
# Model ve Tahmin
from lightgbm import LGBMClassifier

lgbm = LGBMClassifier()
lgbm_model = lgbm.fit(X_train,y_train)

y_pred = lgbm_model.predict(X_test)
accuracy_score(y_test,y_pred)
# 0.01 , 3, 0.6, 500, 20
# Model Tuning
lgbm_model.get_params()
lgbm_params = {
    "n_estimators" : [100,500,1000,2000],
    "subsample" : [0.6,0.8,1.0],
    "max_depth" : [3,4,5,6],
    "learning_rate" : [0.1,0.01,0.02,0.05],
    "min_child_samples" : [5,10,20]
}

lgbm = LGBMClassifier()

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # Uyarıları bastır
    lgbm_cv_model = GridSearchCV(lgbm, lgbm_params, cv=10, n_jobs=-1, verbose=2)
    lgbm_cv_model.fit(X_train, y_train)  # Veya uygun diğer işlemleriniz
lgbm_cv_model.best_params_

# Final Model
lgbm_tuned = LGBMClassifier(learning_rate=0.01,max_depth=3,subsample=0.6,n_estimators=500,min_child_samples=20)

lgbm_tuned.fit(X_train,y_train)

y_pred = lgbm_tuned.predict(X_test)
accuracy_score(y_test,y_pred)



"""
Boosting ve Bagging (Bootstrap Aggregating), makine öğreniminde topluluk öğrenme (ensemble learning) yöntemlerinin iki temel yaklaşımıdır. Her iki teknik 
de birden çok öğreniciyi (genellikle zayıf öğrenicileri) bir araya getirerek daha güçlü ve kararlı bir model oluşturmayı hedefler, fakat bunu yaparken farklı 
stratejiler izlerler.

Boosting
Boosting, bir dizi zayıf öğreniciyi (genellikle basit karar ağaçları) ardışık olarak eğitmek üzerine kuruludur. Her bir öğrenici, öncekilerin hatalarını 
düzeltmeye çalışır:

Ardışık Eğitim: Öğreniciler ardışık olarak eğitilir, yani her bir öğrenici öncekinin yaptığı hataları azaltmaya odaklanır.
Hata Odaklı: Her yeni öğrenici, önceki öğreniciler tarafından yanlış tahmin edilen veri noktalarına daha fazla ağırlık vererek bu hataları düzeltmeye çalışır.
Düşük Sapma: Boosting, modelin sapmasını azaltmaya yöneliktir, bu da modelin veriye daha iyi uyum sağlamasını sağlar ancak aşırı uydurma (overfitting) riskini artırabilir.
Örnekler: AdaBoost, Gradient Boosting Machines (GBM), XGBoost, LightGBM, CatBoost.
Boosting, genellikle dengeli ve az gürültülü veri setlerinde çok iyi performans gösterir ve sınıflandırma ile regresyon görevlerinde etkili sonuçlar elde edebilir.

Bagging
Bagging, birden fazla öğreniciyi paralel olarak eğitmek ve bu öğrenicilerin tahminlerini birleştirmek üzerine kuruludur. Her bir öğrenici bağımsız olarak 
ve rastgele oluşturulmuş alt kümeler üzerinde eğitilir:

Paralel Eğitim: Öğreniciler birbirinden bağımsız olarak ve eşzamanlı olarak eğitilir.
Bootstrap Örneklemesi: Her bir öğrenici, orijinal veri setinden rastgele örnekleme (bootstrap) ile oluşturulmuş farklı alt kümeler üzerinde eğitilir.
Yüksek Varyansı Azaltma: Bagging, özellikle yüksek varyansa sahip öğreniciler için etkilidir (örneğin, karar ağaçları) ve modelin genel varyansını düşürmeye 
yardımcı olur.
Oylama veya Ortalama: Sınıflandırma için genellikle çoğunluk oyu (majority voting) veya yumuşak oylama (soft voting), regresyon için ise ortalama alınarak 
tahminler birleştirilir.
Örnekler: Random Forest, Extra Trees.
Bagging, özellikle aşırı uydurma riskinin olduğu durumlarda ve karar ağaçları gibi yüksek varyanslı modellerle çalışırken yararlıdır. Ayrıca, veri setindeki 
gürültü ve aykırı değerlerden daha az etkilenir.

Her iki yöntem de farklı problemlere ve veri tiplerine göre avantajlar sunar. Boosting genellikle daha doğru tahminler yapar ancak daha fazla ayar gerektirir 
ve aşırı uydurma riski taşır. Bagging ise daha hızlı ve daha kararlı olabilir ama genellikle boosting kadar yüksek doğruluk sağlamaz.
"""













