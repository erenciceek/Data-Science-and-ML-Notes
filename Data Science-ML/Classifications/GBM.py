"""
Gradient Boosting Machines (GBM), gözetimli öğrenme problemleri için kullanılan güçlü ve esnek bir topluluk öğrenme tekniğidir. Karar ağaçları gibi zayıf
öğrenicilerin bir dizi yinelemeli ve adaptif aşamada iyileştirilmesi ilkesine dayanır. Sınıflandırma görevleri için GBM, sınıflara ait olasılıkları tahmin
ederken hataları azaltacak şekilde her adımda modeli günceller.

GBM algoritması, her bir aşamada bir öncekinin hatalarını düzeltecek şekilde yeni bir öğrenici ekleyerek güçlü bir topluluk modeli oluşturur.
İşte GBM sınıflandırma algoritmasının adımları:

İlk Tahmin: Genellikle veri setinin hedef değişkeninin ortalaması veya medyanı gibi basit bir tahmin ile başlar.
Hata Hesaplama: İlk tahminin yarattığı hatalar (rezidüaller) hesaplanır. Bu, gerçek değerler ile tahmin edilen değerler arasındaki farktır.
Yeni Öğrenici Eğitimi: Bu rezidüaller üzerine bir karar ağacı eğitilir. Yani, bu ağaç hatalar üzerine eğitilmiştir ve amaç, bu hataları azaltmaktır.
Ağırlıklandırma ve Ekleme: Yeni öğrenici, mevcut model topluluğuna eklenmeden önce uygun bir ağırlık (learning rate) ile çarpılır. Bu ağırlık, modelin
her adımda ne kadar "öğreneceğini" kontrol eder ve genellikle 0.01 ile 0.3 arasında bir değer alır.
Tahmin Güncellemesi: Yeni öğrenici model topluluğuna eklenir ve tahminler güncellenir.
İterasyon: Yukarıdaki adımlar, belirli bir sayıda iterasyon için veya bir durdurma kriteri karşılanana kadar (örneğin, artık bir iyileşme olmadığında) tekrarlanır.
Sonuç: Eğitim tamamlandığında, tüm öğrenicilerin tahminleri bir araya getirilerek nihai model oluşturulur.


GBM sınıflandırma algoritmasının avantajları:
Yüksek Performans: GBM, karmaşık veri setlerinde bile genellikle çok iyi performans gösterir.
Esneklik: Farklı kayıp fonksiyonları kullanılabilir, bu sayede farklı türde problemlere uygulanabilir.
Özellik Önemleri: Özelliklerin model üzerindeki önemini belirlemede etkili bilgiler sağlar.

GBM sınıflandırma algoritmasının dezavantajları:
Aşırı Öğrenme: Eğer parametreler doğru ayarlanmazsa, GBM veriye aşırı uydurabilir.
Hesaplama Maliyeti: GBM, özellikle çok sayıda ağaç ve/veya derin ağaçlar kullanıldığında, eğitimi zaman alıcı olabilir.
Parametre Ayarı: GBM'in başarısı, öğrenme oranı ve ağaç sayısı gibi hiperparametrelerin doğru ayarlanmasına büyük ölçüde bağlıdır ve bu parametreleri ayarlamak
zor olabilir.


GBM algoritması, sınıflandırma yarışmalarında ve karmaşık veri setlerinde sıkça tercih edilir. Ancak, başarılı modeller oluşturmak için deneyim ve dikkatli
parametre ayarlaması gerektirir.
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
from sklearn.ensemble import GradientBoostingClassifier
gbm_model = GradientBoostingClassifier().fit(X_train,y_train)

y_pred = gbm_model.predict(X_test)
accuracy_score(y_test,y_pred)

# Model Tuning
print(gbm_model.__doc__)
gbm_params = {"learning_rate" : [0.001,0.01,0.1,0.05],
              "n_estimators" : [100,500,1000],
              "max_depth" : [3,5,10],
              "min_samples_split" : [2,5,10]}

gbm = GradientBoostingClassifier()
gbm_cv = GridSearchCV(gbm,gbm_params,cv=10,n_jobs=-1,verbose=2)
gbm_cv.fit(X_train,y_train)

print("En iyi parametreler : " + str(gbm_cv.best_params_))


gbm = GradientBoostingClassifier(learning_rate=0.1, max_depth=3 , min_samples_split=5, n_estimators=100)
gbm_tuned = gbm.fit(X_train,y_train)

y_pred = gbm_tuned.predict(X_test)
accuracy_score(y_test,y_pred)

# videoda grid search sonrası dönen optimum parametre değerleri : learning_rate : 0.01, max_depth : 3, min_samples_split : 5, n_estimators = 500


"""
GridSearchCV sonucunda dönen optimum parametrelerin farklı olmasının birkaç olası nedeni vardır. Bu durum, genellikle veri, kod veya çalışma ortamındaki 
farklılıklardan kaynaklanır:

Veri Setindeki Farklılıklar: Eğer veri setiniz videoda kullanılanla tam olarak aynı değilse (örneğin, farklı bir örnekleme, eksik verilerin farklı şekilde 
işlenmesi vb.) bu, modelin farklı parametreleri optimum olarak belirlemesine neden olabilir. Ayrıca, veri ön işleme adımları (örneğin, ölçeklendirme, 
dönüştürme) da sonuçları etkileyebilir.
Random State Ayarı: Model eğitimi sırasında ve veri bölünmesi (train_test_split) işlemi sırasında kullanılan random state değeri, sonuçların tutarlılığını 
etkileyebilir. Eğer siz ve video kaydı random_state için farklı değerler kullanıyorsanız, bu, farklı sonuçlara yol açabilir.
GridSearchCV Parametre Aralıkları: Parametre arama aralıklarının (örneğin, n_estimators, max_depth gibi parametreler için belirlenen değerler) videoda 
kullanılanlarla tam olarak aynı olup olmadığını kontrol edin. Farklı aralıklar, farklı sonuçlara yol açabilir.
Scikit-Learn Sürümü: Kullandığınız Scikit-Learn kütüphanesinin sürümü, videoda kullanılandan farklı olabilir. Zaman içinde, algoritmaların uygulanması ve 
varsayılan parametre değerleri değişiklik gösterebilir, bu da farklı sonuçlara yol açabilir.
Çapraz Doğrulama Stratejisi: GridSearchCV'nin cv parametresi ile belirlediğiniz çapraz doğrulama stratejisi (örneğin, kat sayısı) da sonuçları etkileyebilir. 
Videoda kullanılan çapraz doğrulama stratejisi ile sizinkinin aynı olup olmadığını kontrol edin.
Çalışma Ortamı ve Kaynak Kullanımı: Model eğitimi sırasında kullanılan donanım ve kaynaklar (CPU, bellek vb.) da sonuçları etkileyebilir, özellikle n_jobs 
parametresi ile belirlediğiniz iş parçacığı (thread) sayısı farklı ise.

Bu faktörleri göz önünde bulundurarak, kodunuzun ve veri setinizin videodakilerle tam olarak aynı olduğundan emin olun. Aynı zamanda, kütüphane sürümleri ve 
parametre ayarlarının da aynı olup olmadığını kontrol edin. Bu faktörlerden herhangi birindeki farklılık, farklı sonuçlara yol açabilir.
"""


































