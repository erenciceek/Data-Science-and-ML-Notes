"""
Yapay Sinir Ağları (Artificial Neural Networks - ANN), insan beyninin işleyiş biçiminden esinlenerek oluşturulan ve makine öğrenimi ile sınıflandırma,
regresyon, örüntü tanıma gibi çeşitli görevlerde kullanılan bilgisayar sistemleridir. Sınıflandırma bağlamında, ANN'ler etiketli veri setleri üzerinde
eğitilerek, girdi olarak alınan verileri belirli sınıflara atanabilir.

Bir yapay sinir ağı modeli genellikle üç ana kattan oluşur:
Girdi Katmanı (Input Layer): Modelin alacağı özellikleri (features) temsil eder. Her bir girdi düğümü (nöron), veri setindeki bir özelliğe karşılık gelir.
Gizli Katmanlar (Hidden Layers): Girdi katmanı ile çıktı katmanı arasında yer alır. Bu katmanlardaki nöronlar, girdileri işleyerek ara temsiller oluşturur
ve bu temsiller, sınıflandırma işlemi için bilgiyi soyutlamaya yardımcı olur. Bir ANN birden fazla gizli katmana sahip olabilir.
Çıktı Katmanı (Output Layer): Modelin tahminlerini veya sınıflandırma kararlarını verir. İkili sınıflandırma durumunda genellikle tek bir nöron bulunurken,
çok sınıflı sınıflandırmada her bir sınıf için bir nöron bulunur.

Bir yapay sinir ağı, girdi verisini alır ve ağırlıklar (weights) ile çarpılarak her bir nöronda toplanır. Her nöron, bir aktivasyon fonksiyonu aracılığıyla
bu toplamı işler ve bir çıktı üretir. Bu aktivasyon fonksiyonları, nöronların doğrusal olmayan ilişkileri modellemesini sağlar. Yaygın aktivasyon fonksiyonları
ReLU (Düzeltilmiş Lineer Birim), sigmoid ve tanh fonksiyonlarıdır.

Eğitim sürecinde, ANN'nin ağırlıkları, gerçek çıktılar ile tahmin edilen çıktılar arasındaki hata (loss) miktarını azaltacak şekilde güncellenir.
Bu süreç genellikle geri yayılım (backpropagation) algoritması ve gradyan inişi (gradient descent) veya türevlerini kullanarak yapılır.

ANN'nin sınıflandırmadaki adımları şöyle özetlenebilir:
Girdi Verisinin Hazırlanması: Girdi verileri, modelin beklediği formata uygun hale getirilir.
İleri Besleme (Forward Propagation): Girdi verisi ağın her katmanından sırayla geçirilir ve her katmandaki nöronların çıktıları hesaplanır.
Kayıp Hesaplama (Loss Calculation): Çıktı katmanında üretilen tahminler ve gerçek değerler arasındaki kayıp hesaplanır.
Geri Yayılım (Backpropagation): Hesaplanan kaybın her bir ağırlığa etkisi, gradyan inişi kullanılarak hesaplanır ve ağırlıklar bu gradyanlar yardımıyla güncellenir.
Optimizasyon: Ağın performansını en iyi hale getirecek ağırlıklar bulunana kadar eğitim süreci tekrarlanır.
Değerlendirme ve Tahmin: Eğitilmiş model, yeni veri üzerinde test edilerek sınıflandırma yapar.


Yapay sinir ağları genellikle karmaşık sınıflandırma problemlerinde iyi performans gösterir ve insan beyninin karmaşık karar verme yeteneğini taklit etme
potansiyeline sahiptir. Ancak, uygun ağ mimarisini ve hiperparametreleri (katman sayısı, nöron sayısı, öğrenme oranı, vb.) bulmak genellikle deneme yanılma
yoluyla yapılır ve bu süreç zaman alıcı olabilir. Ayrıca, büyük veri setlerini ve çok sayıda parametre içeren karmaşık ağları eğitmek için önemli miktarda
hesaplama gücüne ihtiyaç duyulabilir.
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
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,
                                                 random_state=42)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Model ve Tahmin
from sklearn.neural_network import MLPClassifier
mlpc = MLPClassifier().fit(X_train_scaled,y_train)
mlpc.coefs_


y_pred = mlpc.predict(X_test_scaled)
accuracy_score(y_test,y_pred)


# Model Tuning
mlpc_params = {"alpha": [0.1, 0.01, 0.02, 0.005, 0.0001,0.00001],
              "hidden_layer_sizes": [(10,10,10),
                                     (100,100,100),
                                     (100,100),
                                     (3,5),
                                     (5, 3)],
              "solver" : ["lbfgs","adam","sgd"],
              "activation": ["relu","logistic"]}


mlpc = MLPClassifier()
mlpc_cv_model = GridSearchCV(mlpc, mlpc_params,
                         cv = 10,
                         n_jobs = -1,
                         verbose = 2)

mlpc_cv_model.fit(X_train_scaled, y_train)
print("En iyi parametreler: " + str(mlpc_cv_model.best_params_))

mlpc_tuned = MLPClassifier(activation = "logistic",
                           alpha = 0.1,
                           hidden_layer_sizes = (100, 100,100),
                          solver = "adam")
mlpc_tuned.fit(X_train_scaled, y_train)

y_pred = mlpc_tuned.predict(X_test_scaled)
accuracy_score(y_test, y_pred)









