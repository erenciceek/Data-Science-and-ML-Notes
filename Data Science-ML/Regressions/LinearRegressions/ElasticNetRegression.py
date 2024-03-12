# Amaç hata kareler toplamını minimize eden katsayıları bu katsayılara bir ceza uygulayarak bulmaktır. ElasticNet L1 ve L2 yaklaşımlarını birleştirir.

"""
ElasticNet regresyonu, Ridge ve Lasso regresyon tekniklerinin avantajlarını birleştiren bir lineer regresyon modelidir.
Bu yöntem, hem Ridge regresyonunun L2 ceza terimini (katsayıların karelerinin toplamı) hem de Lasso regresyonunun
L1 ceza terimini (katsayıların mutlak değerlerinin toplamı) içeren bir ceza terimi kullanır. ElasticNet, özellikle
çoklu doğrusallık sorunu olan ve birçok özellik arasında önemli olanları seçmek istediğiniz yüksek boyutlu veri kümeleri için uygundur


Değişken Seçimi ve Küçültme: ElasticNet, Lasso gibi bazı katsayıları sıfıra indirebilirken, Ridge gibi katsayıları küçülterek değişkenler
arasındaki ilişkileri koruyabilir. Bu, modelin karmaşıklığını azaltırken bilgi kaybını minimize etmeye yardımcı olur.
Çoklu Doğrusallık: Çoklu doğrusallık sorunu olan durumlarda, ElasticNet, değişkenler arasındaki ilişkileri daha iyi yönetebilir ve daha kararlı tahminler sağlayabilir.
Parametre Ayarlama:
λ1 ve λ2 parametreleri, L1 ve L2 cezalarının ağırlığını ayarlamak için kullanılır. Bu parametreler, genellikle çapraz doğrulama yoluyla en iyi performansı sağlayacak şekilde seçilir.
Yüksek Boyutlu Veriler: ElasticNet, özellik sayısının gözlem sayısından fazla olduğu durumlarda etkili bir şekilde çalışabilir ve önemli özellikleri belirleyebilir.

"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score

hit = pd.read_csv("../Hitters.csv")
df = hit.copy()
df = df.dropna()
dms = pd.get_dummies(df[['League', 'Division', 'NewLeague']])
y = df["Salary"]
X_ = df.drop(['Salary','League','Division','NewLeague'], axis=1).astype('float64')
X = pd.concat([X_, dms[['League_N','Division_W','NewLeague_N']]], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=42)


from sklearn.linear_model import ElasticNet

enet_model = ElasticNet().fit(X_train,y_train)
enet_model.coef_

#   TAHMİN
y_pred = enet_model.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))
r2_score(y_test,y_pred)

# Model Tuning

from sklearn.linear_model import ElasticNetCV
enet_cv_model = ElasticNetCV(cv = 10, random_state = 0).fit(X_train,y_train)
enet_cv_model.alpha_

# elde etmiş olduğumuz alpha değerini final modeli için kullanalım.
enet_tuned = ElasticNet(alpha = enet_cv_model.alpha_).fit(X_train,y_train)
y_pred = enet_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))



"""
Önemli Not:
Birinci konu:
"alphas" parametresi için bir liste oluşturmadık ve bunu modele bıraktık. ElasticNetCV fonksiyonu uygun olan değeri buldu.
 Eğer oluşturulmak istenirse alphas [....] şeklinde bir liste oluşturulabilir ve ElasticNetCV içerisine argüman olarak ifade edilebilir.
İkinci konu:
"I1_ratio" parametresi. Bu parametre değeri 0 olduğunda L2 cezalandırması (Ridge regresyonuna eşdeğer), 1 olduğunda ise L1 cezalandırması 
(Lasso Regresyonuna eşdeğer) yapar. 
Dolayısıyla 0-1 arasında değişimi aslında cezalandırma metodlarının göreceli etkilerini ifade eder. Uygulamamızda ikisininde etkisinin 
aynı düzeyde olmasını istediğimizden dolayı bunun ön tanımlı değerini olduğu gibi bıraktık: 0.5.
Üçüncü konu:
l1_ratio için bir değerler listesi belirlendiğinde, her bir l1_ratio değeri için farklı alphas değerleri aranır. Bu durum, L1 ve L2 cezalarının 
göreceli etkilerinin farklı kombinasyonlarını denemek ve modelin performansını en iyi şekilde optimize etmek için kullanılır. 
Yani, l1_ratio ve alphas birlikte ayarlandığında, modelin cezalandırma terimlerinin göreceli etkilerini ve ceza gücünü ince ayar 
yaparak modelin genel performansını iyileştirmek mümkündür.
"""