# Temel amaç, bağımlı ve bağımsız değişken arasındaki ilişkiyi ifade eden doğrusal fonksiyonu bulmaktır.
import pandas as pd
ad = pd.read_csv("../Advertising.csv")
df = ad.copy()
df.head()
df = df.iloc[:,1:len(df)]
df.head()
df.info()
df.describe().T
df.isnull().sum()
df.corr()

import seaborn as sns
sns.pairplot(df, kind = "reg");
sns.jointplot(x = "TV" , y = "sales", data = df, kind = "reg");

# Statsmodels ile modelleme

import statsmodels.api as sm
X = df[["TV"]]
X[0:5]
X = sm.add_constant(X)
X[0:5]
y = df["sales"]
y[0:5]

lm = sm.OLS(y,X)  #modelin initialize işlemi

model = lm.fit()
model.summary()

import statsmodels.formula.api as smf
lm = smf.ols("sales ~ TV" ,df)
model = lm.fit()
model.summary()

model.params
print(model.summary().tables[1])
model.conf_int() # güven aralığı
model.f_pvalue
print("fvalue:" , "%.2f" % model.fvalue)

print("tvalue:" , "%.2f" % model.tvalues[0:1])
model.mse_model
model.rsquared
model.rsquared_adj

model.fittedvalues[0:5]  # tahmin edilen değerler
y[0:5]                   # gerçek değerler

# modelin denklemi
print("Sales = " + str("%.2f" % model.params[0]) + " + TV*" + str("%.2f" % model.params[1]))


# denklemin görselleştirilmesi
g = sns.regplot(x=df["TV"], y=df['sales'], ci=None, scatter_kws={'color': 'r', 's':9})
g.set_title("Model Denklemi: Sales = 7.03 + TV*0.05")
g.set_ylabel("Satış Sayısı")
g.set_xlabel("TV Harcamaları")
import matplotlib.pyplot as plt
plt.xlim(-10,310)
plt.ylim(bottom=0);


from sklearn.linear_model import LinearRegression

X = df[["TV"]]
y = df["sales"]
reg = LinearRegression()
model = reg.fit(X, y)
model.intercept_
model.coef_
model.score(X,y) # r-squared değeri
model.predict(X)[0:10] #tahmin edilen değerler

# TAHMİN

# Model denklemi : Sales = 7.03 + TV*0.04
# Örneğin 30 birim TV harcaması olduğunda satışların tahmini değeri ne olur ?


X = df[["TV"]]
y = df["sales"]
reg = LinearRegression()
model = reg.fit(X, y)
model.predict([[30]])

yeni_veri = [[5],[90],[200]]  # bir veri listesi üzerinden tahmin yaptırıyoruz.
model.predict(yeni_veri)



# Artıklar ve Makine Öğrenmesindeki Önemi
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

lm = smf.ols("sales ~ TV", df)
model = lm.fit()
model.summary()

mse = mean_squared_error(y, model.fittedvalues)
mse

rmse = np.sqrt(mse)
rmse

reg.predict(X)[0:10]
y[0:10]
k_t  = pd.DataFrame({"gercek_y" : y[0:10],
                     "tahmin_y" : reg.predict(X)[0:10]})
k_t
k_t["hata"] = k_t["gercek_y"] - k_t["tahmin_y"]

k_t["hata_kare"] = k_t["hata"]**2
np.sum(k_t["hata_kare"])
np.mean(k_t["hata_kare"])
np.sqrt(np.mean(k_t["hata_kare"]))

model.resid[0:10]
plt.plot(model.resid)