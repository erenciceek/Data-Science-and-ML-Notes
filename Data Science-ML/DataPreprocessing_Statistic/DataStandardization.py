import numpy as np
import pandas as pd
from sklearn import preprocessing

V1 = np.array([1, 3, 6, 5, 7])
V2 = np.array([7, 7, 5, 8, 12])
V3 = np.array([6, 12, 5, 6, 14])
df = pd.DataFrame(
        {"V1" : V1,
         "V2" : V2,
         "V3" : V3})

df = df.astype(float)

# Standardizasyon
preprocessing.scale(df)

# Normalizasyon
preprocessing.normalize(df)

# Min-Max Dönüşümü
scaler = preprocessing.MinMaxScaler(feature_range = (100,200))
scaler.fit_transform(df)



# DEĞİŞKEN DÖNÜŞÜMLERİ
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
df = sns.load_dataset("tips")
df.head()

# 0-1 Dönüşümü
lbe = LabelEncoder()
lbe.fit_transform(df["sex"])

df["yeni_sex"] = lbe.fit_transform(df["sex"])
df.head()

# 1 ve Diğerleri(0) Dönüşümü
df["day"].str.contains("Sun")
df["yeni_day"] = np.where(df["day"].str.contains("Sun"), 1, 0)
df.head()

# Çok Sınıflı Dönüşüm
lbe = LabelEncoder()
lbe.fit_transform(df["day"]) # güne göre sayılar veriyor.
# Bu dönüşüm sonucunda kategorik değişkenler numerik değerlere dönüşüyor. Ve bu öncelik olarak algılanabilir
# Örneği salı ve çarşamba arasında öncelik vs hiçbir fark yokken 2 ve 3 gibi sayılara dönüştüklerinde algoritmalar 3'ün 2'den daha büyük
# olduğunu kabul edebilirler. Hatalı sonuçlar doğurabilir. Bu nedenle one-hot encoding'e başvurulur.

# ONE HOT Dönüşümü
df_one_hot = pd.get_dummies(df, columns = ["sex"], prefix = ["sex"])
df_one_hot.head()
pd.get_dummies(df, columns = ["day"], prefix = ["day"]).head()


# Sürekli Değişkeni Kategorik Değişkene Çevirme
df = sns.load_dataset("tips")
df.head()

dff = df.select_dtypes(include=["float64", "int64"])
est = preprocessing.KBinsDiscretizer(n_bins=[3, 2, 2], encode="ordinal", strategy="quantile").fit(dff)
est.transform(dff)[0:10]


# Değişkeni indexe, indexi değişkene çevirmek
df = sns.load_dataset("tips")
df.head()
df["yeni_degisken"] = df.index
df["yeni_degisken"] = df["yeni_degisken"] + 10
df.index = df["yeni_degisken"]
df.index







