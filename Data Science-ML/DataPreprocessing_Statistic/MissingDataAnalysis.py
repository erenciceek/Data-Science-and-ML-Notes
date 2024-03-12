
"""
 Test for missing data randomness
 1-Görsel Teknikler
 2-Bağımsız İki Örneklem Testi
 3-Korelasyon Testi
 4-Little'nin MCAR testi



"""

import numpy as np
import pandas as pd

V1 = np.array([1,3,6,np.NaN,7,1,np.NaN,9,15])
V2 = np.array([7,np.NaN,5,8,12,np.NaN,np.NaN,2,3])
V3 = np.array([np.NaN,12,5,6,14,7,np.NaN,2,31])
df = pd.DataFrame(
        {"V1" : V1,
         "V2" : V2,
         "V3" : V3}
)

df.isnull().sum()
df.notnull().sum()

df[df.isnull().any(axis = 1)]
df[df.notnull().all(axis = 1)]
df[df["V1"].notnull() & df["V2"].notnull()& df["V3"].notnull()]

#eksik degerlerin direk silinmesi

df.dropna()
#basit deger atama

df["V1"]
df["V1"].mean()
df["V1"].fillna(df["V1"].mean())
df["V2"].fillna(0)
df.apply(lambda x: x.fillna(x.mean()), axis = 0)

#değişkenlerdeki tam değer sayısı
df.notnull().sum()

#değişkenlerdeki eksik değer sayısı
df.isnull().sum()

#veri setindeki toplam eksik değer sayısı
df.isnull().sum().sum()

#en az bir eksik değere sahip gözlemler
df[df.isnull().any(axis=1)]
#tüm değerleri tam olan gözlemler
df[df.notnull().all(axis=1)]

# GÖRSELLEŞTİRME
import missingno as msno
msno.bar(df)
msno.matrix(df)

import seaborn as sns
df = sns.load_dataset("planets")
df.head()

df.isnull().sum()
msno.matrix(df)
msno.heatmap(df)

# SİLME YÖNTEMLERİ
df.dropna(how = "all") # bütün değişkenlerin null olduğu satırı sil
df.dropna(axis = 1) # null içeren sütunu(değişkeni) sil
df.dropna(axis = 1, how = "all") # tüm değerleri NA olan değişkenleri sil
df.dropna(axis = 1, how = "all", inplace = True) # kalıcı olarak silmek için


# DEĞER ATAMA YÖNTEMLERİ
V1 = np.array([1,3,6,np.NaN,7,1,np.NaN,9,15])
V2 = np.array([7,np.NaN,5,8,12,np.NaN,np.NaN,2,3])
V3 = np.array([np.NaN,12,5,6,14,7,np.NaN,2,31])

df = pd.DataFrame(
        {"V1" : V1,
         "V2" : V2,
         "V3" : V3}
)

#sayısal degiskenlerde atama
df["V1"].fillna(0)
df["V1"].fillna(df["V1"].mean())

#tum degiskenler icin birinci yol
df.apply(lambda x: x.fillna(x.mean()), axis = 0)

#ikinci yol
df.fillna(df.mean()[:]) # bütün değişkenlerin ortalaması baz alınır
df.fillna(df.mean()["V1":"V2"]) # v1 ve v2 sütunlarındaki null değerlere her değişkenin kendi ortalamasını at
df["V3"].fillna(df["V3"].median()) # v3 değişkenindeki na'lara v3 medyanını ata.

#ucuncu yol
df.where(pd.notna(df), df.mean(), axis = "columns") # yakaladığımış olduğu eksik değerlerin yerine ilgili değişkenin ortalamasını atar.




# KATEGORİK DEĞİŞKENLER İÇİN
V1 = np.array([1,3,6,np.NaN,7,1,np.NaN,9,15])
V2 = np.array([7,np.NaN,5,8,12,np.NaN,np.NaN,2,3])
V3 = np.array([np.NaN,12,5,6,14,7,np.NaN,2,31])
V4 = np.array(["IT","IT","IK","IK","IK","IK","IK","IT","IT"])

df = pd.DataFrame(
        {"maas" : V1,
         "V2" : V2,
         "V3" : V3,
        "departman" : V4}
)

df.groupby("departman")["maas"].mean()

df["maas"].fillna(df.groupby("departman")["maas"].transform("mean"))




#Kategorik Değişkenler için Eksik Değer Atama
V1 = np.array([1,3,6,np.NaN,7,1,np.NaN,9,15])
V4 = np.array(["IT",np.nan,"IK","IK","IK","IK","IK","IT","IT"], dtype=object)

df = pd.DataFrame(
        {"maas" : V1,
        "departman" : V4}
)

df["departman"].fillna(df["departman"].mode()[0])

df["departman"].fillna(method="bfill") # eksikliği kendinden bir sonraki değerle doldur.
df["departman"].fillna(method="ffill") # eksikliği kendinden bir önceki değerle doldur.


