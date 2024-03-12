# Doğrusal Olmayan Regresyon Modelleri :

"""
1- K-En yakın Komşu (KNN)
2- Destek Vektör Regresyonu(SVR)
3- Çok Katmanlı Algılayıcılar(ANN)
4- Classification and Regression Tress(CART)
5- Bagging(Bootstrap Aggregation)
6- Random Forests (RF)
7- Gradient Boosting Machines (GBM)
8- Extreme Gradient Boosting (XGBoost)
9- LightGBM
10- CatBoost

"""

"""
KNN (K-En Yakın Komşu) regresyonu, hem sınıflandırma hem de regresyon problemlerinde kullanılan basit, ancak güçlü bir makine 
öğrenmesi algoritmasıdır. KNN regresyonunda, bir noktanın değeri, ona en yakın k komşunun değerlerinin basit bir ortalaması alınarak 
tahmin edilir. Bu yöntem, veriler arasındaki yerel ilişkileri yakalamak için özellikle yararlıdır ve modelin eğitimi esnasında 
varsayılan bir fonksiyonel form gerektirmez.

KNN Regresyonunun Çalışma Prensibi
Komşu Sayısının Belirlenmesi (k): KNN algoritmasının en önemli parametresi, komşu sayısı olan k'dır. 
k, bir noktanın değerini tahmin ederken dikkate alınacak en yakın komşu sayısını belirler.

Mesafe Ölçütü: Her bir veri noktası arasındaki mesafe hesaplanır. Bu mesafe genellikle Öklid mesafesi olarak hesaplanır, ancak diğer mesafe ölçütleri de kullanılabilir.

En Yakın k Komşunun Bulunması: Tahmin edilecek noktanın, tüm eğitim seti içindeki noktalara olan mesafeleri hesaplanır ve en yakın k komşu seçilir.
Tahmin: Seçilen k komşunun çıktı değerlerinin (regresyon durumunda genellikle aritmetik ortalaması) hesaplanmasıyla tahmin yapılır.

"""


import numpy as np
import pandas as pd
from sklearn.model_selection import train
































