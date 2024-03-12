
"""
1- Data Cleaning
 -> Noisy Data
 -> Missing Data Analysis
 -> Outlier Analysis

2- Data Standardization
 ->0-1 Transformation (Normalization)
 ->z-skoruna dönüştürme (Standardization)
 ->Log Transformation

3- Data Reduction
 -> Observation sayısının azaltılması
 -> Variable sayısının azaltılması

4- Variable Transformation
 -> Sürekli(continuous) değişkenlerde dönüşümler
 -> Kategorik değişkenlerde dönüşümler
"""


import seaborn as sns
import pandas as pd


df = sns.load_dataset("diamonds")
df = df.select_dtypes(include=["float64", "int64"])
df = df.dropna()
df.head()

df_table = df["table"]
df_table.head()


Q1 = df_table.quantile(0.25)
Q3 = df_table.quantile(0.75)
IQR = Q3 - Q1
alt_sinir = Q1 - 1.5*IQR
ust_sinir = Q3 + 1.5*IQR
((df_table < alt_sinir) | (df_table > ust_sinir))
aykiri_tf = (df_table < alt_sinir)
aykiri_tf.head()


df_table = pd.DataFrame(df_table)
t_df = df_table[~((df_table < alt_sinir) | (df_table > ust_sinir)).any(axis=1)]

# outlier değerleri yerine veri setinin ortalamasını atamak istiyorsak:
df_table[aykiri_tf] = df_table.mean()

# baskılama yapıyorsak (yani üst sınırı aşan outlier değerlerini üst sınır ile güncelleyerek
# bu örnekte alt sınır versiyonu var
df_table[aykiri_tf] = alt_sinir






