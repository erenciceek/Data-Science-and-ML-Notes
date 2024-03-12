# Multivariate outlier analysis / Local Outlier Factor
import seaborn as sns
import numpy as np
from sklearn.neighbors import LocalOutlierFactor

diamonds = sns.load_dataset("diamonds")
diamonds = diamonds.select_dtypes(["float64", "int64"])
df = diamonds.copy()
df = df.dropna()
df.head()

clf = LocalOutlierFactor(n_neighbors=20 ,contamination = 0.1)
clf.fit_predict(df)

df_scores = clf.negative_outlier_factor_


np.sort(df_scores)[0:20]
esik_deger = np.sort(df_scores)[13]


aykiri_tf = df_scores > esik_deger
type(aykiri_tf)
aykiri_tf
yeni_df = df[df_scores > esik_deger]
yeni_df
yeni_df = df[df_scores < esik_deger]

baski_deger = df[df_scores == esik_deger]
