import pandas as pd
import numpy as np
from spicy import stats

A = pd.DataFrame([30, 27, 21, 27, 29, 30, 20, 20, 27, 32, 35, 22, 24, 23, 25, 27, 23, 27, 23, 25, 21, 18, 24, 26, 33, 26, 27, 28, 19, 25])
B = pd.DataFrame([37, 39, 31, 31, 34, 38, 30, 36, 29, 28, 38, 28, 37, 37, 30, 32, 31, 31, 27, 32, 33, 33, 33, 31, 32, 33, 26, 32, 33, 29])

# DATAFRAME'de tutma işlemi 1
A_B = pd.concat([A,B], axis = 1)
A_B.columns = ["A","B"]
A_B.head()

# DATAFRAME'de tutma yöntemi 2
# A ve A'nın grubu
GRUP_A = np.arange(len(A))
GRUP_A = pd.DataFrame(GRUP_A)
GRUP_A[:] = "A"
A = pd.concat([A, GRUP_A], axis = 1)

# B ve B'nin grubu
GRUP_B = np.arange(len(B))
GRUP_B = pd.DataFrame(GRUP_B)
GRUP_B[:] = "B"
B = pd.concat([B, GRUP_B], axis = 1)


AB = pd.concat([A , B])
AB.columns = ["gelir", "GRUP"]
print(AB.head())
print(AB.tail())

# INDEPENDENT TWO SAMPLE TEST için varsayım kontrolleri : shapiro testi (normalliği ispatlar)
# Normallik Varsayımı :
# H0 : sample distribution ile population distribution arasında anlamlı bir bağ yoktur.
# H1 : sample distribution ile population distribution arasında anlamlı bir bağ vardır.

stats.shapiro(A_B.A)
stats.shapiro(A_B.B)

# çıkan iki p değeri için de 0.05 ten büyüklük sağlanmış yani H0 hipotezi reddedilememiştir.
# yani sample ile population arasında anlamlı bir bağ yoktur.
# Normallik varsayımı geçerli bir diğerine geçiyoruz.


# Varyans Homojenliği :
# H0 : Varyanslar Homojendir.
# H1 : Varyanslar Homojen değildir.

stats.levene(A_B.A, A_B.B)

# p value 0.05ten (alpha'dan) büyük çıktığı için H0 reddedilememiştir.
# İki varsayım da sağlandığına göre artık uygulama adımlarına geçilebilir.

# UYGULAMA
test_istatistigi, p_value = stats.ttest_ind(A_B["A"], A_B["B"], equal_var=True)
print("Test İstatistiği = %.4f, p-değeri = %.4f" % (test_istatistigi,p_value))

# p value 0.05'ten küçük çıkmıştır. Bu nedenle H0 hipotezi reddedilir.
# Örnekte H0 : ML modeli uygulandıktan sonra sağlanan gelirler eski gelirler ile eşittir.
# Yani eşit olmadığı ortaya çıkmıştır.


# NONPARAMETRİK BAĞIMSIZ 2 ÖRNEKLEM TESTİ:
# eğer varsayım testleri sonuçları negatif olursa :

stats.mannwhitneyu(A_B["A"], A_B["B"])
# p değeri yine 0.05'ten küçük çıkmıştır. Bu nedenle H0 hipotezi reddedilir.
# Yani ML modeli uygulandıktan sonra sağlanan gelirler eski gelirler ile eşit değildir.
