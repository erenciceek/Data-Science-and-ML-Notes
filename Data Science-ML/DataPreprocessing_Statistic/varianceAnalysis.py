import pandas as pd
import numpy as np
from spicy import stats

# İki ya da daha fazla grup ortalaması arasında istatistiksel olarak
# anlamlı farklılık olup olmadığı öğrenilmek istenildiğinde kullanılır.

# H0 : M1 = M2 = M3
# H1 : Eşit değillerdir (en az birisi farklıdır)


# Varsayımlar :
# Gözlemlerin birbirinden bağımsız olması
# Normal Dağılım
# Varyans Homojenliği

# ÖRNEK : Anasayfa içerik stratejisi belirlemek
# Bir web sitesi için başarı kriterleri : ort ziyaret süresi, hemen çıkış oranı vb
# 3 adet strateji
# A: Doğal şekilde(olduğu gibi)  B:Yönlendirici  C:İlgi Çekici
# H0: Tüm stratejilerin web sitesinin başarılı olmasına etkisi aynıdır.

A = pd.DataFrame([28,33,30,29,28,29,27,31,30,32,28,33,25,29,27,31,31,30,31,34,30,32,31,34,28,32,31,28,33,29])

B = pd.DataFrame([31,32,30,30,33,32,34,27,36,30,31,30,38,29,30,34,34,31,35,35,33,30,28,29,26,37,31,28,34,33])

C = pd.DataFrame([40,33,38,41,42,43,38,35,39,39,36,34,35,40,38,36,39,36,33,35,38,35,40,40,39,38,38,43,40,42])

dfs = [A, B, C]
# DATAFRAME'de düzenlenmesi ve birleştirilmesi
ABC = pd.concat(dfs, axis = 1)
ABC.columns = ["GRUP_A","GRUP_B","GRUP_C"]
ABC.head()


# Varsayım kontrolü : normal distribution

stats.shapiro(ABC["GRUP_A"])
stats.shapiro(ABC["GRUP_B"])
stats.shapiro(ABC["GRUP_C"])
# P değerleri 3ü için de 0.05'ten küçük olmadığı için h0 hipotezi reddedilemez
# yani sample ile population dağılımları arasında anlamlı bir bağ yoktur, dağılım normaldir.

# Varsayım kontrolü : varyans homojenliği
stats.levene(ABC["GRUP_A"], ABC["GRUP_B"], ABC["GRUP_C"])
# p-value 0.05'ten büyük bu nedenle varyanslar homojendir diyen hipotezin reddedilemediği anlaşılıyor.
# Yani varsayım geçerli


# UYGULAMA
stats.f_oneway(ABC["GRUP_A"], ABC["GRUP_B"], ABC["GRUP_C"])
# p-value 0.05'ten küçük bu nedenle problemin hipotezi reddedilir.
# Yani en az bir grubun ortalama değeri diğerinden farklı olacaktır.


ABC.describe().T
# C grubu daha çok katkı sağlayacaktır.

# NONPARAMETRİK HİPOTEZ TESTİ

stats.kruskal(ABC["GRUP_A"], ABC["GRUP_B"], ABC["GRUP_C"])
# p-value 0.05'ten küçük, H0 hipotezi reddedilir. En az biri farklıdır.