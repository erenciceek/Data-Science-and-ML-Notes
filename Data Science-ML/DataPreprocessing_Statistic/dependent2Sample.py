import pandas as pd
import numpy as np
import seaborn as sns
from spicy import stats

oncesi = pd.DataFrame([123, 119, 119, 116, 123,123,121,120,117,118,121,121,123,119,
            121,118,124,121,125,115,115,119,118,121,117,117,120,120,
            121,117,118,117,123,118,124,121,115,118,125,115])

sonrasi = pd.DataFrame([118,127,122,132,129,123,129,132,128,130,128,138,140,130,
             134,134,124,140,134,129,129,138,134,124,122,126,133,127,
             130,130,130,132,117,130,125,129,133,120,127,123])

# DATAFRAME'DE düzenleme ve birleştirme


# Birinci Veri Seti
AYRIK = pd.concat([oncesi, sonrasi], axis=1)
AYRIK.columns = ["ONCESI", "SONRASI"]
print("'AYRIK' Veri Seti: \n\n ", AYRIK.head() , "\n\n")

# İkinci Veri Seti
GRUP_ONCESI = np.arange(len(oncesi))
GRUP_ONCESI = pd.DataFrame(GRUP_ONCESI)
GRUP_ONCESI[:] = "ONCESI"
A = pd.concat([oncesi, GRUP_ONCESI], axis=1)

GRUP_SONRASI = np.arange(len(sonrasi))
GRUP_SONRASI = pd.DataFrame(GRUP_SONRASI)
GRUP_SONRASI[:] = "SONRASI"
B = pd.concat([sonrasi, GRUP_SONRASI], axis=1)

AB = pd.concat([A,B])
AB.columns = ["PERFORMANS", "ONCESI_SONRASI"]
print("'AB' Veri Seti: \n\n", AB.head(), "\n", AB.tail(), "\n\n")

sns.boxplot(x = "ONCESI_SONRASI", y = "PERFORMANS", data = AB)

# VARSAYIM KONTROLLERI
# 1 - Normallik kontrolü için shapiro testi:


stats.shapiro(AYRIK.ONCESI)
stats.shapiro(AYRIK.SONRASI)
# iki p değeri de 0.05 ten büyük olduğu için H0 hipotezi reddedilemez.
# yani sample ile population distributions arasında anlamlı bir bağ yoktur, normal dağılmaktadır.

# 2 - Varyans Homojenligi

stats.levene(AYRIK.ONCESI, AYRIK.SONRASI)
# P VALUE 0.05'ten küçük. Bu nedenle H0 reddedilir. Yani Homojenlik sağlanmamamaktadır.
# Varyans homojenliği varsayımı geçerli olmadığında bazen gözardı edilebilir.
# Bu örnek için öyle yapılacaktır.

# UYGULAMA

stats.ttest_rel(AYRIK.ONCESI, AYRIK.SONRASI)
# P-value 0.05'ten küçük olduğu için asıl H0 hipotezi reddedilir.
# örnekteki H0 : eğitimler sonrası katma değer 0'dır.
# yani çalışanlara eğitim verildiğinde çalışan performansları değişmiştir,katma değer vardır.

# NONPARAMETRİK BAĞIMLI İKİ ÖRNEKLEM TESTİ
# varsayımlar sağlanmadığı durumda :
stats.wilcoxon(AYRIK.ONCESI, AYRIK.SONRASI)
# p değerine göre yine H0 hipotezi reddedilir . Yani katma değer vardır.


# AB TESTLERİ DE ÖNEMLİ , TEKRAR ET.
