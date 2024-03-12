import pandas as pd
import seaborn as sns
from scipy import stats

# H0 : p (p value değil) = 0 --> değişkenler arasında anlamlı bir ilişki yoktur.
# H1 : p != 0 --> değişkenler arasında anlamlı bir ilişki vardır.
# p --> ro
# Varsayımlar :
# Normallik varsayımı
# eğer varsayım sağlanıyorsa --> Pearson Korelasyon Katsayısı
# eğer varsayım sağlanmıyorsa --> Spearman Korelasyon Katsayısı


# Bahşiş ile ödenen hesap arasında korelasyon var mı ?

tips = sns.load_dataset("tips")
df = tips.copy()
df.head()
# total_bill değişkeni hem bahşiş hem hesabı içerir. Onları ayırmak gerekir.

df["total_bill"] = df["total_bill"] - df["tip"]
df.head()
df.plot.scatter("tip","total_bill");

# Varsayım Kontrolü
test_istatistigi, pvalue = stats.shapiro(df["tip"])
print('Test İstatistiği = %.4f, p-değeri = %.4f' % (test_istatistigi, pvalue))

test_istatistigi, pvalue = stats.shapiro(df["total_bill"])
print('Test İstatistiği = %.4f, p-değeri = %.4f' % (test_istatistigi, pvalue))

# p değerleri 0.05'ten küçük çıktığı için H0 reddedilir.
# Yani sample distribution ile population distribution arasında anlamlı bir bağ yoktur hipotezi(h0) reddedilir.
# Normal bir dağılım yoktur.

# Öncelikli olarak normallik varsayımı sağlanıyormuş gibi yapıp parametrik çözümü yapacağız :
df["tip"].corr(df["total_bill"]) # default olarak pearson tanımlı

# normallik sağlanmadığı için :
df["tip"].corr(df["total_bill"], method="spearman")

# Değişkenlerin arasında pozitif yönlü orta şiddette bir ilişki var.
# Anlamlılığı sorgulama:
from scipy.stats.stats import pearsonr
test_istatistigi, pvalue = pearsonr(df["tip"], df["total_bill"])

print('Korelasyon Katsayısı = %.4f, p-değeri = %.4f' % (test_istatistigi, pvalue))
# p değeri 0.05 geldiği için aralarında anlamlı bir ilişki yoktur diyen h0 hipotezini reddediyoruz.

# NONPARAMETRİK Korelasyon Testi
from scipy.stats import stats
test_istatistigi, pvalue = stats.spearmanr(df["tip"],df["total_bill"])  # bu örnek için en güveniliri
print('Korelasyon Katsayısı = %.4f, p-değeri = %.4f' % (test_istatistigi, pvalue))
# p değeri 0.05'ten küçük çıktı.  Değişkenler arasında anlamlı bir ilişki vardır.





# spearmanr fonksiyonuna bir alternatif :
test_istatistigi, pvalue = stats.kendalltau(df["tip"], df["total_bill"])

print('Korelasyon Katsayısı = %.4f, p-değeri = %.4f' % (test_istatistigi, pvalue))