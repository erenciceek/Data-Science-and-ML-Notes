"""
Metin Madenciliği
Metin madenciliği, büyük metin koleksiyonlarından anlamlı bilgi çıkarmak için kullanılan yöntemler ve teknikler bütünüdür. Genellikle veri madenciliğinin
bir alt dalı olarak kabul edilir. Metin madenciliği uygulamaları arasında özetleme, sınıflandırma, duygu analizi, konu modelleme gibi işlemler bulunur.
Bu süreç, genellikle metni önce işlenebilir bir forma dönüştürmeyi (tokenizasyon, kök bulma, lemmatizasyon gibi) içerir ve ardından çeşitli algoritmalar
kullanarak yararlı bilgileri çıkarmayı amaçlar.

Doğal Dil İşleme (NLP)
Doğal dil işleme, insan dilinin anlaşılması ve üretilmesi için bilgisayar algoritmalarının kullanılmasıdır. NLP, metin madenciliğinin ötesine geçerek dilin
daha derin yapısını ve anlamını anlamayı amaçlar. NLP'nin temel uygulamaları arasında dil çeviri, otomatik özetleme, adlandırılmış varlık tanıma, dil
modelleme ve konuşma tanıma bulunur. Modern NLP sistemleri, genellikle makine öğrenimi modellerini, özellikle derin öğrenme yöntemlerini kullanarak dilin
daha karmaşık özelliklerini modellemeye çalışır.

İlişkiler ve Uygulamalar
Her iki alan da, veriye dayalı karar verme süreçlerinde önemli bir rol oynar ve iş dünyası, sağlık, hukuk, medya ve eğitim gibi birçok sektörde uygulanabilir.
Örneğin, bir şirket müşteri geri bildirimlerini analiz ederek ürünlerini geliştirebilir veya bir sağlık kurumu hasta kayıtlarını inceleyerek tedavi
süreçlerini optimize edebilir.

Son yıllarda, büyük dil modelleri (örneğin, GPT serisi, BERT) ve onların uygulamaları, NLP'nin sınırlarını önemli ölçüde genişletti ve pek çok yenilikçi
uygulamanın kapılarını açtı. Metin madenciliği ve NLP, sürekli gelişen teknolojilerle birlikte, veri odaklı bir dünyada giderek daha fazla önem kazanmaktadır.
"""

import pandas as pd

# TEXT PREPROCESSING

metin = """
A Scandal in Bohemia! 01
The Red-headed League,2
A Case, of Identity 33
The Boscombe Valley Mystery4
The Five Orange Pips1
The Man with? the Twisted Lip
The Adventure of the Blue Carbuncle
The Adventure of the Speckled Band
The Adventure of the Engineer's Thumb
The Adventure of the Noble Bachelor
The Adventure of the Beryl Coronet
The Adventure of the Copper Beeches"""

metin


# String'ten pandas dataframe'e
metin.split("\n")
v_metin = metin.split("\n")
v = pd.Series(v_metin)
metin_vektoru = v[1:len(v)]
mdf = pd.DataFrame(metin_vektoru,columns=["hikayeler"])

# Büyük-küçük harf
d_mdf = mdf.copy()
d_mdf

list1 = [1,2,3]
str1 = "".join(str(i) for i in list1)

d_mdf["hikayeler"].apply(lambda x: " ".join(i.lower() for i in x.split()))
d_mdf = d_mdf["hikayeler"].apply(lambda x: " ".join(i.lower() for i in x.split()))

 # d_mdf = pd.DataFrame(d_mdf,columns=["hikayeler"])
 
# Noktalama işaretlerinin silinmesi
d_mdf = d_mdf.str.replace(r"[^\w\s]", "",regex = True)

# Sayıların silinmesi
d_mdf = d_mdf.str.replace("\d","",regex = True)

# Stopwords silinmesi
d_mdf = pd.DataFrame(d_mdf,columns=["hikayeler"])
import nltk
nltk.download("stopwords")

from nltk.corpus import stopwords
sw = stopwords.words("english")

d_mdf = d_mdf["hikayeler"].apply(lambda x: " ".join(i for i in x.split() if i not in sw))

# Az geçen kelimelerin silinmesi
type(d_mdf)
d_mdf = pd.DataFrame(d_mdf,columns=["hikayeler"])
pd.Series(" ".join(d_mdf["hikayeler"]).split()).value_counts()
sil = pd.Series(" ".join(d_mdf["hikayeler"]).split()).value_counts()[-3:]

d_mdf = d_mdf["hikayeler"].apply(lambda x: " ".join(i for i in x.split() if i not in sil))


# Tokenization
nltk.download("punkt")
import textblob
from textblob import TextBlob

d_mdf = pd.DataFrame(d_mdf,columns=["hikayeler"])
TextBlob(d_mdf["hikayeler"][1]).words

d_mdf["hikayeler"].apply(lambda x: TextBlob(x).words)

# Stemming # kelimeleri köklerine indirger.
d_mdf = pd.DataFrame(d_mdf,columns=["hikayeler"])
from nltk.stem import PorterStemmer
st = PorterStemmer()
d_mdf["hikayeler"].apply(lambda x: " ".join([st.stem(i) for i in x.split()]))


# Lemmatization
from textblob import Word
nltk.download("wordnet")
d_mdf = d_mdf["hikayeler"].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
d_mdf = pd.DataFrame(d_mdf,columns=["hikayeler"])


mdf["hikayeler"][0:5]
d_mdf["hikayeler"][0:5]

#### NLP UYGULAMALARI ####

# N-Gram
"""
N-gram, metindeki kelimelerin veya karakterlerin ardışık gruplarını ifade eder. "N", grubun içereceği öğe sayısını belirtir.
Örneğin, bir "bigram" (2-gram) ardışık iki kelimeyi, bir "trigram" (3-gram) ise ardışık üç kelimeyi temsil eder. N-gramlar,
metin üzerinde istatistiksel dil modelleme yapmak için sıkça kullanılır. Özellikle, metin sınıflandırma, dil modelleme ve
metin benzerliği gibi görevlerde kullanışlıdır.
"""


a = """Bu örneği anlaşılabilmesi için daha uzun bir metin üzerinden göstereceğim.
N-gram'lar birlikte kullanılan kelimelerin kombinasyolarını gösterir"""
TextBlob(a).ngrams(3)



# Part of Speech Tagging (POS) - Kelime Türü Etiketleme
"""
Part of Speech Tagging, metindeki her kelimenin dil bilgisine göre kategorize edilmesi işlemidir (örneğin, isim, fiil, sıfat). 
Bu analiz, metinlerin yapısal olarak daha derinlemesine incelenmesini sağlar ve dilin anlamını ve bağlamını daha iyi anlamak 
için kullanılır. POS tagging, dil yapısını analiz etme, anlam belirsizliğini giderme ve dil öğrenme uygulamalarında önemli bir 
rol oynar.
"""
nltk.download("averaged_perceptron_tagger")
TextBlob(d_mdf["hikayeler"][3]).tags
d_mdf["hikayeler"].apply(lambda x: TextBlob(x).tags)


pos = d_mdf["hikayeler"].apply(lambda x: TextBlob(x).tags)
pos



# Chunking (Shallow Parsing) - Yüzeysel Ayrıştırma
"""
Chunking, metindeki kelimeleri daha büyük birimler olan "chunk"lara (kelime gruplarına) ayırma işlemidir. Bu gruplar genellikle dilbilgisel yapıları 
temsil eder (örneğin, isim grupları veya fiil grupları). Chunking, derin ayrıştırmanın (full parsing) aksine daha yüzeysel bir analiz sağlar ve 
metindeki anlam birimlerini belirlemek için kullanılır. Bu teknik, metin özetleme, anlam analizi ve dil modelleme görevlerinde kullanılabilir.
"""


cumle = "R and Python are useful data science tools for the new or old data scientists who eager to do efficent data science task"
pos = TextBlob(cumle).tags

reg_exp = "NP: {<DT>?<JJ>*<NN>}"
rp = nltk.RegexpParser(reg_exp)
sonuclar = rp.parse(pos)
sonuclar
print(sonuclar)
sonuclar.draw()



# Named Entity Recognition (NER) - Adlandırılmış Varlık Tanıma
"""
Named Entity Recognition, metinde geçen adlandırılmış varlıkları (kişi isimleri, organizasyonlar, yerler, tarihler gibi) tanıma ve sınıflandırma işlemidir. 
NER, büyük veri setlerinden özel bilgileri otomatik olarak çıkarmak için kullanılır ve bilgi yönetimi, müşteri ilişkileri yönetimi, haberlerden bilgi 
çıkarma gibi alanlarda değerlidir.
"""

from nltk import word_tokenize, pos_tag, ne_chunk
nltk.download('maxent_ne_chunker')
nltk.download('words')
cumle = "Hadley is creative people who work for R Studio AND he attented conference at Newyork last year"
print(ne_chunk(pos_tag(word_tokenize(cumle))))



# Matematiksel İşlemler ve Basit Özellik Çıkarımı
o_df = d_mdf.copy()
o_df["hikayeler"].str.len()
o_df["harf_sayisi"] = o_df["hikayeler"].str.len()


a = "scandal in a bohemia"
a.split()
len(a.split())
o_df.iloc[0:1,0:1].split() # çalışmaz
o_df["kelime_sayisi"] = o_df["hikayeler"].apply(lambda x:len(str(x).split(" ")))


o_df["ozel_karakter_sayisi"] = o_df["hikayeler"].apply(lambda x: len([x for x in x.split() if x.startswith("adventure")]))



o_df["sayi_sayisi"] = mdf["hikayeler"].apply(lambda x: len([x for x in x.split() if x.isdigit()]))









