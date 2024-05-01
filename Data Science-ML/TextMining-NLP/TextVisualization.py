import pandas as pd

data = pd.read_csv("Data Science-ML/TextMining-NLP/train.tsv",sep="\t")
pd.set_option('display.max_columns', None)
data.head()
data.info()


# büyük-küçük dönüşümü
data["Phrase"] = data["Phrase"].apply(lambda x: " ".join(i.lower() for i in x.split()))

# Noktalama işaretleri
data["Phrase"] = data["Phrase"].str.replace("[^\w\s]","",regex=True)

# sayılar
data["Phrase"] = data["Phrase"].str.replace("\d","",regex = True)

# stopwords
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
sw = stopwords.words('english')
data['Phrase'] = data['Phrase'].apply(lambda x: " ".join(x for x in x.split() if x not in sw))


# seyreklerin silinmesi
sil = pd.Series(' '.join(data['Phrase']).split()).value_counts()[-1000:]
data['Phrase'] = data['Phrase'].apply(lambda x: " ".join(x for x in x.split() if x not in sil))


# lemmi
from textblob import Word
nltk.download('wordnet')
data['Phrase'] = data['Phrase'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

data['Phrase'].head(10)

# Terim Frekansı
tf1 = (data["Phrase"]).apply(lambda x:
                             pd.value_counts(x.split(" "))).sum(axis=0).reset_index()

tf1.columns = ["words","tf"]
tf1.head()
tf1.info()
tf1.nunique()


# barplot
a = tf1[tf1["tf"] > 1000]
a.plot.bar(x = "words", y = "tf");


# Wordcloud
import numpy as np
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt

text = data["Phrase"][0]
wordcloud = WordCloud().generate(text)
plt.imshow(wordcloud,interpolation ="bilinear")
plt.axis("off")
plt.show()

wordcloud = WordCloud(max_font_size=50, max_words = 100,
                      background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud,interpolation ="bilinear")
plt.axis("off")
plt.show()
wordcloud.to_file("kelime_bulutu.png")


# Tüm Metin
text = " ".join(i for i in data.Phrase)
wordcloud = WordCloud(max_font_size=50, max_words = 100,
                      background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud,interpolation ="bilinear")
plt.axis("off")
plt.show()



# Şablonlara göre Word Cloud

vbo_mask = np.array(Image.open("Data Science-ML/TextMining-NLP/VBO.jpg"))

wc = WordCloud(background_color="white",
               max_words=1000,
               mask=vbo_mask,
               contour_width = 3,
               contour_color = "firebrick")
wc.generate(text)
wc.to_file("vbo.png")
plt.figure()
plt.imshow(wc,interpolation ="bilinear")
plt.axis("off")
plt.show()



vbo_mask = np.array(Image.open("Data Science-ML/TextMining-NLP/VBO.jpg"))

wc = WordCloud(background_color="white",
               max_words=1000,
               mask=vbo_mask,
               contour_width = 3,
               contour_color = "firebrick")
wc.generate(text)
wc.to_file("vbo.png")
plt.figure()
plt.imshow(wc,interpolation ="bilinear")
plt.axis("off")
plt.show()













