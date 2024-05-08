from sklearn import model_selection, preprocessing, linear_model, naive_bayes
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import ensemble

import xgboost



from warnings import filterwarnings
filterwarnings('ignore')


import pandas as pd
data = pd.read_csv("Data Science-ML/TextMining-NLP/train.tsv",sep="\t")
data.head()

data["Sentiment"].replace(0,value = "negatif",inplace = True)
data["Sentiment"].replace(1,value = "negatif",inplace = True)

data["Sentiment"].replace(3,value = "pozitif",inplace=True)
data["Sentiment"].replace(4,value = "pozitif",inplace=True)

data = data[(data.Sentiment == "negatif") | (data.Sentiment == "pozitif")]

data.groupby("Sentiment").count()

df = pd.DataFrame()
df["text"] = data["Phrase"]
df["label"] = data["Sentiment"]
df.head()

### TEXT PREPROCESSING SECTION

# büyük-küçük dönüşümü
df['text'] = df['text'].apply(lambda x: " ".join(i.lower() for i in x.split()))

# noktalama işaretleri
df["text"] = df["text"].str.replace("[^\w\s]","",regex=True)

#sayılar
df["text"] = df["text"].str.replace("\d","",regex=True)

# stopwords
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
sw = stopwords.words('english')
df["text"] = df["text"].apply(lambda x: " ".join(i for i in x.split() if i not in sw))

# seyreklerin silinmesi
sil = pd.Series(" ".join(df["text"]).split()).value_counts()[-1000:]
df["text"] = df["text"].apply(lambda x: " ".join(i for i in x.split() if i not in sil))

# lemmi
from textblob import Word
# nltk.download("wordnet")
df["text"] = df["text"].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))



### FEATURE ENGINEERING SECTION
"""
Feature Engineering, makine öğrenimi modellerinde kullanılacak verileri hazırlama sürecidir. Bu süreç, ham verileri alıp onları modelin daha iyi çalışmasını 
sağlayacak şekilde işlemeyi içerir. Temel olarak, veri setindeki bilgileri modelin anlayabileceği ve üzerinden öğrenme yapabileceği formata dönüştürmekle 
ilgilidir. İyi yapılmış bir feature engineering, modelin performansını önemli ölçüde artırabilir.
Feature Engineering'in Temel Adımları:
1-Feature Extraction (Özellik Çıkarımı):Ham veriden yararlı özelliklerin çıkarılması işlemidir.
2-Feature Creation (Özellik Oluşturma): Mevcut verilerden yeni özellikler türetme işlemidir. 
3-Feature Transformation (Özellik Dönüşümü): Verileri modelleme için daha uygun bir forma sokmak adına özelliklerin dönüştürülmesi işlemidir. Bu, 
                                             normalizasyon veya log dönüşümü gibi matematiksel işlemleri içerebilir.
4-Feature Selection (Özellik Seçimi): En etkili özellikleri seçme ve az önemli veya alakasız olanları çıkarma sürecidir. 
5-Feature Scaling (Özellik Ölçeklendirme): Özellikler arasında farklı ölçeklerin olması durumunda, bu ölçekleri birbirine yakınlaştırmak için yapılan işlemdir. 

3 Popüler Text Feature Extraction yöntemi (NLP özelinde):
1-Count Vectors
2-TF-IDF Vectors (words,characters,n-grams)
3-Word Embeddings

TF(t) = (Bir t teriminin bir dökümanda gözlenme frekansı) / (dökümandaki toplam terim sayısı) 
IDF(t) = log_e(Toplam döküman sayısı / içinde t terimi olan belge sayısı)
"""

df.head()


### TEST-TRAIN
train_x, test_x, train_y,test_y = model_selection.train_test_split(df["text"],
                                                                   df["label"],
                                                                   random_state=1)
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
test_y = encoder.fit_transform(test_y)

# Count Vectors
"""
Count Vector, bir metin koleksiyonundaki (corpus) her kelimenin her belgede kaç kez geçtiğini gösteren bir matristir. Bu yöntem, basitçe kelimelerin 
sıklığını sayar ve bir belge-kelime matrisi şeklinde ifade eder. Her sütun bir kelimeyi ve her satır bir belgeyi temsil eder. Bir belgede bir kelime 
ne kadar sık geçiyorsa, o kelimenin o belge için o kadar önemli olduğu varsayılır. Ancak, bu yöntem kelimenin belgedeki önemini dikkate almaz, yalnızca 
frekansına bakar.
"""
vectorizer = CountVectorizer()
vectorizer.fit(train_x)
vectorizer.get_params()

x_train_count = vectorizer.transform(train_x)
x_test_count = vectorizer.transform(test_x)

vectorizer.get_feature_names_out()[0:5]
x_train_count.toarray()

# TF-IDF Vectors
"""
TF-IDF, bir kelimenin bir belgedeki önemini, o kelimenin tüm belgeler arasındaki dağılımını dikkate alarak hesaplayan bir yöntemdir. Bu yöntem iki farklı 
metriği birleştirir:

Term Frequency (TF): Bir kelimenin bir belgedeki frekansı. Yani bir kelimenin belgedeki toplam kelime sayısına oranı.
Inverse Document Frequency (IDF): Bir kelimenin nadirliğini ölçen bir metrik. Logaritmik olarak hesaplanır ve tüm belgeler içinde yalnızca az sayıda 
                                  belgede geçen kelimeler yüksek IDF değeri alır.
                                  
TF-IDF, hem kelime frekansını hem de kelimenin corpus içindeki nadirliğini dikkate alarak, belgeler arasında farklılıkları belirlemek için kullanılır. 
Bu vektörleme yöntemi kelime, karakter ve n-gram düzeylerinde uygulanabilir.
"""
tf_idf_word_vectorizer = TfidfVectorizer()
tf_idf_word_vectorizer.fit(train_x)
tf_idf_word_vectorizer.get_params()

x_train_tf_idf_word = tf_idf_word_vectorizer.transform(train_x)
x_test_tf_idf_word = tf_idf_word_vectorizer.transform(test_x)

tf_idf_word_vectorizer.get_feature_names_out()[0:5]
x_train_tf_idf_word.toarray()



# ngram level tf-idf
tf_idf_ngram_vectorizer = TfidfVectorizer(ngram_range=(2,3))
tf_idf_ngram_vectorizer.fit(train_x)
tf_idf_ngram_vectorizer.get_params()

x_train_tf_idf_ngram = tf_idf_ngram_vectorizer.transform(train_x)
x_test_tf_idf_ngram = tf_idf_ngram_vectorizer.transform(test_x)


# characters level tf-idf
tf_idf_chars_vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2,3))
tf_idf_chars_vectorizer.fit(train_x)
tf_idf_chars_vectorizer.get_params()

x_train_tf_idf_chars = tf_idf_chars_vectorizer.transform(train_x)
x_test_tf_idf_chars = tf_idf_chars_vectorizer.transform(test_x)

# Veriye 4 farklı özellik çıkarımı yaklaşımı uygulandı. Şimdi bunları modeller ile test edeceğiz.




### ML İLE SENTİMENT SINIFLANDIRMASI


## Logistic Regression
loj = linear_model.LogisticRegression()
loj_model = loj.fit(x_train_count,train_y)
accuracy = model_selection.cross_val_score(loj_model,
                                           x_test_count,
                                           test_y,
                                           cv=10).mean()
print("Count Vectors Doğruluk Oranı :",accuracy)


loj = linear_model.LogisticRegression()
loj_model = loj.fit(x_train_tf_idf_word,train_y)
accuracy = model_selection.cross_val_score(loj_model,
                                           x_test_tf_idf_word,
                                           test_y,
                                           cv = 10).mean()
print("Word-Level TF-IDF Doğruluk Oranı:", accuracy)


loj = linear_model.LogisticRegression()
loj_model = loj.fit(x_train_tf_idf_ngram,train_y)
accuracy = model_selection.cross_val_score(loj_model,
                                           x_test_tf_idf_ngram,
                                           test_y,
                                           cv = 10).mean()
print("N-GRAM TF-IDF Doğruluk Oranı:", accuracy)


loj = linear_model.LogisticRegression()
loj_model = loj.fit(x_train_tf_idf_chars,train_y)
accuracy = model_selection.cross_val_score(loj_model,
                                           x_test_tf_idf_chars,
                                           test_y,
                                           cv = 10).mean()
print("CHARLEVEL Doğruluk Oranı:", accuracy)



## Naive Bayes

nb = naive_bayes.MultinomialNB()
nb_model = nb.fit(x_train_count,train_y)
accuracy = model_selection.cross_val_score(nb_model,
                                           x_test_count,
                                           test_y,
                                           cv = 10).mean()
print("Count Vectors Doğruluk Oranı:", accuracy)


nb = naive_bayes.MultinomialNB()
nb_model = nb.fit(x_train_tf_idf_word,train_y)
accuracy = model_selection.cross_val_score(nb_model,
                                           x_test_tf_idf_word,
                                           test_y,
                                           cv = 10).mean()
print("Word-Level TF-IDF Doğruluk Oranı:", accuracy)


nb = naive_bayes.MultinomialNB()
nb_model = nb.fit(x_train_tf_idf_ngram,train_y)
accuracy = model_selection.cross_val_score(nb_model,
                                           x_test_tf_idf_ngram,
                                           test_y,
                                           cv = 10).mean()
print("N-GRAM TF-IDF Doğruluk Oranı:", accuracy)



nb = naive_bayes.MultinomialNB()
nb_model = nb.fit(x_train_tf_idf_chars,train_y)
accuracy = model_selection.cross_val_score(nb_model,
                                           x_test_tf_idf_chars,
                                           test_y,
                                           cv = 10).mean()
print("CHARLEVEL Doğruluk Oranı:", accuracy)



## Random Forest
rf = ensemble.RandomForestClassifier()
rf_model = rf.fit(x_train_count,train_y)
accuracy = model_selection.cross_val_score(rf_model,
                                           x_test_count,
                                           test_y,
                                           cv = 10).mean()
print("Count Vectors Doğruluk Oranı:", accuracy)


rf = ensemble.RandomForestClassifier()
rf_model = rf.fit(x_train_tf_idf_word,train_y)
accuracy = model_selection.cross_val_score(rf_model,
                                           x_test_tf_idf_word,
                                           test_y,
                                           cv = 10).mean()
print("Word-Level TF-IDF Doğruluk Oranı:", accuracy)



rf = ensemble.RandomForestClassifier()
rf_model = loj.fit(x_train_tf_idf_ngram,train_y)
accuracy = model_selection.cross_val_score(rf_model,
                                           x_test_tf_idf_ngram,
                                           test_y,
                                           cv = 10).mean()
print("N-GRAM TF-IDF Doğruluk Oranı:", accuracy)



rf = ensemble.RandomForestClassifier()
rf_model = loj.fit(x_train_tf_idf_chars,train_y)
accuracy = model_selection.cross_val_score(rf_model,
                                           x_test_tf_idf_chars,
                                           test_y,
                                           cv = 10).mean()
print("CHARLEVEL Doğruluk Oranı:", accuracy)



## XGBoost

xgb = xgboost.XGBClassifier()
xgb_model = xgb.fit(x_train_count,train_y)
accuracy = model_selection.cross_val_score(xgb_model,
                                           x_test_count,
                                           test_y,
                                           cv = 10).mean()
print("Count Vectors Doğruluk Oranı:", accuracy)



xgb = xgboost.XGBClassifier()
xgb_model = xgb.fit(x_train_tf_idf_word,train_y)
accuracy = model_selection.cross_val_score(xgb_model,
                                           x_test_tf_idf_word,
                                           test_y,
                                           cv = 10).mean()
print("Word-Level TF-IDF Doğruluk Oranı:", accuracy)



xgb = xgboost.XGBClassifier()
xgb_model = xgb.fit(x_train_tf_idf_ngram,train_y)
accuracy = model_selection.cross_val_score(xgb_model,
                                           x_test_tf_idf_ngram,
                                           test_y,
                                           cv = 10).mean()
print("N-GRAM TF-IDF Doğruluk Oranı:", accuracy)



xgb = xgboost.XGBClassifier()
xgb_model = xgb.fit(x_train_tf_idf_chars,train_y)
accuracy = model_selection.cross_val_score(xgb_model,
                                           x_test_tf_idf_chars,
                                           test_y,
                                           cv = 10).mean()

print("CHARLEVEL Doğruluk Oranı:", accuracy)




### TAHMİN
loj_model.predict("yes i like this film")

yeni_yorum = pd.Series("this film is very nice and good i like it")

yeni_yorum = pd.Series("no not good look at that shit very bad")

v = CountVectorizer()
v.fit(train_x)
yeni_yorum = v.transform(yeni_yorum)

loj_model.predict(yeni_yorum)



