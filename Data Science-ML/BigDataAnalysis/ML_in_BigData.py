import findspark

findspark.init("C:\\spark")

# Konfigürasyon ve Spark Bağlantısı

from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark import SparkContext

spark = SparkSession.builder \
    .master("local") \
    .appName("churn_modellemesi") \
    .config("spark.executer.memory", "16gb") \
    .getOrCreate()


sc = spark.sparkContext

spark_df = spark.read.csv("Data Science-ML/BigDataAnalysis/churn.csv",
                          header = True,
                          inferSchema = True,
                          sep = ",")

spark_df.cache()
spark_df.printSchema()
spark_df.show(5)

# değişkenlerin isimlerinin küçük harflere çevirmek için :
spark_df = spark_df.toDF(*[c.lower() for c in spark_df.columns])

 #  df.columns = map(str.lower,df.columns)
spark_df = spark_df.withColumnRenamed("_c0","index")
spark_df.count() # gözlem sayısı
len(spark_df.columns) # değişken sayısı

spark_df.distinct().count() # tekrar eden eleman yok (unique) , bütün sütunları birbirinden farklı olan eleman sayısının verir
spark_df.select("names").distinct().count() # kaç farklı name var, 1 isim tekrar etmiş.
spark_df.groupby("names").count().sort("count",ascending=False).show(3) # jennifer wood iki kere yer almış tabloda

# Duplicate kontrolü mutlaka yapılmalı!
spark_df.filter(spark_df.names == "Jennifer Wood").show() # farklı kişilermiş
spark_df.select("names").dropDuplicates().groupBy("names").count().sort("count" ,ascending=False).show(3)
# dropDuplicates() den sonraki işlemleri gözlemlemek için yapıyoruz.

# belirli bir indexe erişmek için where'i kullanabiliriz.
spark_df.where(spark_df.index == 439).show()

# spark çıktısını işlenebilir bir girdi olarak kullanılabilir bir nesne olmasını sağlamak için
jen = spark_df.where(spark_df.index == 439).collect()[0]["names"]
type(jen)
dir(jen)
jen.upper()

# toPandas ve collect fonksiyonları veriyi spark'tan lokalimize indirmek için
# kullanabilceğimiz fonksiyonlardandır.

# veriyi incelemeye devam ediyoruz :
spark_df.describe().show()
# eğer iyi bir görüntü elde edilmezse :
spark_df.select("age","total_purchase","account_manager", "years", "num_sites", "churn").describe().toPandas().transpose()


spark_df.filter(spark_df.age > 47).count()

spark_df.groupby("churn").count().show()
spark_df.groupby("churn").agg({"total_purchase" : "mean"}).show() # sonuca göre total_purchase değişkeninin churn değişkenine pek etkisi olmadığı görülmüştür.

spark_df.groupby("churn").agg({"years" : "mean"}).show()

# korelasyon durumunu gözlemlemek için:
kor_data = spark_df.drop("index","names","churn").toPandas()
import seaborn as sns
sns.pairplot(kor_data)

kor_data = spark_df.drop("index","names").toPandas()
sns.pairplot(kor_data, hue = "churn") # churn değişkenini boyut olarak eklemek için

sns.pairplot(kor_data, vars = ["age","total_purchase","years","num_sites"],
             hue = "churn",
             kind="reg")


# DATA PREPROCESSING
spark_df = spark_df.dropna()
spark_df = spark_df.withColumn("age_kare",spark_df.age**2)
spark_df.show()

# bağımlı değişken ayarlama işlemleri
from pyspark.ml.feature import StringIndexer
stringIndexer = StringIndexer(inputCol = "churn", outputCol = "label")
indexed = stringIndexer.fit(spark_df).transform(spark_df)
indexed.dtypes
spark_df = indexed.withColumn("label",indexed["label"].cast("integer"))
spark_df.dtypes

"""
StringIndexer makine öğrenimi modelleri için genellikle ön işleme adımında kullanılır ve kategorik verileri sayısal etiketlere dönüştürmek için kullanılır.

Kodun yaptığı işlem şunlardır:
inputCol: StringIndexer'ın dönüştürme işlemi yapacağı DataFrame içindeki sütunun adını belirler. Bu örnekte, inputCol olarak "churn" adlı sütun seçilmiştir. 
Bu sütun, muhtemelen bir kullanıcının hizmeti bırakıp bırakmadığını (churn) ifade eden kategorik bir veridir.
outputCol: Dönüştürülen sayısal değerlerin yazılacağı yeni sütunun adını belirler. Bu durumda, kategorik "churn" verisi sayısal bir "label" sütununa 
dönüştürülecektir.

İşlevsellik açısından, StringIndexer her bir benzersiz kategori için bir sayısal etiket atar. Örneğin, eğer "churn" sütunu "yes" ve "no" değerlerini içeriyorsa, 
StringIndexer bunları sırasıyla 0 ve 1 gibi sayısal değerlere dönüştürebilir. Burada, sıklık derecesine göre etiketlenir; yani en sık görülen kategori 0, 
ikinci en sık görülen 1, vb. şekilde etiketlenir.

Bu dönüşüm, birçok makine öğrenimi algoritmasının yalnızca sayısal verilerle çalışabilmesi gerçeği göz önüne alındığında önemlidir. Öyle ki, kategorik 
verileri modelleme sürecine dahil etmeden önce sayısallaştırmak gerekir. StringIndexer kullanarak, veri bilimciler kategorik veriyi makine öğrenimi modelleri 
için uygun hale getirebilirler.
"""


# Bağımsız değişkenlerin ayarlaması
from pyspark.ml.feature import VectorAssembler
spark_df.columns
bag = ["age","total_purchase","account_manager","years","num_sites"]
vectorAssembler = VectorAssembler(inputCols = bag, outputCol = "features")
va_df = vectorAssembler.transform(spark_df)

"""
VectorAssembler, birden çok sütunu tek bir vektör sütuna dönüştürerek makine öğrenimi modellerinde özellik vektörü olarak kullanılacak şekilde verileri 
hazırlar. Özellikle, VectorAssembler aşağıdaki işlemleri gerçekleştirir:

VectorAssembler nesnesi oluşturulur ve birden fazla sütunun (feature/özellik) adı inputCols parametresi içinde bir liste olarak verilir. Bu örnekte, 
inputCols için "age", "total_purchase", "account_manager", "years" ve "num_sites" sütunları belirlenmiştir. Bu sütunlar, veri setindeki farklı özellikleri 
temsil ediyor.
outputCol parametresi, dönüştürme sonucunda oluşturulacak olan yeni sütunun adını belirler. Bu örnekte, dönüştürülmüş vektör sütunu "features" adıyla oluşturulacaktır.
VectorAssembler nesnesi transform metodunu kullanarak orijinal spark_df DataFrame'ine uygulanır. Bu işlem, belirtilen inputCols sütunlarını tek bir "features" 
vektör sütunu olarak birleştirir.

Sonuç olarak, va_df adlı yeni bir DataFrame oluşur. Bu DataFrame, eski sütunları içeren spark_df'nin bir kopyasıdır, ancak ek olarak her bir kayıt için bir 
"features" sütunu da içerir. Bu "features" sütunu, her kayıt için bir özellik vektörü içerir ve bu vektör makine öğrenimi modellerinde kullanılmak üzere 
hazır bir formatta sunulur.

Bu hazırlık aşaması, özellikle Spark'ın makine öğrenimi algoritmaları için gereklidir çünkü bu algoritmalar girdi olarak genellikle her kayıt için tek bir 
özellik vektörü bekler. VectorAssembler, bu hazırlığı yaparak model eğitimi veya tahmin yapma süreçlerini kolaylaştırır.
"""

"""
"features" sütunu, daha önce VectorAssembler kullanılarak oluşturulan ve birden çok sayısal özelliği tek bir vektörde birleştiren bir sütundur.
"label" sütunu, genellikle hedef değişkeni veya tahmin edilmeye çalışılan değişkeni içerir. 

Model eğitimi veya tahmin yapma işlemleri genellikle "features" ve "label" sütunlarını gerektirdiğinden, bu sütunları içeren bir DataFrame modelin 
girdisi olarak kullanılır.

Bu nedenle final dataFrame'i oluşturalım :
"""

final_df = va_df.select(["features","label"])
final_df.show()



# TEST-TRAIN
splits = final_df.randomSplit([0.7,0.3])
train_df = splits[0]
test_df = splits[1]


# GBM ile Müşteri Terk Modellemesi
from pyspark.ml.classification import GBTClassifier

gbm = GBTClassifier(maxIter = 10, featuresCol = "features", labelCol = "label")
gbm_model = gbm.fit(train_df)
y_pred = gbm_model.transform(test_df)
y_pred.show()
ac = y_pred.select("label","prediction")
ac.filter(ac.label == ac.prediction).count() / ac.count()
# accuracy değeri 0.871...



# Model Tuning
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder , CrossValidator

evaluator = BinaryClassificationEvaluator()


paramGrid = (ParamGridBuilder()
             .addGrid(gbm.maxDepth, [2, 4, 6])
             .addGrid(gbm.maxBins, [20, 30])
             .addGrid(gbm.maxIter, [10, 20])
             .build())

cv = CrossValidator(estimator = gbm, estimatorParamMaps = paramGrid, evaluator = evaluator, numFolds=10)

"""
BinaryClassificationEvaluator:
Bu sınıf, ikili sınıflandırma görevleri için modelin performansını değerlendirmek üzere kullanılır. Genellikle Area Under ROC (Receiver Operating Characteristic) 
gibi ölçütleri hesaplamak için kullanılır.
ParamGridBuilder:
Bu, farklı parametre kombinasyonlarını deneyerek en iyi modeli bulmak için bir parametre ızgarası (parameter grid) oluşturur. ParamGridBuilder metodu, 
algoritmanın farklı parametre değerleri üzerinde iterasyon yapmasına olanak tanır.

CrossValidator:
CrossValidator, modelin genelleyebilirliğini değerlendirmek için çapraz doğrulama yöntemini uygular. Model, farklı alt veri setlerinde eğitilerek test edilir.
estimator: Ayarlanacak estimator (model) olarak gbm (GBTClassifier) belirtilir.
estimatorParamMaps: Modelin denenmesi istenen parametre kombinasyonlarını içeren parametre ızgarası olarak paramGrid kullanılır.
evaluator: Modelin her bir kombinasyonunun değerlendirilmesinde kullanılacak olan BinaryClassificationEvaluator nesnesi belirtilir.
numFolds=10: Veri setinin 10 katlı çapraz doğrulama için nasıl bölüneceğini belirtir. Yani veri seti 10 farklı alt sete bölünür, her biri bir kez test seti 
olarak kullanılırken diğer dokuzu eğitim seti olarak kullanılır.

Bu adımlar tamamlandığında, CrossValidator en iyi performansı gösteren parametre kombinasyonunu seçerek bir "en iyi model" üretir. Bu model, verilen 
veri setinde makine öğrenimi modelinin genel performansını optimize etmek için kullanılabilir.
"""

cv_model = cv.fit(train_df)
y_pred = cv_model.transform(test_df)

ac = y_pred.select("label","prediction")
ac.filter(ac.label == ac.prediction).count() / ac.count()
# accuracy değeri 0.8905...

evaluator.evaluate(y_pred) # AUC değeri : 0.8966..



#  Bu müşteriler bizi terk eder mi ?
import pandas as pd
names = pd.Series(["Ali Ahmetoğlu", "Berkcan Tanerbey", "Harika Gündüz" , "Polat Alemdar", "Ata Bakmayan Ali"])
age = pd.Series([38,43,34,50,40])
total_purchase = pd.Series([30000,10000,6000,30000,100000])
account_manager = pd.Series([1,0,0,1,1])
years = pd.Series([20,10,3,8,30])
num_sites = pd.Series([30,8,8,6,50])

yeni_musteriler = pd.DataFrame({
    'names' : names,
    'age' : age,
    "total_purchase" : total_purchase,
    "account_manager" : account_manager,
    "years" : years,
    "num_sites" : num_sites})

yeni_musteriler.columns
yeni_musteriler.head()

yeni_sdf = spark.createDataFrame(yeni_musteriler)
type(yeni_sdf)
yeni_sdf.printSchema()
yeni_sdf.show() # ÇALIŞMIYOR

yeni_musteriler = vectorAssembler.transform(yeni_sdf)
sonuclar = cv_model.transform(yeni_musteriler)
sonuclar.select("names","prediction").show()













