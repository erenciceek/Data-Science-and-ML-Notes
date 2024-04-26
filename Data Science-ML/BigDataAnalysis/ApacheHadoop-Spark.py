# APACHE HADOOP
"""
Big data, büyük, çeşitli ve hızlı bir şekilde büyüyen veri kümelerini ifade eder. Bu verilerin geleneksel veri işleme yöntemleriyle işlenmesi zordur. Big data,
genellikle üç ana özellik ile tanımlanır: Hacim, Çeşitlilik ve Hız (3V kuralı).

Hacim (Volume): Big data, çok büyük veri miktarlarını içerir. Örneğin, sosyal medya siteleri, uydu görüntüleri veya büyük perakendecilerin müşteri işlem kayıtları gibi.
Çeşitlilik (Variety): Verilerin yapılandırılmış (örn., veritabanları), yapılandırılmamış (örn., videolar, fotoğraflar, metin belgeleri) ve yarı yapılandırılmış (örn.,
XML, JSON) olmak üzere çeşitli formatlarda gelmesi anlamına gelir.
Hız (Velocity): Verilerin üretildiği ve işlenmesi gereken hız. Örneğin, finansal işlemler, online işlemler veya IoT (Nesnelerin İnterneti) cihazlarından gelen veriler
gerçek zamanlı veya neredeyse gerçek zamanlı analiz edilmelidir.

Big data'nın diğer önemli özellikleri arasında Değer (verilerden elde edilebilecek yarar) ve Doğruluk (veri kalitesi ve güvenilirliği) da bulunur.

Big Data'nın Uygulamaları
Sağlık Sektörü: Hasta kayıtlarını, tedavi sonuçlarını analiz ederek hastalıkları daha iyi anlama ve tedavi yöntemlerini geliştirme.
Perakende ve E-Ticaret: Müşteri alışkanlıklarını analiz ederek kişiselleştirilmiş pazarlama stratejileri oluşturma.
Finans ve Bankacılık: Dolandırıcılık tespiti ve risk yönetimi için gerçek zamanlı veri analizi.
Akıllı Şehirler: Trafik akışını optimize etmek, enerji kullanımını düzenlemek ve güvenlik sistemlerini geliştirmek için verileri kullanma.
Telekomünikasyon: Ağ performansını izleme ve müşteri hizmetleri iyileştirmeleri yapma.

Big Data Teknolojileri
Big data'nın işlenmesi için çeşitli teknolojiler ve platformlar kullanılır. Apache Hadoop, büyük veri kümelerini dağıtık bir şekilde işlemek için yaygın olarak kullanılan
açık kaynaklı bir yazılım çerçevesidir. Hadoop'un yanı sıra Apache Spark daha hızlı işleme ve analiz imkanları sunar. NoSQL veritabanları, büyük veri için esnek şema ve
hızlı erişim sağlar. Veri depolama için çözümler arasında Amazon S3, Google BigQuery ve Microsoft Azure gibi bulut hizmetleri de bulunmaktadır.


Faydaları : Veri saklama ve işleme gücü, açık kaynak, hız, esneklik, ölçeklenebilirlik, hata toleransı
Dezavantajı : Disk tabanlı çalışan bir modeldir. Her MapReduce görevinde diskten okuma ve diske yazma işlemi yapılır.
İteratif işlemler zaman alır ve kaynakları meşgul eder.

Big data, iş dünyasından sağlığa, eğitimden kamu hizmetlerine kadar birçok alanda devrim yaratma potansiyeline sahiptir ve doğru kullanıldığında verimliliği artırabilir,
yenilik sağlayabilir ve karar verme süreçlerini iyileştirebilir.
"""


# APACHE SPARK
"""
Apache Spark, büyük veri işleme için geliştirilmiş açık kaynaklı bir hızlı ve genel amaçlı küme hesaplama sistemi. Hadoop MapReduce’a alternatif olarak 
sunulmuş ve ondan daha hızlı veri işleme yetenekleriyle dikkat çekmiştir. Spark, verileri bellek içinde işleyebilme yeteneğiyle (in-memory computing) 
bilinir, bu da onu özellikle iteratif algoritmalar ve interaktif veri analizi için uygun hale getirir. Apache Spark, makine öğrenimi, gerçek zamanlı akış 
işleme ve grafik işleme gibi çeşitli büyük veri uygulamalarını destekler.

Apache Spark’ın Temel Bileşenleri
RDD (Resilient Distributed Dataset): Spark'ın temel veri yapısı olan RDD, hata toleranslı, paralel olarak işlenebilen dağıtılmış bir veri koleksiyonudur. 
Veriler bellek üzerinde veya disk üzerinde saklanabilir, ve kullanıcıların veri üzerinde dönüşümler yapmasına ve eylemleri gerçekleştirmesine olanak tanır.
DataFrame: Spark 1.3 sürümü ile tanıtılan DataFrame, Spark’ın yüksek seviye API’sidir. R ve Python'daki DataFrame'ler ile benzerlik gösterir. RDD’ler 
üzerinde daha yüksek düzeyde bir soyutlama sağlar ve SQL benzeri operasyonlar yapılmasına imkan verir.
Dataset: Spark 2.0 ile birlikte tanıtılan Dataset API, RDD’nin hata toleransı özelliklerini ve DataFrame’in optimize edilmiş yürütme motorunu birleştirir. 
Tip güvenliği sağlaması ve daha karmaşık veri yapıları ile çalışma yeteneği sunmasıyla ön plana çıkar.
Spark Streaming: Gerçek zamanlı veri akışlarını işlemek için kullanılır. Spark Streaming, veriyi mini-batch'ler halinde işler, böylece neredeyse gerçek zamanlı 
analizler yapabilir.
MLlib (Machine Learning Library): Spark, büyük veri üzerinde makine öğrenimi modelleri geliştirmek ve eğitmek için geniş bir makine öğrenimi kütüphanesine sahiptir.
GraphX: Grafik işleme ve graf tabanlı hesaplamalar için Spark’ın API’sidir. Grafik algoritmalarını paralel olarak uygulama yeteneği sunar.


Apache Spark’ın Avantajları
Hız: Bellek içi hesaplama yeteneği, Spark’ı disk tabanlı sistemlerden (örneğin, Hadoop MapReduce) çok daha hızlı kılar. Özellikle makine öğrenimi ve 
interaktif sorgulama gibi iteratif işlemlerde performans avantajı sağlar.
Esneklik: Çeşitli veri kaynaklarından (HDFS, Cassandra, HBase, S3 vb.) veri okuyabilir ve çoklu dil desteği (Scala, Java, Python, R) sunar.
Kolay Kullanım: Yüksek seviye API’ları ve kapsamlı kütüphaneleri ile geliştiricilerin veri setleri üzerinde kolayca işlem yapmalarını ve analizler gerçekleştirmelerini sağlar.
Geniş Ekosistem: Spark, SQL sorgulamaları, streaming data, makine öğrenimi ve grafik işleme dahil olmak üzere geniş bir uygulama yelpazesini destekler.


Apache Spark, büyük veri analitiği ve işleme alanında güçlü bir araç olarak kabul edilir ve büyük ölçekli veri işleme ihtiyaçlarını karşılamak için giderek 
daha fazla tercih edilen bir platformdur.
"""


# API VE PYPSPARK
"""
API, "Application Programming Interface" (Uygulama Programlama Arayüzü) kısaltmasıdır. API'ler, bir yazılım bileşeni ile diğer yazılım bileşenleri arasında 
iletişim kurmanın standart bir yolunu tanımlarlar. Temel olarak, API'ler belirli bir uygulamanın ya da kütüphanenin işlevlerini, metotlarını, nesnelerini 
veya sınıflarını başka bir yazılım geliştiricisinin kullanımına sunan bir dizi tanım ve protokoldür. API'ler sayesinde geliştiriciler, var olan bir servisin 
veya uygulamanın işlevlerini yeniden yazmak zorunda kalmadan bu işlevleri kendi uygulamalarında kullanabilirler.

PySpark API'lerinin Sundukları:
DataFrame API: Pandas'ın DataFrame'lerine benzer bir yapı sunar, ancak bu DataFrame'ler dağıtık sistemlerde çalışacak şekilde tasarlanmıştır. Veri manipülasyonu 
ve analizi için zengin bir işlev seti sağlar.
RDD (Resilient Distributed Dataset) API: Spark'ın temel veri yapısı olan RDD'ler üzerinde dönüşümler (transformations) ve eylemler (actions) yapmayı sağlar. 
RDD'ler, hata toleranslı ve paralelleştirilmiş veri işleme için kullanılır.
Spark SQL API: SQL sorgulama yetenekleri ve veritabanı tablo benzeri işlevler sağlar. Veri bilimcileri ve analistleri tarafından, veritabanı benzeri 
sorgulamalar ve veri işleme için tercih edilir.
MLlib (Machine Learning Library) API: Makine öğrenimi modellerinin eğitilmesi, değerlendirilmesi ve kullanılması için araçlar sunar. Geniş bir makine öğrenimi 
algoritması yelpazesi içerir.
Spark Streaming API: Gerçek zamanlı veri akışlarını işlemek için kullanılır. Bu API ile kullanıcılar, canlı veri akışlarını işleyebilir ve analiz edebilirler.
GraphX API: Graf tabanlı veri işleme için kullanılır. Ağ analizi, sosyal medya analizi gibi uygulamalarda kullanılabilir.


PySpark'ın API'leri, Python dilinin kolay okunabilirliği ve kullanımı ile Spark'ın dağıtık veri işleme kabiliyetlerini birleştirir, böylece geliştiriciler 
Python ile rahat bir şekilde büyük veri uygulamaları oluşturabilir ve çalıştırabilirler
"""
import findspark
findspark.init("C:\\spark")

# Konfigürasyon ve Spark Bağlantısı
import pyspark
from pyspark import SparkContext
sc = SparkContext(master = "local")
print(sc.uiWebUrl)
print(sc.version)
print(sc.sparkUser())
sc.appName
dir(sc)
sc.stop()  # açılan session'ı kapatmak için


from pyspark.sql import SparkSession
from pyspark.conf import SparkConf

spark = SparkSession.builder \
    .master("local") \
    .appName("pyspark_uygulama") \
    .config("spark.executer.memory", "16gb") \
    .getOrCreate()

sc = spark.sparkContext
print(sc.uiWebUrl)

spark_df = spark.read.csv("Data Science-ML/BigDataAnalysis/diabetes.csv" ,
                          header = True, inferSchema = True)

spark_df.printSchema()
type(spark_df)
spark_df.cache() # verimizi cache'e taşımak için

# pandas kütüphanesindeki dataframe fonksiyonları ile test ediyoruz.
import seaborn as sns
df = sns.load_dataset("diamonds")
df = df.select_dtypes(include=["float64","int64"])
type(df)
df.head()

spark_df.head()
df.dtypes
spark_df.dtypes

df.ndim
spark_df.ndim # spark_df ndim diye bir özelliğe sahip değil.



"""
Pandas DataFrame'leri ve Spark DataFrame'leri, veri analizi ve manipülasyonu için kullanılan iki popüler veri yapıdır, ancak aralarında önemli farklar 
vardır. İşte bu iki DataFrame türü arasındaki başlıca farklılıklar:

Depolama ve İşleme Mekanizması
Pandas DataFrame'leri: Tek bir makinenin belleğinde saklanır ve işlenir. Veri setinin makinenin RAM'ine sığabilecek büyüklükte olması gerekir. Öncelikli 
olarak tekil işlemci üzerinde çalışır (ancak bazı paralel işleme özellikleri eklenmiştir).
Spark DataFrame'leri: Dağıtık bir sistemde, birden çok makine üzerinde saklanabilir ve işlenebilir. Veri setleri çok büyük olsa bile işlenebilir, çünkü 
veriler birden fazla node üzerinde parçalanır ve her node kendi bölümünü işler. Paralel işleme için tasarlanmıştır ve birden çok işlemciyi veya bilgisayarı 
kullanarak büyük veri setleri üzerinde çalışabilir.

Performans ve Ölçeklenebilirlik
Pandas DataFrame'leri: Küçük ila orta ölçekli veri setleri için idealdir. RAM sınırlamaları nedeniyle büyük veri setleri ile çalışırken zorluklar yaşayabilir.
Spark DataFrame'leri: Büyük veri setlerini işleyebilme yeteneği ile ölçeklenebilirlik avantajına sahiptir. Hafızada (in-memory) işleme yeteneği sayesinde, 
büyük veri işlemlerinde yüksek performans sunar.

Syntax ve Fonksiyonlar
Pandas DataFrame'leri: Python programlama diline özgü bir syntax kullanır. Geniş bir topluluk tarafından desteklenir ve birçok kullanışlı kütüphane ile 
entegrasyon sağlar. Birçok kullanıcı için daha sezgisel ve erişilebilir olabilir.
Spark DataFrame'leri: Spark'ın kendi API'sini kullanır, ancak Pandas'a benzer işlevler sunar. Spark SQL'in SQL sorgulama yeteneklerini içerir, bu da veri 
üzerinde SQL sorguları çalıştırmayı mümkün kılar. Pandas API'lerine benzer metodlar sağlamak için sürekli geliştirilir, ancak tam bir birebir eşleme sunmaz.

Ekosistem ve Entegrasyon
Pandas DataFrame'leri: Python veri bilimi ekosisteminin bir parçasıdır ve NumPy, SciPy, Matplotlib, Seaborn gibi diğer kütüphanelerle iyi entegre olur.
Spark DataFrame'leri: Apache Spark ekosisteminin bir parçasıdır ve MLlib, Spark Streaming, GraphX gibi diğer Spark bileşenleriyle entegre çalışabilir.

Kullanım Kolaylığı ve Öğrenme Eğrisi
Pandas DataFrame'leri: Yeni başlayanlar ve tek makinede çalışan veri bilimcileri için daha kolay bir başlangıç noktası sağlar. Geniş çapta kullanımı ve 
kapsamlı dokümantasyonu sayesinde öğrenme kaynaklarına erişmek daha kolaydır.
Spark DataFrame'leri: Dağıtık sistemler ve büyük veri işleme konusunda bilgi gerektirir.Daha dik bir öğrenme eğrisi olabilir, özellikle de Spark'ın diğer 
bileşenlerini ve dağıtık sistem kavramlarını anlamak gerektiğinde.


Bu farklılıklar nedeniyle, hangi tür DataFrame'in kullanılacağı, ihtiyacınız olan ölçeklenebilirlik, performans, veri büyüklüğü ve mevcut sisteminiz gibi 
faktörlere bağlı olarak değişir. Pandas, daha küçük veri setleri ve hızlı prototipleme için mükemmeldir; Spark ise büyük veri setlerini işlemek ve üretim 
ortamında yüksek performans sağlamak için tercih edilir.
"""


# TEMEL DATAFRAME İŞLEMLERİ
spark_df.show(2,truncate = True) # top 2 satırı gösterir
spark_df.count() # total gözlem sayısı
spark_df.columns
len(spark_df.columns)
spark_df.describe().show() # özet istatistikler
spark_df.describe("Glucose").show()

spark_df.select("Glucose","Pregnancies").show() # değişken seçme işlemi
spark_df.select("Glucose").distinct().count() # unique gözlem
spark_df.select("Glucose").dropDuplicates().show() # tekrar eden gözlemleri sildi.

spark_df.crosstab("Outcome","Pregnancies").show()
spark_df.dropna().show(3)

# gözlem seçme
spark_df.filter(spark_df.Age > 40).count() # yaşı 40dan büyük olan gözlem sayısı
spark_df.groupby("Outcome").count().show()
spark_df.groupby("Outcome").agg({"BMI": "mean"}).show()
# yeni değişken ekleme
spark_df.withColumn("yeni_degisken",spark_df.BMI / 2).select("BMI","yeni_degisken").show()

# var olan değişkenin ismini değiştirme
spark_df.withColumnRenamed("Outcome","bagimli_degisken")
spark_df.withColumnRenamed("Outcome","bagimli_degisken").columns

spark_df.head()
spark_df.show(3)
spark_df.drop("Insulin").columns

spark_df.groupby("Outcome").count().show()
a = spark_df.groupby("Outcome").count().toPandas() # spark df to pandas df
a.iloc[0:1,0:1]



# SQL İşlemleri
spark_df.registerTempTable("table_df")
spark.sql("show databases").show()
spark.sql("show tables").show()
spark.sql("select Glucose from table_df").show(5)
spark.sql("select Outcome, mean(Glucose) from table_df group by Outcome").show()


# Büyük Veri Görselleştirme
import matplotlib.pyplot as plt
import seaborn as sns
# alttaki satır çalışmayacak, pandas dataframe'e uyumlu yalnızca
sns.barplot(x = "Outcome", y = spark_df.Outcome.index, data = spark_df)

sdf = spark_df.toPandas()
sdf.head()

# Genelde büyük verinin büyük yükü spark ile halledilir, veri indirgenir.
# Daha sonra elde bulunan indirgenmiş veriyi toPandas ile
# pandas DataFrame'ine çevirip görselleştirilir.

sns.barplot(x = "Outcome", y = sdf.Outcome.index, data = sdf)











