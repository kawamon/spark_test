# Sample code of pyspark
from __future__ import print_function

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

#import seaborn as sns

#import folium 
from pyspark.ml.clustering import KMeans
from pyspark.sql.functions import col
from pyspark.ml.clustering import KMeans


#print (kmeans.explainParams())

rdata = spark.read.csv("/test/devicestatus_etl/*")
#rdata.printSchema()

rdatac1 = rdata.select("_c3", "_c4")\
        .withColumnRenamed("_c3","lat").withColumn("lat", col("lat").cast("double"))\
        .withColumnRenamed("_c4","lon").withColumn("lon", col("lon").cast("double"))

# FIXME: 0のデータをスキップする
rdatac = rdatac1.where((rdatac1.lat > 0))

#rdatac.printSchema()
#rdatac.show(3)
#rdata.take(3)

featuresSelected = ["lat","lon"]

from pyspark.ml.feature import VectorAssembler
va = VectorAssembler(inputCols=featuresSelected,outputCol="features")

rdataAssembled = va.transform(rdatac)


from pyspark.ml.feature import StandardScaler
ss = StandardScaler(inputCol="features",outputCol="featuresScaled", withMean=False,withStd=True)
rdataStandardized=ss.fit(rdataAssembled).transform(rdataAssembled)

#rdataStandardized.describe().show()

############################################################################
## 上記までがデータの準備。SparkのProgramming Guideに記載されている内容は
## libsvm形式のデータを読み込んでいますが、この例では devicestatus_etl
## 以下のデータを読み込んで変換しています

kmeans = KMeans(featuresCol="features",predictionCol="cluster", k=5)

#type(kmeans)
#print(kmeans.explainParams())

kmeansModel = kmeans.fit(rdataStandardized)
kmeansModel.computeCost(rdataStandardized)

centers = kmeansModel.clusterCenters()

print ("cluster centers")

for center in centers:
    print (center) 
