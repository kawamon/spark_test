//Sample code of scala (for spark-shell)

val rdata = spark.read.csv("/devicestatus_etl/*")
val rdatac1 = rdata.select("_c3", "_c4").withColumnRenamed("_c3","lat").withColumn("lat", col("lat").cast("double")).withColumnRenamed("_c4","lon").withColumn("lon", col("lon").cast("double"))
val rdatac = rdatac1.where(($"lat" > 0))
//val rdatac = rdatac1.where((rdatac1.lat > 0))
val featuresSelected = ("lat","lon")

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
val va = new VectorAssembler().setInputCols(Array("lat","lon")).setOutputCol("features")
val rdataAssembled = va.transform(rdatac)

import org.apache.spark.ml.feature.StandardScaler
val ss = new StandardScaler().setInputCol("features").setOutputCol("scaledFeatures").
setWithStd(true).setWithMean(false)
val rdataStandardized=ss.fit(rdataAssembled).transform(rdataAssembled)

import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.ml.clustering.KMeans
val kmeans = new KMeans().setK(5).setSeed(1L)
val kmeansModel = kmeans.fit(rdataStandardized)
kmeansModel.computeCost(rdataStandardized)

println("Cluster Centers: ")
kmeansModel.clusterCenters.foreach(println)
