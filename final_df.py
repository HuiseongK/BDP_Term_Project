from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col,mean,desc,regexp_replace,round,when
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

if __name__ == "__main__":
	spark = SparkSession.builder.appName("final_df").getOrCreate()
	distri = spark.read.load("분양정보.csv",format="csv", sep=",",
inferSchema="true", header="true")
	cmpt_rate = spark.read.load("청약 경쟁률.csv",format="csv", sep=",",
inferSchema="true", header="true")
	count = spark.read.load("공급위치.csv",format="csv", sep=",",
inferSchema="true", header="true")
	price = spark.read.load("consumerPrice.csv",format="csv", sep=",",
inferSchema="true", header="true")
	seoul_pre  = spark.read.load("seoul_pre2.csv",format="csv", sep=",",
inferSchema="true", header="true")
	dongCode  = spark.read.load("dongCode.csv",format="csv", sep=",",
inferSchema="true", header="true")


	join = distri.join(cmpt_rate,["공고번호"],'inner')

	seoul_df = join.where(col('공급지역명')=='서울')

	seoul_df = seoul_df.select(
			col('공급위치'),
			col('청약접수시작일'),
			col('경쟁률')
			)
						 
	count = count.select(
			col('공급규모'),
			col('계약시작일'),
			col('공급위치').alias('공급위치1')
			)

	preprocess=seoul_df.withColumn("청약접수시작일",
			regexp_replace("청약접수시작일","-",""))

	preprocess=preprocess.groupBy("공급위치","청약접수시작일")\
			.agg(F.mean("경쟁률").alias("경쟁률"))
									 
	join_df=count.join(preprocess,(count['계약시작일']==preprocess['청약접수시작일'])\
			&(count['공급위치1']==preprocess['공급위치']),'outer')

	join_df=join_df.withColumn("계약시작일",when(col("계약시작일").isNull(),col("청약접수시작일"))\
			.otherwise(col("계약시작일")))
	join_df=join_df.withColumn("공급위치",when(col("공급위치").isNull(),col("공급위치1"))    \
			.otherwise(col("공급위치")))
	join_df=join_df.drop(col("공급위치1"))
	join_df=join_df.drop(col("청약접수시작일"))
									  

	seoul=dongCode.join(seoul_pre,(dongCode['code']==seoul_pre['지역코드']),'inner')
	seoul=seoul.select(
			col('거래금액'),
			col('year_month'),
			col('dong')
			)
	seoul=seoul.groupBy("dong","year_month").agg(F.mean("거래금액").alias("거래금액"))

	join_df=join_df.join(seoul,(join_df["공급위치"]==seoul["dong"])\
			&(join_df["계약시작일"]==seoul["year_month"]),'outer')
	join_df=join_df.withColumn("공급위치",when(col("공급위치").isNull(),col("dong"))    \
			 .otherwise(col("공급위치")))
	join_df=join_df.withColumn("계약시작일",when(col("계약시작일").isNull(),col("year_month"))    \
			 .otherwise(col("계약시작일")))
	join_df=join_df.join(price,(join_df['계약시작일']==price['date']),'outer')
	join_df=join_df.drop(col("계약시작일"))
	
	join_df=join_df.drop(col("dong"))
	join_df=join_df.drop(col("year_month"))

	join_df=join_df.withColumn("경쟁률",when(col("경쟁률").isNull(),0)\
			.otherwise(col("경쟁률")))
	join_df=join_df.withColumn("공급규모",when(col("공급규모").isNull(),0)\
			.otherwise(col("공급규모")))
	join_df=join_df.withColumn("공급위치",when(col("공급위치").isNull(),0)\
			.otherwise(col("공급위치")))
	join_df=join_df.withColumn("data",when(col("data").isNull(),0)\
			.otherwise(col("data")))
	join_df=join_df.withColumn("date",when(col("date").isNull(),0)\
			.otherwise(col("date")))
	join_df=join_df.withColumn("거래금액",when(col("거래금액").isNull(),0)\
			 .otherwise(col("거래금액")))
	
	join_df=join_df.where(col('공급위치')=='강남구')
	join_df=join_df.drop(col("공급위치"))

	join_df=join_df.select(
		col('공급규모'),
		col('경쟁률'),
		col('거래금액'),
		col('data').alias('소비자물가'),
		col('date').alias('time')
	)

	feature_list = join_df.columns 
	feature_list.remove("거래금액") 

	vecAssembler = VectorAssembler(inputCols=feature_list, outputCol="features") 
	lr = LinearRegression(featuresCol="features",
			labelCol="거래금액").setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8) 
	trainDF, testDF = join_df.randomSplit([0.8, 0.2], seed=42)

	pipeline = Pipeline(stages=[vecAssembler, lr]) 
	pipelineModel = pipeline.fit(trainDF)
	predDF = pipelineModel.transform(testDF)
	predAndLabel = predDF.select("prediction","거래금액")
	predAndLabel.show()

	evaluator = RegressionEvaluator()
	evaluator.setPredictionCol("prediction")
	evaluator.setLabelCol("거래금액")
	print(evaluator.evaluate(predAndLabel, {evaluator.metricName: "r2"}))
	print(evaluator.evaluate(predAndLabel, {evaluator.metricName: "mae"}))
	print(evaluator.evaluate(predAndLabel, {evaluator.metricName: "rmse"}))
