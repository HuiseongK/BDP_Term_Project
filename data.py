from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col,mean,desc,regexp_replace,round
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

if __name__ == "__main__":
	spark = SparkSession.builder.appName("data").getOrCreate()
	distri = spark.read.load("분양정보.csv",format="csv", sep=",",
inferSchema="true", header="true")
	cmpt_rate = spark.read.load("청약 경쟁률.csv",format="csv", sep=",",
inferSchema="true", header="true")
	count = spark.read.load("공급위치.csv",format="csv", sep=",",
nferSchema="true", header="true")
	
	cmpt_rate = cmpt_rate.na.fill(value=0,subset=["경쟁률"])
	
	join = distri.join(cmpt_rate,["공고번호"],'inner')

	seoul_df = join.where(col('공급지역명')=='서울')
	seoul_df = seoul_df.select(
			col('공급지역명'),
			col('공급위치'),
			col('청약접수시작일'),
			col('경쟁률')
		)
	
	result_df = seoul_df.groupBy("공급위치","청약접수시작일")\
	.agg(F.mean("경쟁률").alias("m_cmpt")).orderBy("m_cmpt")
	
	count = count.select(
			col('공급규모'),
			col('계약시작일'),
			col('공급위치').alias('공급위치1')
	)

	preprocess=seoul_df.withColumn("청약접수시작일",
			regexp_replace("청약접수시작일","-",""))
	
	join_df=count.join(preprocess,(count['계약시작일']==preprocess['청약접수시작일'])\
			&(count['공급위치1']==preprocess['공급위치']))

	join_df = join_df.select('공급규모','경쟁률')
	feature_list = join_df.columns
	feature_list.remove("경쟁률")
	print(feature_list)

	vecAssembler = VectorAssembler(inputCols=feature_list, outputCol="공급규모")
	lr = LinearRegression(featuresCol="공급규모",
labelCol="경쟁률").setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
	trainDF, testDF = join_df.randomSplit([0.8, 0.2], seed=42)
	print(trainDF.cache().count())
	print(testDF.count())
	
	trainDF, testDF = join_df.randomSplit([0.8, 0.2], seed=42)
	print(trainDF.cache().count())
	print(testDF.count())
	pipeline = Pipeline(stages=[vecAssembler, lr])
	pipelineModel = pipeline.fit(trainDF)
	predDF = pipelineModel.transform(testDF)
	predAndLabel = predDF.select("prediction","경쟁률")
	predAndLabel.show()

	evaluator = RegressionEvaluator()
	evaluator.setPredictionCol("prediction")
	evaluator.setLabelCol("경쟁률")
	print(evaluator.evaluate(predAndLabel, {evaluator.metricName: "r2"}))
	print(evaluator.evaluate(predAndLabel, {evaluator.metricName: "mae"}))
	print(evaluator.evaluate(predAndLabel, {evaluator.metricName: "rmse"}))
