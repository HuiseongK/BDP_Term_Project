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

	#분양정보와 청약 경쟁률 파일을 조인하여 경쟁률과 지역을 나타나게 한다.
	join = distri.join(cmpt_rate,["공고번호"],'inner')
	#where 사용으로 서울지역으로만 한정을 한다.
	seoul_df = join.where(col('공급지역명')=='서울')
	#필요한 컬럼들만 select한다.
	seoul_df = seoul_df.select(
			col('공급위치'),
			col('청약접수시작일'),
			col('경쟁률')
			)
	#공급위치파일에서 필요한 컬럼들만 select한다.
	count = count.select(
			col('공급규모'),
			col('계약시작일'),
			col('공급위치').alias('공급위치1')
			)
	#청약접수시작일의 년도와 월 사이에 -를 regexp_replace를 사용해 없앤다.
	preprocess=seoul_df.withColumn("청약접수시작일",
			regexp_replace("청약접수시작일","-",""))
	#서울 구와 시간을 기준으로 평균경쟁률을 구한다.
	preprocess=preprocess.groupBy("공급위치","청약접수시작일")\
			.agg(F.mean("경쟁률").alias("경쟁률"))
	#두 개의 컬럼을 기준으로 outer조인을 한다.
	join_df=count.join(preprocess,(count['계약시작일']==preprocess['청약접수시작일'])\
			&(count['공급위치1']==preprocess['공급위치']),'outer')
	#위에서 outer join으로 인해 동일한 내용의 컬럼이 두 개가 있지만 한쪽은 null인 값인 반면 다른쪽은 제대로 된
	#값이 있다. 그러므로 when으로 조건을 사용하여 null일 경우 다른쪽의 값을 대입시킨다.
	join_df=join_df.withColumn("계약시작일",when(col("계약시작일").isNull(),col("청약접수시작일"))\
			.otherwise(col("계약시작일")))
	join_df=join_df.withColumn("공급위치",when(col("공급위치").isNull(),col("공급위치1"))    \
			.otherwise(col("공급위치")))
	#outer join으로 인해 동일한 두 컬럼이 있으므로 한 컬럼을 제외시킨다.		
	join_df=join_df.drop(col("공급위치1"))
	join_df=join_df.drop(col("청약접수시작일"))
									  
	# 두 파일을 조인한다.
	seoul=dongCode.join(seoul_pre,(dongCode['code']==seoul_pre['지역코드']),'inner')
	#필요한 컬럼들만 select한다.
	seoul=seoul.select(
			col('거래금액'),
			col('year_month'),
			col('dong')
			)
	#서울 구와 시간을 기준으로 평균거래금액을 구한다.
	seoul=seoul.groupBy("dong","year_month").agg(F.mean("거래금액").alias("거래금액"))
	#두 개의 컬럼을 기준으로 outer조인을 한다.
	join_df=join_df.join(seoul,(join_df["공급위치"]==seoul["dong"])\
			&(join_df["계약시작일"]==seoul["year_month"]),'outer')
	#위에서 outer join으로 인해 동일한 내용의 컬럼이 두 개가 있지만 한쪽은 null인 값인 반면 다른쪽은 제대로 된
	#값이 있다. 그러므로 when으로 조건을 사용하여 null일 경우 다른쪽의 값을 대입시킨다.
	join_df=join_df.withColumn("공급위치",when(col("공급위치").isNull(),col("dong"))    \
			 .otherwise(col("공급위치")))
	join_df=join_df.withColumn("계약시작일",when(col("계약시작일").isNull(),col("year_month"))    \
			 .otherwise(col("계약시작일")))
	#두 파일을 조인한다.
	join_df=join_df.join(price,(join_df['계약시작일']==price['date']),'outer')
	#필요없는 컬럼을 제외한다.
	join_df=join_df.drop(col("계약시작일"))	
	join_df=join_df.drop(col("dong"))
	join_df=join_df.drop(col("year_month"))
	#많은 조인들로 인해 null들이 발생한다. 이 값들을 0으로 대체한다.
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
	#서울의 구를 임의로 입력해서 원하는 구의 정보를 얻을 수 있다.
	join_df=join_df.where(col('공급위치')=='강남구')
	#mulitful regression에 필요없는 요소를 제외한다.
	join_df=join_df.drop(col("공급위치"))
	#필요한 컬럼들만 select한다.
	join_df=join_df.select(
		col('공급규모'),
		col('경쟁률'),
		col('거래금액'),
		col('data').alias('소비자물가'),
		col('date').alias('time')
	)
	#분석에 들어가는 컬럼들의 이름만 지정
	feature_list = join_df.columns 
	#예측하고자 하는 값만 제외
	feature_list.remove("거래금액") 

	vecAssembler = VectorAssembler(inputCols=feature_list, outputCol="features") 
	lr = LinearRegression(featuresCol="features",
			labelCol="거래금액").setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8) 
	#training data와 test data를 8대2 비율로 나뉜다.
	trainDF, testDF = join_df.randomSplit([0.8, 0.2], seed=42)

	pipeline = Pipeline(stages=[vecAssembler, lr]) 
	pipelineModel = pipeline.fit(trainDF)
	predDF = pipelineModel.transform(testDF)
	#time컬럼을 기준으로 예측값부터 모든 컬럼들을 출력한다.
	predDF.orderBy(desc('time')).show()
	#예측컬럼과 실제값을 출력한다.
	predAndLabel = predDF.select("prediction","거래금액")
	predAndLabel.show()

	evaluator = RegressionEvaluator()
	evaluator.setPredictionCol("prediction")
	evaluator.setLabelCol("거래금액")
	#r제곱 score의 값으로 모델의 정확도를 나타낸다. mae와 rmse도 출력한다.
	print(evaluator.evaluate(predAndLabel, {evaluator.metricName: "r2"}))
	print(evaluator.evaluate(predAndLabel, {evaluator.metricName: "mae"}))
	print(evaluator.evaluate(predAndLabel, {evaluator.metricName: "rmse"}))
