from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

if __name__ == "__main__":
	spark = SparkSession.builder.appName("sparkTest").getOrCreate()
	data = spark.read.load("/user/maria_dev/BDP_Term_Project/seoul_pre.csv",\
			format = "csv", sep = ",", \
			inferSchema = True, header = True)
	consumerPrice_df = spark.read.load("/user/maria_dev/BDP_Term_Project/consumerPrice.csv",\
			format = "csv", sep = ",",\
			inferSchema = True, header = True)
	
	joined = data.join(consumerPrice_df, data["year_month"] == consumerPrice_df["date"], "inner")
	joined.show()

	drop_columns_list = ["_c0", "거래유형", "법정동", "아파트", "중개사소재지", "지번", "해제사유발생일", "해제여부", "year_month", "지역코드", "일", "date"]
	droped = joined.drop(*drop_columns_list)
	droped.show()
	columns = droped.columns
	columns.remove("거래금액")
	print(columns)
	
	vecAssembler = VectorAssembler(inputCols = columns, outputCol = "price")
	lr = LinearRegression(featuresCol = "price", labelCol = "거래금액").setMaxIter(10).setRegParam(0.2).setElasticNetParam(0.9)
	trainDF, testDF = droped.randomSplit([0.8, 0.2], seed = 42)
	print(trainDF.cache().count())
	print(trainDF.count())

	pipeline = Pipeline(stages = [vecAssembler, lr])
	pipelineModel = pipeline.fit(trainDF)
	predDF = pipelineModel.transform(testDF)
	predAndLabel = predDF.select("prediction", "거래금액")
	predAndLabel.show()

	evaluator = RegressionEvaluator()
	evaluator.setPredictionCol("prediction")
	evaluator.setLabelCol("거래금액")
	print(evaluator.evaluate(predAndLabel, {evaluator.metricName: "r2"}))
	print(evaluator.evaluate(predAndLabel, {evaluator.metricName: "mae"}))
	print(evaluator.evaluate(predAndLabel, {evaluator.metricName: "rmse"}))
	
