from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

if __name__ == "__main__":
	spark = SparkSession.builder.appName("sparkTest").getOrCreate()
	data = spark.read.load("/user/maria_dev/BDP_Term_Project/seoul.csv", format = "csv", sep = ";", \
			inferSchema = True, header = True)
	data.show(5)
