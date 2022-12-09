from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col,mean,desc,regexp_replace,round,when
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
inferSchema="true", header="true")
    price = spark.read.load("consumerPrice.csv",format="csv", sep=",",
inferSchema="true", header="true")
    total = spark.read.load("seoul_pre.csv",format="csv", sep=",",
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
 
join_df=join_df.join(price,(join_df['계약시작일']==price['date']),'outer')
join_df=join_df.drop(col("계약시작일"))
 
join_df.show()
 
total=total.select(
    	col('전용면적'),
        col('거래금액'),
        col('year_month')
      )
 
total=total.groupBy("year_month").agg(F.mean('거래금액').alias('거래금액'))
join_df=join_df.join(total,(join_df['date']==total['year_month']),'outer')
join_df=join_df.drop(col('date'))
join_df.show(300)
