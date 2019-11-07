import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.GBTRegressor
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.sql.types._

/*
Beautiful visualization of GBTRegression:
https://arogozhnikov.github.io/2016/06/24/gradient_boosting_explained.html
GBT House Price prediction in Cali:
https://shankarmsy.github.io/stories/gbrt-sklearn.html
*/

class predict_price {

  def load_data(spark: SparkSession): DataFrame = {
    var csvpath: String = """C:\Users\Laagi\IdeaProjects\spark_test\clean_data.csv"""

    val schema = new StructType()
      .add("MlsNumber",StringType).add("PostalCode",StringType).add("Price",DoubleType)
      .add("PropertyType",StringType).add("Address",StringType).add("Longitude",DoubleType)
      .add("Latitude",DoubleType).add("BuildingType",StringType).add("Bathrooms",DoubleType)
      .add("Bedrooms",StringType).add("Stories",DoubleType).add("Size",StringType)
      .add("RealEstateCompany",StringType).add("FullBedrooms",DoubleType).add("OtherRooms",DoubleType)
      .add("LotArea",DoubleType).add("ElectoralDiv",StringType)

    var df = spark.read.format("csv")
      .option("header", "true")
      .schema(schema)
      .load(csvpath);

    return df;
  }

  def split_data(df: DataFrame): (DataFrame,DataFrame) = {
    var splitData = df.randomSplit(Array(0.7, 0.3));
    var train_df = splitData(0);
    var test_df = splitData(1);
    return (train_df,test_df);
  }

  def pipeline(): CrossValidator = {

    // Stringindexer
    // Don't need one hot encoding for tree-based models:
    // https://stackoverflow.com/questions/32277576/how-to-handle-categorical-features-with-spark-ml
    var indexer = new StringIndexer()
      .setInputCol("ElectoralDiv")
      .setOutputCol("location")
      .setStringOrderType("alphabetAsc")
      .setHandleInvalid("skip")

    var assembler = new VectorAssembler()
      .setInputCols(Array("Bathrooms", "Stories", "FullBedrooms", "OtherRooms", "LotArea", "location"))
      .setOutputCol("features")
      .setHandleInvalid("skip")

    var gbt = new GBTRegressor()
      .setMaxBins(60)
      .setLabelCol("Price")
      .setFeaturesCol("features")
      .setMaxIter(10)
      .setPredictionCol("gbtPrediction")

    var pipeline = new Pipeline()
      .setStages(Array(indexer, assembler, gbt))

    var paramGrid = new ParamGridBuilder()
      .addGrid(gbt.maxDepth, Array(3, 5, 7, 9, 11))
      .addGrid(gbt.minInstancesPerNode, Array(1, 2, 3, 4, 5))
      .addGrid(gbt.stepSize, Array(0.01,0.05,0.1,0.2,0.3))
      .build()

    var evaluator = new RegressionEvaluator()
      .setLabelCol("Price")
      .setPredictionCol("gbtPrediction")
      .setMetricName("mse");

    var cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)
      .setParallelism(2)

    return cv

  }

  def train_model(cv: CrossValidator, train_df: DataFrame): CrossValidatorModel = {
    // Run cross-validation, and choose the best set of parameters.
    var cvModel = cv.fit(train_df)

    // Get best params:
    cvModel.getEstimatorParamMaps.zip(cvModel.avgMetrics).maxBy(_._2)._1
    // More Info:
    cvModel.bestModel.asInstanceOf[PipelineModel]
      .stages.foreach(stage => println(stage.extractParamMap))

    return cvModel
  }

  def test_model(cvModel: CrossValidatorModel, test_df: DataFrame): Unit = {
    // Make predictions on test data
    var preds = cvModel.transform(test_df)
    // Get r-square
    preds.stat.corr("Price","gbtPrediction")
  }

  def save_csv (df: DataFrame) {
    var df2 = df.drop("features")
    df2.coalesce(1)
      .write
      .option("mapreduce.fileoutputcommitter.marksuccessfuljobs", "false")
      .option("header", "true")
      .csv("house_price_Predictions.csv")
  }

  def main(spark: SparkSession) {
    var df = load_data(spark)
    var (train_df,test_df) = split_data(df)
    var cv = pipeline()
    var cvModel = train_model(cv, train_df)
    test_model(cvModel,test_df)

  }
}

object Main {
  def main(args: Array[String]) {
    val spark = SparkSession.builder().master("local[*]")
      .appName("Price_Prediction").getOrCreate();
    Logger.getLogger("org").setLevel(Level.ERROR)
    new data().main(spark)
  }
}