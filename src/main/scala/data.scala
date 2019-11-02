import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types._;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.sql.functions.{min, max, log};
import org.apache.spark.sql.DataFrame;
import breeze.plot._;

class data {
  def loadData(csvpath:String): DataFrame = {
    val spark = SparkSession.builder().master("local[*]")
      .appName("myapp").getOrCreate();

    var df = spark.read.format("csv")
      .option("header", "true")
      .load(csvpath);

    df.show();
    return df;
  };

  def cleanData(df: DataFrame): DataFrame = {
    // Cast lot area and price to double
    var colNames = Array("Price","LotArea","Bathrooms","FullBedrooms","OtherRooms");
    var df2 = colNames.foldLeft(df) { (df, col) =>
      df.withColumn(col, df(col).cast(DoubleType))
    };

    // Remove all rows with nans
    df2 = df2.na.drop();

    // Remove values that are too high
    // Price from 100,000 to 10,000,000
    var df3 = df2.filter(s"Price > 10000 and Price < 15000000");
    // Lot Area from 100 to 20,000 - because several houses were just "under 1/2 acre" which is 21780 sq feet
    var df4 = df3.filter(s"LotArea > 100 and LotArea < 20000");

    // StringIndexer to get locations as numbers
    var indexer = new StringIndexer()
      .setInputCol("ElectoralDiv")
      .setOutputCol("location")
      .setStringOrderType("alphabetAsc");

    var df5 = indexer.fit(df4).transform(df4);

    var df6 = df5.withColumn("logPrice", log("Price"))
    var df7 = df5.withColumn("logArea", log("LotArea"))




    return df5

  }


  def doStats(df: DataFrame): Unit ={
    // Check max and min
    df.agg(min("LotArea"),max("LotArea")).show();
    // Summary statistics
    df.select("Price").describe().show();
    // Correlation
    df.stat.corr("Price","LotArea");
  }

  def plot_data(df: DataFrame){
    var price = df.select( df("Price").cast(DoubleType).as("Price") ).collect.map(row => row.getDouble(0));
    var lotArea = df.select( df("LotArea").cast(DoubleType).as("LotArea") ).collect.map(row => row.getDouble(0));
    var bathroom = df.select( df("Bathrooms").cast(DoubleType).as("Bathrooms") ).collect.map(row => row.getDouble(0));
    var location = df.select( df("location").cast(DoubleType).as("location") ).collect.map(row => row.getDouble(0));

    var logPrice = df.select( df("logPrice").cast(DoubleType).as("logPrice") ).collect.map(row => row.getDouble(0));
    var logArea = df.select( df("logArea").cast(DoubleType).as("logArea") ).collect.map(row => row.getDouble(0));

    var size = Array.fill(salePrice.length){100} //units of x-axis
    var size2 = lotArea.map(_*0.00005)

    var f = Figure();
    var p = f.subplot(0);
    p += scatter(location,salePrice,size2.apply);

    var q = f.subplot(2,1,1);
    p += hist(logPrice,bins=100);
    p.title = "Price";
    q += hist(logArea,bins=100);
    q.title = "Lot Area";
    p += scatter(lotArea,price,size.apply);

    p += scatter(location,salePrice,size2.apply);
    // p += hist(lotArea,bins=100)
    // p += plot(location,salePrice,'.')
    // p += scatter(salePrice,bathroom,size.apply)

    // p.xlim = (0,10000);
    // p.ylim = (0,10000000);
    // p.xlabel = "Lot Area (sq feet)";
    // p.ylabel = "Price";

  }

  def main(csvpath: String) {
    var df = loadData(csvpath)
    var clean_df = cleanData(df)
    plot_data(clean_df)
    df.coalesce(1) //So just a single part- file will be created
      .write
      .option("mapreduce.fileoutputcommitter.marksuccessfuljobs","false") //Avoid creating of crc files
      .option("header","true") //Write the header
      .csv("csvFullPath")
  }
}

object Main {
  // Main method
  def main(args: Array[String])
  {
    var csvpath: String = """all_houses_regions.csv"""
    // Class object
    new data().main(csvpath)
  }

}