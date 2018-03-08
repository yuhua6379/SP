import java.util
import java.util.{Arrays, HashMap}

import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature._

object Utils {
  def LogLoss(df: DataFrame): Double = {
    val df2 = df.withColumn("log_loss", Udfs.LogLoss(col("probability"), col("is_trade")))
    val n = df2.count()
    -df2.agg(sum("log_loss")).first.getDouble(0) / n
  }

  def OneHot(cols: Array[String], df: DataFrame): DataFrame = {
    var ddf = df
    for (col <- cols) {
      val INDEX_col = "_INDEX_"+col;
      val VEC_col = "_VEC_"+col;
      val indexer = new StringIndexer().setInputCol(col).setOutputCol(INDEX_col).fit(ddf)
      ddf = indexer.transform(ddf)

      val encoder = new OneHotEncoder().setInputCol(INDEX_col).setOutputCol(VEC_col).setDropLast(false)
      ddf = encoder.transform(ddf)
      ddf = ddf.drop(col).withColumnRenamed(col, VEC_col)
    }
    ddf
  }


  private def scaleNumericFeature(input: Dataset[Row], numericFeatures: Array[String]): Dataset[Row] = {
    if (numericFeatures.length <= 0) return input

    val numericFeats: Dataset[Row] = new VectorAssembler().setInputCols(numericFeatures).setOutputCol("numeric_vec").transform(input)
    val model: MinMaxScalerModel = new MinMaxScaler().setInputCol("numeric_vec").setOutputCol("scaled_numeric_vec").fit(numericFeats)
    System.out.println("Min: " + util.Arrays.toString(model.originalMin.toArray))
    System.out.println("Max: " + util.Arrays.toString(model.originalMax.toArray))
    var scaleDict = new util.HashMap[String, Double]
    var i = 0;
    for (field <- numericFeatures){
      scaleDict.put(field + "_max", model.originalMax.apply(i))
      scaleDict.put(field + "_min", model.originalMin.apply(i))
      i += 1
    }
    model.transform(numericFeats).drop("numeric_vec").withColumnRenamed("scaled_numeric_vec", "numeric_vec")
  }
}
