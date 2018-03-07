import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}

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
    }
    ddf
  }
}
