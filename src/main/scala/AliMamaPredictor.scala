//package O98K;

import org.apache.spark.{SparkConf, SparkContext}

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.GBTRegressor

object AliMamaPredictor {
  def main(args: Array[String]): Unit = {

    val sc = new SparkContext(new SparkConf().setMaster("local").setAppName("IJCAI-18"))
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)

    var df = sqlContext.read.parquet("/user/utp/competition/IJCAI-18/1st_df")
    df = df.repartition(200)
    var selectFeatures = new Array[String](0);
    for ( f <- df.schema.fields){
      if (f.name.equals("is_trade") == false
        && f.name.equals("instance_id") == false
        && f.name.equals("context_id") == false
        && f.name.equals("context_timestamp") == false
      ) {
        selectFeatures = selectFeatures :+ f.name
      }
    }


    System.out.println("selected features:")
    System.out.println(selectFeatures.mkString(", "))
    df = df.na.fill(0)

    //Assemble!
    df = new VectorAssembler()
      .setInputCols(selectFeatures)
      .setOutputCol("feat_vec").transform(df)

    var trainSet = df.filter("day < 6")
    var testSet = df.filter("day == 7")
    trainSet = trainSet.select("instance_id", "is_trade", "feat_vec")
    testSet = testSet.select("instance_id", "is_trade", "feat_vec")

    //用GBDT训练和预测
    val model = new GBTRegressor()
      .setMaxIter(91)
      .setMinInstancesPerNode(10)
      .setImpurity("gini")
      .setMaxDepth(5)
      .setStepSize(0.1)
      .setSeed(13)
      .setFeaturesCol("feat_vec").setLabelCol("is_trade").fit(trainSet);

    System.out.println("trainSet schema:")
    trainSet.printSchema()

    testSet = model.setFeaturesCol("feat_vec").transform(testSet);

    System.out.println("testSet schema:")
    testSet.printSchema()
    System.out.println("LogLoss = " + Utils.LogLoss(testSet))

    sc.stop();
  }
}
