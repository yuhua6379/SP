//package O98K;
import ml.dmlc.xgboost4j.scala.spark.XGBoost
import org.apache.spark.sql.functions._
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.GBTRegressor
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DataTypes

object AliMamaPredictor {
  def main(args: Array[String]): Unit = {
    val session = SparkSession.builder().appName("IJCAI18")
      .getOrCreate();
    var df = session.read.format("csv")
      .option("header", true)
      .option("sep", " ")
      .load("/user/utp/competition/IJCAI-18/training_data/round1_ijcai_18_train_20180301.txt")

    //加速！
    df = df.cache()

    df = df
        //TODO: ID类型特征
        .withColumn("instance_id", col("instance_id").cast(DataTypes.LongType))
        .withColumn("item_id", col("item_id").cast(DataTypes.LongType))
        .withColumn("user_id", col("user_id").cast(DataTypes.LongType))
        .withColumn("shop_id", col("shop_id").cast(DataTypes.LongType))
        .withColumn("context_id", col("context_id").cast(DataTypes.LongType))

        //TODO: 离散型特征
        .withColumn("item_brand_id", col("item_brand_id").cast(DataTypes.LongType))
        .withColumn("item_city_id", col("item_city_id").cast(DataTypes.LongType))
        .withColumn("item_price_level", col("item_price_level").cast(DataTypes.LongType))
        .withColumn("item_sales_level", col("item_sales_level").cast(DataTypes.LongType))
        .withColumn("item_collected_level", col("item_collected_level").cast(DataTypes.LongType))
        .withColumn("item_pv_level", col("item_pv_level").cast(DataTypes.LongType))
        .withColumn("user_gender_id", col("user_gender_id").cast(DataTypes.LongType))
        .withColumn("user_age_level", col("user_age_level").cast(DataTypes.LongType))
        .withColumn("user_occupation_id", col("user_occupation_id").cast(DataTypes.LongType))
        .withColumn("user_star_level", col("user_star_level").cast(DataTypes.LongType))
        .withColumn("context_page_id", col("context_page_id").cast(DataTypes.LongType))
        .withColumn("shop_review_num_level", col("shop_review_num_level").cast(DataTypes.LongType))
        .withColumn("shop_star_level", col("shop_star_level").cast(DataTypes.LongType))

        //TODO: 连续型数据
        .withColumn("shop_score_service", col("shop_score_service").cast(DataTypes.DoubleType))
        .withColumn("shop_score_delivery", col("shop_score_delivery").cast(DataTypes.DoubleType))
        .withColumn("shop_score_description", col("shop_score_description").cast(DataTypes.DoubleType))
        .withColumn("shop_review_positive_rate", col("shop_review_positive_rate").cast(DataTypes.DoubleType))

        //TODO:时间
        .withColumn("context_timestamp", col("context_timestamp").cast(DataTypes.LongType))

        //TODO:复合型特征
        .drop("item_category_list")
        .drop("item_property_list")
        .drop("predict_category_property")

        //TODO: 预测目标
        .withColumn("is_trade", col("is_trade").cast(DataTypes.LongType))

        //TODO:null应该filter还是fill zero更好？
        .na.fill(0)

    System.out.println("原始数据的Schema:")
    df.printSchema()

    //抽离除了is_trade（label）以外的列名，做Assemble
//    val _trainingFields = df.schema.fields;
//    val trainingFields = new Array[String](_trainingFields.length - 1)
//    var i = 0
//    for (v <- _trainingFields){
//      if (v.name.equals("is_trade") == false){
//        trainingFields.update(i, v.name)
//        i += 1
//      }
//    }

    val enumFields = Array(
      "shop_id",
      "item_id",
      "item_brand_id",
      "item_city_id",
      "item_price_level",
      "item_sales_level",
      "item_collected_level",
      "item_pv_level",
      "user_gender_id",
      "user_age_level",
      "user_occupation_id",
      "user_star_level",
      "context_page_id",
      "shop_review_num_level",
      "shop_star_level");

    val numericFields = Array(
      "shop_score_service",
      "shop_score_delivery",
      "shop_score_description",
      "shop_review_positive_rate"
    )



    df = Utils.OneHot(enumFields, df)

    System.out.println("OneHot之后的Schema:")
    df.printSchema()
    df.show(10, false)

    //Assemble!
    df = new VectorAssembler()
      .setInputCols(enumFields ++ numericFields)
      .setOutputCol("feat_vec").transform(df)

    df.select("feat_vec", "is_trade")

    System.out.println("Assembler之后的Schema:")
    df.printSchema()
    df.show(10, false)

    //抽样用于训练和测试，666是种子，保证数据是每次一样的伪随机抽样
    val dfs = df.randomSplit(Array(0.05,0.95), 666);

    var testSet = dfs(0)
    var trainSet = dfs(1)


//    //用XGBoost跑
//    val numRound = 50
//    val nWorkers = 64
//    val useExternalMemory = true
//    val paramMap = List(
//      "eta" -> 0.1f,
//      "max_depth" -> 2,
//      "objective" -> "binary:logistic").toMap
//
//    val model = XGBoost.trainWithDataFrame(
//      trainingData = trainSet,
//      params = paramMap,
//      round = numRound,
//      nWorkers = nWorkers ,
//      useExternalMemory = true,
//      missing=0.0f,
//      featureCol = "feat_vec",
//      labelCol = "is_trade")
//
//    System.out.println("trainSet schema:")
//    trainSet.printSchema()
//
//    testSet = model.setFeaturesCol("feat_vec").transform(testSet);
//
//    System.out.println("testSet schema:")
//    testSet.printSchema()
//    System.out.println("LogLoss = " + Utils.LogLoss(testSet))

//    //用LR训练和预测
//    val model = new LogisticRegression()
//      .setRegParam(0.0)
//      .setMaxIter(100)
//      .setTol(1e-7)
//      .setElasticNetParam(0)
//      .setFeaturesCol("feat_vec")
//      .setLabelCol("is_trade").fit(trainSet)
//
//    System.out.println("trainSet schema:")
//    trainSet.printSchema()
//
//    testSet = model.setFeaturesCol("feat_vec").transform(testSet);
//
//    System.out.println("testSet schema:")
//    testSet.printSchema()
//    System.out.println("LogLoss = " + Utils.LogLoss(testSet))

//    //用GBDT训练和预测
//    val model = new GBTRegressor()
//      .setMaxIter(100)
//      .setMinInstancesPerNode(10)
//      .setImpurity("gini")
//      .setMaxDepth(20)
//      .setStepSize(0.1)
//      .setSeed(13)
//      .setFeaturesCol("feat_vec").setLabelCol("is_trade").fit(trainSet);
//
//    System.out.println("trainSet schema:")
//    trainSet.printSchema()
//
//    testSet = model.setFeaturesCol("feat_vec").transform(testSet);
//
//    System.out.println("testSet schema:")
//    testSet.printSchema()
//    System.out.println("LogLoss = " + Utils.LogLoss(testSet))
  }
}
