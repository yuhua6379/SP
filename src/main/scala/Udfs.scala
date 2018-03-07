import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._

object Udfs {
  def Register(session: SparkSession): UserDefinedFunction = {
    session.udf.register("LogLoss", _LogLoss)
    session.udf.register("CombineVec", _CombineVec)
  }

  val _LogLoss = (probability: org.apache.spark.ml.linalg.Vector, label: Double) => {
    val predict = probability.toArray(1)
    label * Math.log(predict) + (1 - label) * Math.log(1 - predict);
  }
  val LogLoss = udf(_LogLoss)

  val _CombineVec = (v1:SparseVector, v2:SparseVector) => {
    val size = v1.size + v2.size
    val maxIndex = v1.size
    val indices = v1.indices ++ v2.indices.map(e => e + maxIndex)
    val values = v1.values ++ v2.values
    new SparseVector(size, indices, values)
  }
  val CombineVec = udf(_CombineVec)
}
