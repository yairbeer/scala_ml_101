import scala.io.Source.fromFile
import scala.math.{log1p,expm1, sqrt}
import java.io.{BufferedWriter, FileWriter, File}
import breeze.linalg._
import breeze.numerics._

object breezeRegression extends App {
  /*
  * Reading train get mean log1p of the prices and submit it
  */



  // number parser
  def read_number(num_string: String): Double = {
    if (num_string == "NA") 0 else num_string.toDouble
  }

  def coef_prediction(x_hat: DenseMatrix[Double], x: DenseMatrix[Double]): DenseVector[Double] = {
    val x_hat_mat = tile(x_hat.t, x.rows)
    sum(x *:* x_hat_mat, Axis._1)
  }

  //Root mean square log error result
  def rmsle_metric(targets: Array[Double], preds: Array[Double]): Double = {
    val log_preds = for (pred <- preds) yield log1p(pred)
    val log_targets = for (target <- targets) yield log1p(target)
    sqrt((for ((t, p) <- log_targets zip log_preds) yield (t-p) * (t-p)).sum)
  }

  // line parser for features
  def read_line(line: String, col_indices: Array[Int]): Array[Double] = {
    val line_splited = line.split(",")
    for (index <- col_indices) yield read_number(line_splited(index))
  }

  /*
  * Train
  */
  // read file iterator
  val train_path = "/home/yair/IdeaProjects/scala_ml_101/train.csv"
  val train_lines = for (line <- fromFile(train_path).getLines()) yield line.toString

  // read first line - columns
  val columns = train_lines.next().split(",")

  // features used
  val cols_chosen = List("LotFrontage", "LotArea", "OverallQual", "OverallCond", "SalePrice")
  // get relevant indices for chosen columns
  val column_map = columns.zipWithIndex.toMap
  val col_indices_chosen = (for (col <- cols_chosen) yield column_map(col)).toArray
  val n_col = col_indices_chosen.length

  // read rest of lines to array into features and target
  val train_values = (for(line <- train_lines) yield read_line(line, col_indices_chosen)).toArray
  val targets = for(line <- train_values) yield log1p(line(n_col-1))
  val n_rows = train_values.length

  // Creating features
  val train = DenseMatrix((for(line <- train_values) yield line):_*)
  val train_x = DenseMatrix.horzcat(DenseMatrix.ones[Double](n_rows, 1), train(::, 0 to -2))
  val x_hat = inv(train_x.t * train_x) * train_x.t * DenseMatrix(targets:_*)
  println("x_hat: " + x_hat.t)
  val predictions = coef_prediction(x_hat, train_x)


  // create array of log1p of prices
  val mean_price_log = targets.sum / n_rows
  // changing back from log1p to normal price
  val mean_price_prediction = expm1(mean_price_log)

  // comparing targets with an array of size target full of mean_price_log
  println("Mean log RMSLE: " + rmsle_metric(targets, for (_ <- targets) yield mean_price_log))
  println("Linear Regression RMSLE: " + rmsle_metric(targets, predictions.toArray))

  /*
  * Test
  */
  val test_path = "/home/yair/IdeaProjects/scala_ml_101/test.csv"
  val test_lines = for (line <- fromFile(test_path).getLines()) yield line.toString
  test_lines.next()

  // features used
  val cols_chosen_test = List("LotFrontage", "LotArea", "OverallQual", "OverallCond")

  // get relevant indices for chosen columns
  val column_map_test = columns.zipWithIndex.toMap
  val col_indices_chosen_test = (for (col <- cols_chosen_test) yield column_map_test(col)).toArray

  // read rest of lines to array into features and target
  val test_values = (for(line <- test_lines) yield read_line(line, col_indices_chosen_test)).toArray
  val n_rows_test = test_values.length

  // Creating features
  val test = DenseMatrix(test_values:_*)
  val test_x = DenseMatrix.horzcat(DenseMatrix.ones[Double](n_rows_test, 1), test)


  val predictions_log_test = coef_prediction(x_hat, test_x).toArray
  val predictions_test = for (pred <- predictions_log_test) yield expm1(pred)
  /*
  * Submission
  */
  // read file iterator
  val sample_sub_path = "/home/yair/IdeaProjects/scala_ml_101/sample_submission.csv"
  val sub_lines = for (line <- fromFile(sample_sub_path).getLines()) yield line.toString

  val sub_header = sub_lines.next()
  val sub_ids = (for(line <- sub_lines) yield line.split(",")(0)).toArray

  //Writing submission
  val sub_name = "/home/yair/IdeaProjects/scala_ml_101/first_scala_regression_submission.csv"

  def submit(file_name: String, predictions: Array[Double]): Unit ={
    val file = new File(file_name)
    val bw = new BufferedWriter(new FileWriter(file))
    try {
      bw.write(sub_header+"\n")
      for ((id, pred) <- sub_ids zip predictions) bw.append(id + "," + pred.toString + "\n")
    }
    finally
      bw.close()
  }

  submit(sub_name, predictions_test)

}
