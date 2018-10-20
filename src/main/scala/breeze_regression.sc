/*
* Reading train get mean log1p of the prices and submit it
*/

import scala.io.Source.fromFile
import scala.math.{log1p,expm1, sqrt}
import java.io.{BufferedWriter, FileWriter, File}
import breeze.linalg._
import breeze.numerics._

// number parser
def read_number(num_string: String): Int = {
  if (num_string == "NA") 0 else num_string.toInt
}

//Root mean square log error result
def rmsle_metric(targets: Array[Double], preds: Array[Double]): Double = {
  val log_preds = for (pred <- preds) yield log1p(pred)
  val log_targets = for (target <- targets) yield log1p(target)
  sqrt((for ((t, p) <- log_targets zip log_preds) yield (t-p) * (t-p)).sum)
}

// read file iterator
val train_path = "/home/yair/IdeaProjects/scala_ml_101/train.csv"
val lines = for (line <- fromFile(train_path).getLines()) yield line.toString

// read first line - columns
val columns = lines.next().split(",")

// features used
val cols_chosen = List("LotFrontage", "LotArea", "OverallQual", "OverallCond", "SalePrice")
// get relevant indices for chosen columns
val column_map = columns.zipWithIndex.toMap
val col_indices_chosen = (for (col <- cols_chosen) yield column_map(col)).toArray
val n_col = col_indices_chosen.length

// line parser for features
def read_line(line: String): Array[Int] = {
  val line_splited = line.split(",")
  for (index <- col_indices_chosen) yield read_number(line_splited(index))
}

// read rest of lines to array
val values = (for(line <- lines) yield read_line(line)).toArray
val n_row = values.length

// create array of log1p of prices
val targets = for(line <- values) yield log1p(line(n_col-1).toDouble)
val mean_price_log = targets.sum / n_row

// comparing targets with an array of size target full of mean_price_log
rmsle_metric(targets, for (_ <- targets) yield mean_price_log)


// changing back from log1p to normal price
val mean_price_prediction = expm1(mean_price_log)

// read file iterator
val sample_sub_path = "/home/yair/IdeaProjects/scala_ml_101/sample_submission.csv"
val sub_lines = for (line <- fromFile(sample_sub_path).getLines()) yield line.toString

val sub_header = sub_lines.next()
val sub_ids = for(line <- sub_lines) yield line.split(",")(0)

//Writing submission
val sub_name = "/home/yair/IdeaProjects/scala_ml_101/first_scala_regression_submission.csv"

def submit(file_name: String): Unit ={
  val file = new File(file_name)
  val bw = new BufferedWriter(new FileWriter(file))
  try {
    bw.write(sub_header+"\n")
    for (id <- sub_ids) bw.append(id + "," + mean_price_prediction.toString + "\n")
  }
  finally
    bw.close()
}

submit(sub_name)
