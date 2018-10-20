import scala.io.Source.fromFile
import scala.math.{log1p,expm1}
import java.io.{BufferedWriter, FileWriter, File}
// read file iterator
val train_path = "/home/yair/IdeaProjects/scala_ml_101/train.csv"
val lines = for (line <- fromFile(train_path).getLines()) yield line.toString

// read first line
val columns = lines.next().split(",")
val n_col = columns.length

// read rest of lines to array
val values = (for(line <- lines) yield line.split(",")).toArray
val n_row = values.length

// create array of log1p of prices
val target = for(line <- values) yield log1p(line(n_col-1).toFloat)
val mean_price_log = target.sum / n_row

//
val mean_price = expm1(mean_price_log)

// read file iterator
val sample_sub_path = "/home/yair/IdeaProjects/scala_ml_101/sample_submission.csv"
val sub_lines = for (line <- fromFile(sample_sub_path).getLines()) yield line.toString

val sub_header = sub_lines.next()
val sub_ids = for(line <- sub_lines) yield line.split(",")(0)

//Writing submission
val sub_name = "/home/yair/IdeaProjects/scala_ml_101/first_scala_submission.csv"

def submit(file_name: String): Unit ={
  val file = new File(file_name)
  val bw = new BufferedWriter(new FileWriter(file))
  try {
    bw.write(sub_header+"\n")
    for (id <- sub_ids) bw.append(id + "," + mean_price.toString + "\n")
  }
  finally
    bw.close()
}

submit(sub_name)

