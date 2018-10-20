import scala.io.Source.fromFile
import scala.math.{log1p,expm1}

// read file iterator
val train_path = "C:\\Users\\yaia1\\Documents\\kaggle\\house\\train.csv"
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

val mean_price = expm1(mean_price_log)



