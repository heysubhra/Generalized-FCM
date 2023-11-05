# https://cran.r-project.org/web/packages/fcm/vignettes/vignettes.html
library(fcm)
library(stringr)

parse_args <- function(args) {
  # Get the act.vec and w.mat command line arguments
  act_vec_arg <- args[[1]]
  col_names_arg <- args[[2]]
  w_mat_file <- args[[3]]
  num_iter <- args[[4]]

  # Convert the act.vec and w.mat command line arguments to vectors
  act_vec <- as.numeric(unlist(strsplit(act_vec_arg, ",")))
  col_names <- as.matrix(unlist(strsplit(col_names_arg, ",")), ncol = length(act_vec))

  # Return the act.vec and w.mat vectors
  return(list(act_vec = act_vec,  col_names =  col_names, w_mat_file = w_mat_file, iter= num_iter))
}


infer<- function(act.vec,w.mat,iter){
  output <- fcm.infer(act.vec, w.mat, iter:iter, "r", "s", lambda = 2, e = 0.0001)
  return(output)
}

main <- function() {
  # Get the act.vec and w.mat vectors from the command line arguments
  args <- commandArgs(trailingOnly = TRUE)
  act_vec_col_names <- parse_args(args)
  act_vec <-  act_vec_col_names$act_vec
  col_names <-  act_vec_col_names$col_names
  w_mat_file <-  act_vec_col_names$w_mat_file
  iter <-  act_vec_col_names$iter


  w_mat <- read.csv(w_mat_file)
  colnames(act_vec)<-colnames(col_names)

  result <- infer(act_vec, w_mat,iter)

  iterations <- as.numeric(rownames(result$values))
  df <- data.frame(iterations, result$values)

  filename <- paste("simulation_result_",w_mat_file)
  filename <- str_replace_all(filename," ", "")
  write.csv(df, filename)

  return(filename)

}
suppressWarnings({
  main()
})