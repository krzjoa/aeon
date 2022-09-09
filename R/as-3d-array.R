#' Create a 3-dimensional array out of a data.frame
#'
#' The `key` and `index` are used to create `batch_size` and `timesteps`
#' dimension respectively. At the end, they excluded from the data.frame,
#' so they're not present anymore in the final 3-dimensional array.
#' All the time series lengths must be equal.
#'
#' For example, if we have a subset of the `global_economy` dataset from
#' the `{tibbledata}` package, with 263 countries (key), 58 years (index) and
#' two features assigned to each country/year pair, the output array will have
#' shape (263, 58, 2).
#' Bear in mind that is only a toy example, because we haven't split
#' the dataset into at least two parts to have historical context from the past
#' and expected future being target to a ML model.
#'
#' It's a basic helper function -to create complete input tensors, please use
#' [make_arrays()] function.
#'
#' @param data A data.frame object
#' @param index A time-related column
#' @param key Columns, which creates unique time series IDs
#'
#' @returns
#' A 3-dimensional array whith the following dimensions:
#' (n_unique_ids, n_timesteps, n_features)
#'
#' @examples
#' library(tsibbledata)
#' library(dplyr, warn.conflicts=FALSE)
#'
#' # `global_economy` dataset comes from the `{tsibbledata}` package
#'
#' selected_ge <-
#'  global_economy %>%
#'  select(Country, Year, Imports, Exports)
#'
#' tensor <- as_3d_array(selected_ge, "Year", "Country")
#' dim(tensor)
#' @export
as_3d_array <- function(data, index, key){
  # TODO: turn this function into S3 method
  # A hack to pass CRAN test as well as build site
  # ..key <- ..index <- ..rest <- NULL

  cols <- colnames(data)
  rest <- setdiff(cols, c(key, index))
  col_order   <- c(key, index)

  setDT(data)
  print(class(data))

  batch_dim   <- nrow(unique(data[, ..key]))
  time_dim    <- nrow(unique(data[, ..index]))
  feature_dim <- length(rest)

  setorderv(data, col_order)

  data[, ..rest] %>%
    as.matrix() %>%
    array(dim = c(batch_dim, time_dim, feature_dim))
}

