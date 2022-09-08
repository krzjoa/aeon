#' Create a 3-dimensional array out of a data.frame
#'
#' @param data A data.frame object
#' @param index A time-related column
#' @param key
#'
#' @examples
#' library(tsibbledata)
#' library(dplyr)
#'
#' # We'll use the `global_economy` dataset
#' global_economy
#'
#' selected_ge <-
#'  global_economy %>%
#'  select(Country, Year, Imports, Exports)
#'
#'
#'
#' @export
as_3d_array <- function(data, index, key){

  # browser()

  cols <- colnames(data)
  rest <- setdiff(cols, c(key, index))

  col_order   <- c(key, index)
  # col_order_2 <- c("n", key)



  batch_dim   <- dplyr::n_distinct(select(data, !!key))
  time_dim    <- dplyr::n_distinct(select(data, !!index))
  feature_dim <- length(rest)

  data %>%
    arrange(across(all_of(key))) %>%
    select(-all_of(col_order)) %>%
    as.matrix() %>%
    array(dim = c(batch_dim, time_dim, feature_dim))
}

