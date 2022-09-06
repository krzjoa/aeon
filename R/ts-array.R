#' Create a 3-dimensional array out of a data.frame
#'
#' @data A data.frame object
#' @index A time-related column
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

  cols <- colnames(data)
  rest <- setdiff(cols, c(key, index))

  col_order   <- c(key, index)
  # col_order_2 <- c("n", key)

  batch_dim   <- dplyr::n_distinct(data[key])
  time_dim    <- dplyr::n_distinct(data[index])
  feature_dim <- length(rest)

  # array3d <-
  #   data %>%
  #   arrange(!!col_order) %>%
  #   group_by(!!key) %>%
  #   mutate(n = 1:n()) %>%
  #   arrange(col_order_2) %>%
  #   array(dim = c())

  # browser()

  data %>%
    group_by(!!key) %>%
    arrange(Year, Country) %>%
    select(-col_order) %>%
    #arrange(all_of(col_order)) %>%
    as.matrix() %>%
    array(dim = c(batch_dim, time_dim, feature_dim))
}


# data <- selected_ge
#
# index <- "Year"
# key <- "Country"
# features <- ""
#
#

#arr <-
#  data %>%
#  select(Year, Country, Imports, Exports) %>%
#  as_3d_array("Year", "Country")

