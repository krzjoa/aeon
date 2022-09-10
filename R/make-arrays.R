#' Prepare arrays (tensors), which can be used a
#'
#' @inheritParams as_3d_array
#' @param lookback The length of the context from the past
#' @param horizon The forecast length
#' @param stride Stride of the moving window
#' @param shuffle Shuffle samples. Set `FALSE` for the test dataset.
#' @param sample_frac
#' @param target Target variable(s)
#' @param numeric Numeric variables
#' @param categorical Categorical variables
#' @param static Static variables
#'
#' @examples
#' library(m5)
#'
#' make_arrays(
#'   data = tiny_m5,
#'   key = c('item_id', 'store_id'),
#'   index = 'date',
#'   lookback = 28,
#'   horizon = 28
#' )
#'
#' @export
make_arrays <- function(data, key, index, lookback,
                        horizon, stride=1, shuffle=TRUE,
                        sample_frac = 1.,
                        target, numeric=NULL, categorical=NULL, static=NULL,
                        past=NULL, future=NULL, keep_all_ids=TRUE, ...){

  # Start of each time series we can identify with unique key
  total_window_length <- lookback + horizon

  ts_starts <-
    data[, .(start_time = min(get(index)),
             end_time = max(get(index)) - total_window_length),
         by = eval(key)]

  ts_starts[, window_starts := Map(\(x, y) seq(x, y, stride),
                                   ts_starts$start_time,
                                   ts_starts$end_time)]

  if (sample_frac < 1.)
    ts_starts <-
      ts_starts[,.SD[sample(.N,as.integer(floor(.N * sample_frac)))],by = a]

  # Consider using Rcpp here to speed it up
  # One loop instead of separate ones
  fo


  # TODO: check, if all the items has at least length of the full window

  ts_starts
}
