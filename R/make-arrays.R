resolve_variables <- function(period_set, type_set){
  if (is.null(period_set))
    return(type_set)
  base::intersect(period_set, type_set)
}

.get_slice <- function(data, slice){

}



.get_arrays <- function(data, lookback,
                        horizon=NULL, col_list){

  # tensors <- Map(\(x))

  abind::abind(tensors, along = 1)
}


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
#' library(recipes)
#' library(zeallot)
#'
#' train <- tiny_m5[date < '2016-01-01']
#' test  <- tiny_m5[date >= '2016-01-01']
#'
#' m5_recipe <-
#'    recipe(value ~ ., data=train) %>%
#'    step_mutate(item_id_idx=item_id, store_id_idx=store_id) %>%
#'    step_integer(item_id_idx, store_id_idx,
#'                 wday, month,
#'                 event_name_1, event_type_1,
#'                 event_name_2, event_type_2,
#'                 zero_based=TRUE) %>%
#'    prep()
#'
#' train <- bake(m5_recipe, train)
#' test  <- bake(m5_recipe, test)
#'
#' STATIC      <- c('item_id_idx', 'store_id_idx')
#' CATEGORICAL <- c('item_id_idx', 'store_id_idx')
#' NUMERIC     <- c('')
#'
#' setDT(train)
#'
#' c(data, index_table) %<-%
#'    make_arrays(
#'        data = train,
#'        key = c('item_id', 'store_id'),
#'        index = 'date',
#'        lookback = 28,
#'        horizon = 28,
#'        static=STATIC,
#'        categorical=CATEGORICAL
#'    )
#'
#' @export
make_arrays <- function(data, key, index, lookback,
                        horizon, stride=1, shuffle=TRUE, sample_frac = 1.,
                        target, numeric=NULL, categorical=NULL, static=NULL,
                        past=NULL, future=NULL, keep_all_ids=TRUE, ...){

  # Start of each time series we can identify with unique key
  total_window_length <- lookback + horizon

  ts_starts <-
    data[, .(start_time = min(get(index)),
             end_time = max(get(index)) - total_window_length),
         by = eval(key)]

  # Types
  ts_starts[, window_start := Map(\(x, y) seq(x, y, stride),
                                   ts_starts$start_time,
                                   ts_starts$end_time)]

  # https://stackoverflow.com/questions/15659783/why-does-unlist-kill-dates-in-r
  ts_starts <- ts_starts[, .(window_start = do.call('c', window_start)),
                         by = eval(key)]

  if (sample_frac < 1.)
    ts_starts <-
      ts_starts[,.SD[sample(.N,as.integer(floor(.N * sample_frac)))]
                ,by = eval(key)]

  # Consider using Rcpp here to speed it up
  # One loop instead of separate ones
  # Static
  if (!is.null(static))
    static_categorical <- resolve_variables(static, categorical)
    static_numeric     <- resolve_variables(static, numeric)

    ext_static_cat <- c(static_categorical, static_numeric, key, index)
    static_features <-
      merge(ts_starts, data[, ..ext_static_cat],
            by.x = c(key, 'window_start'),
            by.y = c(key, index))

    if (!is.null(static_categorical))
      X_static_cat <-
        as.matrix(static_features[, ..static_categorical])

    if (!is.null(static_numeric))
      X_static_num <-
        as.matrix(static_features[, ..static_numeric])

  # Auxiliary IDs
  batch_id_cols <- c(key, 'window_start')
  ts_starts[
    , aux_id := do.call(paste, c(.SD, sep = "-")),
    .SDcols=batch_id_cols
  ]

  # Sort & create indices
  setorderv(data, c(key, index))

  index_table <-
    data[, .(n = .N), by = eval(key)] %>%
    .[, start_idx := cumsum(lag(n, default = 1))]


  # past_dates <- copy(ts_starts)
  # past_dates[, date := Map(\(x, y) seq(x, y, 1),
  #                                   ts_starts$window_start,
  #                                   ts_starts$window_start + lookback)]

  # browser()

  # past_dates <- past_dates[
  #   , .(date = do.call('c', date)), by = .I
  # ]





  # PAST
  # past_categorical <- resolve_variables(past, categorical)
  # past_numeric     <- resolve_variables(past, numeric)
  #
  # past_cols <-
  #   list(y_past       = target,
  #        X_past_cat   = past_categorical,
  #        X_past_num   = past_numeric)
  #
  # past_values <-
  #   .get_arrays(data, ts_starts, past_cols)
  #
  #
  # # FUTURE
  # future_categorical <- resolve_variables(future, categorical)
  # future_numeric     <- resolve_variables(future, numeric)
  #
  # fut_values <-
  #   .get_arrays(data, lookback, ts_starts, fut_cols)
  #
  # # TODO: check, if all the items has at least length of the full window
  # output <-
  #   c(
  #     static_values,
  #     past_values,
  #     fut_values
  #   )

  list(data, index_table)
}
