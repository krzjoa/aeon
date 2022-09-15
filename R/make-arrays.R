#' @keywords internal
resolve_variables <- function(period_set, type_set){
  if (is.null(period_set))
    return(type_set)
  base::intersect(period_set, type_set)
}

#' Prepare input/taget arrays for time series models
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
#' @returns
#' A list of arrays. The maximal possible content embraces eight arrays:
#' - **y_past**
#' - **X_past_cat**
#' - **X_past_num**
#' - **y_fut**
#' - **X_fut_cat**
#' - **X_fut_num**
#' - **X_static_cat**
#' - **X_static_num**
#'
#' @examples
#' library(m5)
#' library(recipes, warn.conflicts=FALSE)
#' library(zeallot)
#' library(dplyr, warn.conflicts=FALSE)
#' library(data.table, warn.conflicts=FALSE)
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
#' TARGET      <- 'value'
#' STATIC      <- c('item_id_idx', 'store_id_idx')
#' CATEGORICAL <- c('event_name_1', 'event_type_1', STATIC)
#' NUMERIC     <- c('sell_price', 'sell_price')
#' KEY         <- c('item_id', 'store_id')
#' INDEX       <- 'date'
#' LOOKBACK    <- 28
#' HORIZON     <- 14
#' STRIDE      <- LOOKBACK
#'
#' setDT(train)
#' setDT(test)
#'
#' train_arrays <-
#'    make_arrays(
#'        data = train,
#'        key = KEY,
#'        index = INDEX,
#'        lookback = LOOKBACK,
#'        horizon = HORIZON,
#'        stride = STRIDE,
#'        target=TARGET,
#'        static=STATIC,
#'        categorical=CATEGORICAL,
#'        numeric=NUMERIC
#'    )
#'
#' print(names(train_arrays))
#' print(dim(train_arrays$X_past_num))
#'
#' test_arrays <-
#'    make_arrays(
#'        data = test,
#'        key = KEY,
#'        index = INDEX,
#'        lookback = LOOKBACK,
#'        horizon = HORIZON,
#'        stride = STRIDE,
#'        target=TARGET,
#'        static=STATIC,
#'        categorical=CATEGORICAL,
#'        numeric=NUMERIC
#'    )
#'
#' print(names(test_arrays))
#' print(dim(test_arrays$X_past_num))
#' @export
make_arrays <- function(data, key, index, lookback,
                        horizon, stride=1, shuffle=TRUE, sample_frac = 1.,
                        target, numeric=NULL, categorical=NULL, static=NULL,
                        past=NULL, future=NULL, ...){

  # Start of each time series we can identify with unique key
  total_window_length <- lookback + horizon
  key_index <- c(key, index)

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

  static_arrays <-
    list(X_static_cat=X_static_cat,
         X_static_num=X_static_num)

  setnames(ts_starts, 'window_start', index)

  # Sort & create indices
  setorderv(data, c(key, index))
  setorderv(ts_starts, c(key, index))

  data[, row_idx := 1:.N]
  ts_starts[data, row_idx := row_idx, on=(eval(key_index))]

  # Resolve variables
  past_cat <- resolve_variables(past, categorical)
  past_num <- resolve_variables(past, numeric)
  fut_cat  <- resolve_variables(future, categorical)
  fut_num  <- resolve_variables(future, numeric)

  # Convert to numeric - required by the Rcpp function, which uses NumericVector
  all_dynamic_vars <- c(categorical, numeric, target)

  for (v in all_dynamic_vars)
    data.table::set(data, j = v, value = as.numeric(data[[v]]))

  # Dynamic variables
  past_var <-
    list(
      y_past     = target,
      X_past_num = past_num,
      X_past_cat = past_cat
    )

  fut_var <-
    list(
      y_fut      = target,
      X_fut_num  = fut_num,
      X_fut_cat  = fut_cat
    )

  dynamic_arrays <-
    get_arrays(
      data           = data,
      ts_starts      = ts_starts,
      lookback       = lookback,
      horizon        = horizon,
      past_var       = past_var,
      fut_var        = fut_var
    )

  c(dynamic_arrays, static_arrays)
}
