#' Create a generator for time series data
#'
#' The advantage of the generator over the explicit arrays creation by `make_arrays`
#' at the beginning is the lower RAM space volume needed for this kind of operation.
#' When full arrays are created, we have to allocate space for all the examples
#' selected from the passed `data.frame`. If we use `ts_generator` instead,
#' the following examples are deliver as the batch sub-arrays and created on fly.
#' It means that we don't have to store all the examples in RAM at the same time.
#'
#' @inheritParams make_arrays
#' @param batch_size Batch size
#'
#' @include utils.R
#'
#' @examples
#' library(m5)
#' library(recipes, warn.conflicts=FALSE)
#' library(zeallot)
#' library(dplyr, warn.conflicts=FALSE)
#' library(data.table, warn.conflicts=FALSE)
#'
#' # ==========================================================================
#' #                          PREPARING THE DATA
#' # ==========================================================================
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
#'    step_naomit(all_predictors()) %>%
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
#' BATCH_SIZE  <- 32
#'
#' setDT(train)
#' setDT(test)
#'
#' # ==========================================================================
#' #                          CREATING GENERATORS
#' # ==========================================================================
#'
#' c(train_generator, train_steps) %<-%
#'    ts_generator(
#'        data = train,
#'        key = KEY,
#'        index = INDEX,
#'        lookback = LOOKBACK,
#'        horizon = HORIZON,
#'        stride = STRIDE,
#'        target=TARGET,
#'        static=STATIC,
#'        categorical=CATEGORICAL,
#'        numeric=NUMERIC,
#'        batch_size=BATCH_SIZE
#'    )
#'
#' batch <- train_generator()
#' print(names(batch))
#'
#' test_generator <-
#'    ts_generator(
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
#' @export
ts_generator <- function(data, key, index,
                         lookback, horizon, stride=1,
                         target, numeric=NULL, categorical=NULL,
                         static=NULL, past=NULL, future=NULL,
                         shuffle=TRUE, sample_frac = 1.,
                         y_past_sep = FALSE, batch_size=1, ...){

  setDT(data)

  # Start of each time series we can identify with unique key
  total_window_length <- lookback + horizon
  key_index <- c(key, index)

  ts_starts <-
    data[, .(start_time = min(get(index)),
             end_time = max(get(index))),
         by = eval(key)]

  if (any(ts_starts$end_time - ts_starts$start_time < total_window_length))
    warning("Found samples with end_time - start_time < total_window_length. They'll be removed.")

  ts_starts <-
    ts_starts[end_time - start_time  >= total_window_length]

  ts_starts[, end_time := end_time - total_window_length]

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

  n_steps <- ceiling(nrow(ts_starts) / batch_size)

  # Static
  if (!is.null(static)) {
    static_categorical <- resolve_variables(static, categorical)
    static_numeric     <- resolve_variables(static, numeric)
  } else {
    static_categorical <- static_numeric <- NULL
  }

  ext_static <- c(static_categorical, static_numeric, key, index)

  # Resolve variables
  past_cat <- resolve_variables(past, categorical)
  past_num <- resolve_variables(past, numeric)
  fut_cat  <- resolve_variables(future, categorical)
  fut_num  <- resolve_variables(future, numeric)

  setnames(ts_starts, 'window_start', index)

  # Sort & create indices
  setorderv(data, c(key, index))
  setorderv(ts_starts, c(key, index))

  indices <- data[, ..key_index][, row_idx := 1:.N]
  ts_starts[indices, row_idx := row_idx, on=(eval(key_index))]
  rm(indices)
  gc()

  # Convert to numeric - required by the Rcpp function, which uses NumericVector
  all_dynamic_vars <- c(categorical, numeric, target)

  for (v in all_dynamic_vars)
    data.table::set(data, j = v, value = as.numeric(data[[v]]))

  if (!y_past_sep) {
    past_num <- c(target, past_num)
    target_past <- NULL
  } else {
    target_past <- target
  }

  # Dynamic variables
  past_var <-
    list(
      y_past     = target_past,
      X_past_num = past_num,
      X_past_cat = past_cat
    )

  fut_var <-
    list(
      y_fut      = target,
      X_fut_num  = fut_num,
      X_fut_cat  = fut_cat
    )

  past_var <- remove_nulls(past_var)
  fut_var  <- remove_nulls(fut_var)

  i <- 0

  generator <- function(){

    # Reset counter
    if (i == n_steps)
      i <<- 0

    start     <- i * batch_size + 1
    samples   <- ts_starts[start:(start + batch_size - 1), ]

    rows <- samples$row_idx

    static_arrays <- list()

    if (length(static_categorical) > 0)
      static_arrays$X_static_cat <-
          as.matrix(data[rows, ..static_categorical])

    if (length(static_numeric) > 0)
      static_arrays$X_static_num <-
          as.matrix(data[rows, ..static_numeric])

    dynamic_arrays <-
      get_arrays(
        data           = data,
        ts_starts      = samples,
        lookback       = lookback,
        horizon        = horizon,
        past_var       = past_var,
        fut_var        = fut_var
      )

    c(dynamic_arrays, static_arrays)
  }

  list(generator, n_steps)
}
