#' An utility function for `make_arrays` and `ts_generator`
#'
#' @include utils.R
#' @keywords internal
prepare_idx_and_vars <- function(data, lookback, horizon,
                                 key, index, stride=1,
                                 target, numeric=NULL, categorical=NULL,
                                 static=NULL, past=NULL, future=NULL,
                                 shuffle=TRUE, sample_frac = 1.,
                                 y_past_sep = FALSE){
  setDT(data)

  # Check if data contains gaps
  check_gaps(data, key, index)

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

  # n_steps <- ceiling(nrow(ts_starts) / batch_size)

  # Static
  if (!is.null(static)) {
    static_categorical <- resolve_variables(static, categorical)
    static_numeric     <- resolve_variables(static, numeric)
  } else {
    static_categorical <- static_numeric <- NULL
  }

  #ext_static <- c(static_categorical, static_numeric, key, index)

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
      x_past_num = past_num,
      x_past_cat = past_cat
    )

  fut_var <-
    list(
      y_fut      = target,
      x_fut_num  = fut_num,
      x_fut_cat  = fut_cat
    )

  past_var <- remove_nulls(past_var)
  fut_var  <- remove_nulls(fut_var)

  list(ts_starts, past_var, fut_var,
       static_numeric, static_categorical)
}
