library(m5)
library(recipes)
library(data.table)
library(dplyr)

test_that("Test get_arrays", {

  train <- tiny_m5[date < '2016-01-01']
  test  <- tiny_m5[date >= '2016-01-01']

  m5_recipe <-
     recipe(value ~ ., data=train) %>%
     step_mutate(item_id_idx=item_id, store_id_idx=store_id) %>%
     step_integer(item_id_idx, store_id_idx,
                  wday, month,
                  event_name_1, event_type_1,
                  event_name_2, event_type_2,
                  zero_based=TRUE) %>%
     prep()

  train <- bake(m5_recipe, train)
  #test  <- bake(m5_recipe, test)

  categorical <- c('event_name_1', 'event_type_1')
  target      <- "value"
  numeric     <- 'sell_price'

  setDT(train)

  data     <- train
  lookback <- 28
  horizon  <- 14
  key      <- c('item_id', 'store_id')
  index    <- 'date'
  stride   <- 28
  key_index <- c(key, index)

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

  # if (sample_frac < 1.)
  #   ts_starts <-
  #   ts_starts[,.SD[sample(.N,as.integer(floor(.N * sample_frac)))]
  #             ,by = eval(key)]

  setnames(ts_starts, 'window_start', index)

  # Sort & create indices
  setorderv(data, c(key, index))
  data[, row_idx := 1:.N]
  ts_starts[data, row_idx := row_idx, on=(eval(key_index))]

  # Convert all variable first?
  all_vars <- c(categorical, target)

  for (v in all_vars)
    set(train, j = v, value = as.numeric(train[[v]]))

  x <-
    aion:::get_arrays(
      data = train,
      ts_starts = ts_starts,
      lookback = lookback,
      horizon = horizon,
      past_var = list(X_past_cat=categorical, X_past_num=numeric, y_past=target),
      fut_var = list(X_fut_cat=categorical)
    )

  # ============================================================================
  #                         CHECKING THE RESULTS
  # ============================================================================

  # Past
  row_idx <- 356
  col_idx <- 1
  idx_start <- ts_starts[row_idx]$row_idx
  expect_identical(
    x$y_past[row_idx, 1:lookback, col_idx],
    data[idx_start:(idx_start+lookback-1), get(target[col_idx])]
  )

  row_idx <- 1
  col_idx <- 1
  idx_start <- ts_starts[row_idx]$row_idx
  expect_identical(
    x$X_past_cat[row_idx, 1:lookback, col_idx],
    data[idx_start:(idx_start+lookback-1), get(categorical[col_idx])]
  )

  # Future
  row_idx <- 1
  col_idx <- 1
  idx_start <- ts_starts[row_idx]$row_idx + lookback
  expect_identical(
    x$X_fut_cat[row_idx, 1:horizon, col_idx],
    data[idx_start:(idx_start+horizon-1), get(categorical[col_idx])]
  )

  row_idx <- 53
  col_idx <- 2
  idx_start <- ts_starts[row_idx]$row_idx + lookback
  expect_identical(
    x$X_fut_cat[row_idx, 1:horizon, col_idx],
    data[idx_start:(idx_start+horizon-1), get(categorical[col_idx])]
  )

})
