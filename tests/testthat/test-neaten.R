# Dataset
library(m5)

# Neural Networks
library(aion)
library(keras)

# Data wrangling
library(dplyr, warn.conflicts=FALSE)
library(data.table, warn.conflicts=FALSE)
library(recipes, warn.conflicts=FALSE)


test_that("Test neaten", {

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
    step_naomit(all_predictors()) %>%
    prep()

  train <- bake(m5_recipe, train)
  test  <- bake(m5_recipe, test)

  TARGET      <- 'value'
  STATIC_CAT  <- c('store_id_idx')
  STATIC_NUM  <- 'item_id_idx'
  DYNAMIC_CAT <- c('event_name_1', 'event_type_1')
  CATEGORICAL <- c(DYNAMIC_CAT, STATIC_CAT)
  NUMERIC     <- c('sell_price', 'sell_price', 'item_id_idx')
  KEY         <- c('item_id', 'store_id')
  INDEX       <- 'date'
  LOOKBACK    <- 28
  HORIZON     <- 14
  STRIDE      <- LOOKBACK
  BATCH_SIZE  <- 32

  c(x_test, y_test) %<-%
    make_arrays(
      data        = test,
      key         = KEY,
      index       = INDEX,
      lookback    = LOOKBACK,
      horizon     = HORIZON,
      stride      = STRIDE,
      target      = TARGET,
      static      = c(STATIC_CAT, STATIC_NUM),
      categorical = CATEGORICAL,
      numeric     = NUMERIC,
      shuffle     = TRUE
    )

  tft <-
    model_tft(
      lookback                = LOOKBACK,
      horizon                 = HORIZON,
      past_numeric_size       = length(NUMERIC) + 1,
      past_categorical_size   = length(DYNAMIC_CAT),
      future_numeric_size     = length(NUMERIC),
      future_categorical_size = length(DYNAMIC_CAT),
      vocab_static_size       = dict_size(train, STATIC_CAT),
      vocab_dynamic_size      = dict_size(train, DYNAMIC_CAT),
      hidden_dim              = 10,
      state_size              = 5,
      num_heads               = 10,
      dropout_rate            = NULL,
      output_size             = 1
    )

  tft %>%
    compile(optimizer='adam', loss='mae')


  forecast <- tft(x_test)
  forecast_2 <- abind::abind(as.array(forecast),
                             as.array(forecast))

  out <-
    neaten(
      forecast = forecast,
      data     = test,
      key      = KEY,
      index    = INDEX,
      lookback = LOOKBACK,
      horizon  = HORIZON,
      target   = TARGET,
      stride   = STRIDE
    )

  expect_s3_class(out, "data.table")
  expect_true(".pred" %in% colnames(out))

  out_2 <-
    neaten(
      forecast = forecast_2,
      data     = test,
      key      = KEY,
      index    = INDEX,
      lookback = LOOKBACK,
      horizon  = HORIZON,
      target   = TARGET,
      stride   = STRIDE,
      suffix   = c(0.5, 0.9)
    )

  expect_s3_class(out_2, "data.table")
  expect_true(".pred_0.5" %in% colnames(out_2))
  expect_true(".pred_0.9" %in% colnames(out_2))


})
