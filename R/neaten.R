.neaten_single <- function(forecast, ts_starts, index, key_index){
  forecast <- as.data.frame(forecast)
  forecast <- cbind(ts_starts, forecast)
  setDT(forecast)
  forecast <- melt(forecast, id.vars=key_index)
  forecast[, variable := as.numeric(variable) - 1]
  forecast[, (index) := get(index) + variable]
  setnames(forecast, "value", ".pred")
  forecast[, !"variable"]
}


#' Construct a data.frame out of the output forecast tensor
#'
#' @param x An `array` or `tensorflow.tensor` containing forecasts
#' @param data A data.frame containing test data
#' @param suffix Needed only when more than one forecast is generated
#' @inheritParams make_arrays
#'
#' @include utils.R
#'
#' @examples
#' \donttest{
#' #' Dataset
#' library(m5)
#'
#' #' Neural Networks
#' library(aion)
#' library(keras)
#'
#' #' Data wrangling
#' library(dplyr, warn.conflicts=FALSE)
#' library(data.table, warn.conflicts=FALSE)
#' library(recipes, warn.conflicts=FALSE)
#'
#' train <- tiny_m5[date < '2016-01-01']
#' test  <- tiny_m5[date >= '2016-01-01']
#'
#' m5_recipe <-
#'   recipe(value ~ ., data=train) %>%
#'   step_mutate(item_id_idx=item_id, store_id_idx=store_id) %>%
#'   step_integer(item_id_idx, store_id_idx,
#'                wday, month,
#'                event_name_1, event_type_1,
#'                event_name_2, event_type_2,
#'                zero_based=TRUE) %>%
#'   step_naomit(all_predictors()) %>%
#'   prep()
#'
#' train <- bake(m5_recipe, train)
#' test  <- bake(m5_recipe, test)
#'
#' TARGET      <- 'value'
#' STATIC_CAT  <- c('store_id_idx')
#' STATIC_NUM  <- 'item_id_idx'
#' DYNAMIC_CAT <- c('event_name_1', 'event_type_1')
#' CATEGORICAL <- c(DYNAMIC_CAT, STATIC_CAT)
#' NUMERIC     <- c('sell_price', 'sell_price', 'item_id_idx')
#' KEY         <- c('item_id', 'store_id')
#' INDEX       <- 'date'
#' LOOKBACK    <- 28
#' HORIZON     <- 14
#' STRIDE      <- LOOKBACK
#' BATCH_SIZE  <- 32
#'
#' c(x_test, y_test) %<-%
#'   make_arrays(
#'     data        = test,
#'     key         = KEY,
#'     index       = INDEX,
#'     lookback    = LOOKBACK,
#'     horizon     = HORIZON,
#'     stride      = STRIDE,
#'     target      = TARGET,
#'     static      = c(STATIC_CAT, STATIC_NUM),
#'     categorical = CATEGORICAL,
#'     numeric     = NUMERIC,
#'     shuffle     = TRUE
#'   )
#'
#' tft <-
#'   model_tft(
#'     lookback                = LOOKBACK,
#'     horizon                 = HORIZON,
#'     past_numeric_size       = length(NUMERIC) + 1,
#'     past_categorical_size   = length(DYNAMIC_CAT),
#'     future_numeric_size     = length(NUMERIC),
#'     future_categorical_size = length(DYNAMIC_CAT),
#'     vocab_static_size       = dict_size(train, STATIC_CAT),
#'     vocab_dynamic_size      = dict_size(train, DYNAMIC_CAT),
#'     hidden_dim              = 10,
#'     state_size              = 5,
#'     num_heads               = 10,
#'     dropout_rate            = NULL,
#'     output_size             = 1
#'   )
#'
#' tft %>%
#'   compile(optimizer='adam', loss='mae')
#'
#' # One target variable
#' forecast <- tft(x_test)
#'
#' out <-
#'   neaten(
#'     forecast = forecast,
#'     data     = test,
#'     key      = KEY,
#'     index    = INDEX,
#'     lookback = LOOKBACK,
#'     horizon  = HORIZON,
#'     target   = TARGET,
#'     stride   = STRIDE
#'   )
#'
#' # Multiple target variables
#' forecast_2 <- abind::abind(as.array(forecast),
#'                            as.array(forecast))
#'
#' out_2 <-
#'   neaten(
#'     forecast = forecast_2,
#'     data     = test,
#'     key      = KEY,
#'     index    = INDEX,
#'     lookback = LOOKBACK,
#'     horizon  = HORIZON,
#'     target   = TARGET,
#'     stride   = STRIDE,
#'     suffix   = c(0.5, 0.9)
#'   )
#' }
#' @export
neaten <- function(forecast, data, key, index,
                   lookback, horizon, target, stride,
                   suffix = NULL){

  forecast <- as.array(forecast)

  c(ts_starts, total_window_length, key_index) %<-%
    get_ts_starts(
      data        = data,
      key         = key,
      index       = index,
      lookback    = lookback,
      horizon     = horizon,
      stride      = stride,
      sample_frac = 1
    )

  colnames(ts_starts)[1:length(key_index)] <- key_index

  # Squeeze 3-dimensional array when ;ast_dim equals 1
  if ((ndim(forecast) == 3 & safe_logical(last_dim(forecast) == 1))){
    forecast <- forecast[,,1]
  }

  if (ndim(forecast) == 2) {
    forecast <- .neaten_single(forecast, ts_starts, index, key_index)
  } else {

    outputs <- .neaten_single(forecast[,,1], ts_starts, index, key_index)
    setDT(outputs)
    setnames(outputs, ".pred", paste0(".pred", "_", suffix[1]))

    for (i in 2:last_dim(forecast)) {
      single_out <- .neaten_single(forecast[,,i], ts_starts, index, key_index)
      name <- paste0(".pred", "_", suffix[i])
      outputs[, (name) := single_out$.pred]
    }
    forecast <- outputs
  }
  setorderv(forecast, key_index)
  forecast
}
