.neaten_single <- function(forecast, ts_starts, index, key_index, suffix=""){
  if (suffix != "")
    suffix <- paste0("_", suffix)
  forecast <- as.data.frame(forecast)
  forecast <- cbind(ts_starts, forecast)
  setDT(forecast)
  forecast <- melt(forecast, id.vars=key_index)
  forecast[, variable := as.numeric(variable) - 1]
  forecast[, (index) := get(index) + variable]
  setnames(forecast, "value", paste0(".pred", suffix))
  forecast[, !"variable"]
}


#' Construct a data.frame out of the output forecast tensor
#'
#' @param x An `array` or `tensorflow.tensor` containing forecasts
#' @param data A data.frame containing test data
#' @inheritParams make_arrays
#'
#' @export
neaten <- function(forecast, data, key, index,
                   lookback, horizon, target, stride){

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

  colnames(ts_starts)[1:3] <- key_index

  if (length(dim(forecast)) == 3){


  } else {
    forecast
  }

  forecast <- .neaten_single(forecast[,,1], ts_starts, index, key_index)
  setorderv(forecast, key_index)
  forecast
}
