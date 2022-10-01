#' @keywords internal
resolve_variables <- function(period_set, type_set){
  if (is.null(period_set))
    return(type_set)
  base::intersect(period_set, type_set)
}

#' Var function, which accepts one-element vectors
#' @keywords internal
safe_var <- function(x){
  if (length(x) == 1)
    return(0)
  else
    return(var(x))
}

#' @keywords internal
try_import <- function(name, site){
  if (!reticulate::py_module_available(name))
    stop(glue::glue("Module {name} is not available and must be installed first. ",
                    "See: {site}"))
  reticulate::import(name)
}

#' @keywords internal
remove_nulls <- function(l){
  Filter(\(x)!is.null(x), l)
}

#' Get cardinalities of the categorical variables
#'
#' Needed as an input to the `[layer_multi_embedding()]`
#'
#' @param data A `data.frame` object
#' @param categorical List of the categorical variables
#'
#' @examples
#' dict_size(m5::tiny_m5, c('event_name_1', 'event_type_1'))
#' @export
dict_size <- function(data, categorical){
  # We can't use names, because it causes an error in 'keras'
  sapply(as.data.frame(data)[categorical], dplyr::n_distinct)
}

#' @keywords internal
remove_names <- function(x){
  names(x) <- NULL
  x
}


#' Generate an array of random values with the given dimensions
#'
#' Useful when testing developed model. We can check if the model accepts
#' input arrays of the expected size and if it returns output with desired shape.
#'
#' @param ... Array dimensions.
#' @returns An array of random values between 0 and 1 with the defined dimensions.
#'
#' @examples
#' rand_array(2,3,4)
#' dim(rand_array(32,14,8))
#' @export
rand_array <- function(...){
  dims <- c(...)
  array(runif(prod(dims)), dims)
}

#' Check if time series in the panel dataset contain time gaps.
#'
#' @inheritParams make_arrays
#' @returns Logical value - TRUE if input data contains time gaps.
#' @examples
#' glob_econ <- as.data.table(tsibbledata::global_economy)
#' find_gaps(glob_econ, 'Country', 'Year')
#' glob_econ <- glob_econ[Year != 1970]
#' find_gaps(glob_econ, 'Country', 'Year')
#' @export
find_gaps <- function(data, key, index){
  setDT(data)
  setorderv(data, key)
  potential_gaps <-
    data[, .(diffs = diff(get(index))), by = eval(key)]
  any(potential_gaps$diffs > 1)
}


#' @keywords internal
check_gaps <- function(data, key, index){
  if (find_gaps(data, key, index))
    stop("The data constains time gaps! Add missing time points or introduce and articifial time index.")
}


#' @keywords internal
get_ts_starts <- function(data, key, index, lookback,
                          horizon, stride, sample_frac){

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

  list(ts_starts, total_window_length, key_index)
}


