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
#' # ==========================================================================
#' #                          CREATING GENERATOR
#' # ==========================================================================
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
#' print(names(batch[[1]]))
#' @export
ts_generator <- function(data, key, index,
                         lookback, horizon, stride=1,
                         target, numeric=NULL, categorical=NULL,
                         static=NULL, past=NULL, future=NULL,
                         shuffle=TRUE, sample_frac = 1.,
                         y_past_sep = FALSE, batch_size=1, ...){

  output <- prepare_idx_and_vars(
      data        = data,
      lookback    = lookback,
      horizon     = horizon,
      key         = key,
      index       = index,
      stride      = stride,
      target      = target,
      numeric     = numeric,
      categorical = categorical,
      past        = past,
      future      = future,
      static      = static,
      shuffle     = shuffle,
      sample_frac = sample_frac,
      y_past_sep  = y_past_sep
  )

  ts_starts          <- output[[1]]
  past_var           <- output[[2]]
  fut_var            <- output[[3]]
  static_numeric     <- output[[4]]
  static_categorical <- output[[5]]

  n_steps <- ceiling(nrow(ts_starts) / batch_size)

  past_var <- remove_nulls(past_var)
  fut_var  <- remove_nulls(fut_var)

  all_var_groups <-
    c(names(past_var), names(fut_var), c('x_static_cat', 'x_static_num'))

  x_var_groups <- setdiff(all_var_groups, 'y_fut')

  i <- 0

  generator <- function(){

    # Reset counter
    if (i == n_steps)
      i <<- 0

    start     <- i * batch_size + 1
    samples   <- ts_starts[start:(start + batch_size - 1), ]

    rows <- samples$row_idx

    static_arrays <- list()

    if (length(static_numeric) > 0)
      static_arrays$x_static_num <-
          as.matrix(data[rows, ..static_numeric])

    if (length(static_categorical) > 0)
      static_arrays$x_static_cat <-
      as.matrix(data[rows, ..static_categorical])

    dynamic_arrays <-
      get_arrays(
        data      = data,
        ts_starts = samples,
        lookback  = lookback,
        horizon   = horizon,
        past_var  = past_var,
        fut_var   = fut_var
      )

    all_vars_list <-
      c(dynamic_arrays, static_arrays)

    y_list <- all_vars_list$y_fut
    x_list <- all_vars_list[x_var_groups]

    list(x_list, y_list)
  }

  list(generator, n_steps)
}
