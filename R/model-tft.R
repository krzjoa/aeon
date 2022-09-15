#' Temporal Fusion Transformer model
#'
#' @param lookback Number of timesteps from the past
#' @param horizon Forecast length (number of timesteps)
#' @param past_numeric_size Number of numeric features from the past
#' @param past_categorical_size Number of categorical features from the past
#' @param future_numeric_size Number of numeric features from the future
#'
#' @include layer-interpretable-mh-attention.R layer-grn.R
#'
#' @examples
#' tft <- model_tft(
#'    lookback                = 28,
#'    horizon                 = 14,
#'    past_numeric_size       = 5,
#'    past_categorical_size   = 2,
#'    future_numeric_size     = 4,
#'    future_categorical_size = 2,
#'    vocab_static_size       = c(3, 4),
#'    vocab_dynamic_size      = 6,
#'    optimizer               = 'adam',
#'    hidden_dim              = 12,
#'    state_size              = 7,
#'    n_heads                 = 10,
#'    dropout_rate            = 0.1,
#'    output_size             = 3
#'    #quantiles               = 0.5
#' )
#'
#'
#' @export
model_tft <- keras::new_model_class(

  classname = "TemporalFusionTransformer",

  initialize = function(lookback,
                        horizon,
                        past_numeric_size = 2, # value + cena domyślnie
                        past_categorical_size = 7,
                        future_numeric_size = 1,
                        future_categorical_size = 7,
                        vocab_static_size,
                        vocab_dynamic_size,
                        optimizer,
                        hidden_dim = 10,
                        state_size = 5,
                        n_heads = 10,
                        dropout_rate = NULL,
                        output_size = 1, ...){

    super()$`__init__`(...)

    self$lookback <- lookback
    self$horizon  <- horizon

    # Inputs
    self$input_numeric_past <-
      layer_input(shape = c(lookback, past_numeric_size))
    self$input_categorical_past <-
      layer_input(shape = c(lookback, past_categorical_size))
    self$input_numeric_future <-
      layer_input(shape = c(horizon, future_numeric_size))
    self$input_categorical_future <-
      layer_input(shape = c(horizon, future_categorical_size))
    self$input_static <-
      layer_input(shape = length(vocab_static_size))

    # Embeddings and projections
    self$static_embedding <- layer_multi_embedding(
      input_dims= vocab_static_size,
      output_dims = state_size,
      new_dim = TRUE
    )

    self$past_embedding <- layer_multi_embedding(
      input_dims = vocab_dynamic_size,
      output_dims = state_size,
      new_dim = TRUE
    )

    self$future_embedding <- layer_multi_embedding(
      input_dims = vocab_dynamic_size,
      output_dims = state_size,
      new_dim = TRUE
    )

    self$static_numeric <- layer_multi_dense(
      units = state_size,
      new_dim = TRUE
    )

    self$past_numeric <- layer_multi_dense(
      units = state_size,
      new_dim = TRUE
    )

    self$future_numeric <- layer_multi_dense(
      units = state_size,
      new_dim = TRUE
    )

  },

  call = function(X_past_num, X_past_cat,
                  X_fut_num, X_fut_cat,
                  X_static_cat, X_static_num){

    # ==========================================================================
    #                       EMBEDDINGS & PROJECTIONS
    # ==========================================================================

    #static_embedding <-

  }

)
