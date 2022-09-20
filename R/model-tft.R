#' Temporal Fusion Transformer model
#'
#' @param lookback Number of timesteps from the past
#' @param horizon Forecast length (number of timesteps)
#' @param past_numeric_size Number of numeric features from the past
#' @param past_categorical_size Number of categorical features from the past
#' @param future_numeric_size Number of numeric features from the future
#' @param output_size Number of the models output. For simple point estimate set `1`.
#'
#' @include layer-interpretable-mh-attention.R layer-grn.R
#'
#' @seealso
#' TFT components:
#'
#' [layer_glu()], [layer_grn()], [layer_multi_embedding()], [layer_multi_dense()],
#' [layer_scaled_dot_attention()], [layer_interpretable_mh_attention()], [layer_temporal_fusion_decoder()]
#'
#' @references
#' 1. [Paper](https://arxiv.org/abs/1912.09363)
#' 2. [Original TFT implementation in TensorFlow](https://github.com/google-research/google-research/blob/master/tft/libs/tft_model.py)
#' 3. [A very clear implementation in PyTorch](https://github.com/PlaytikaResearch/tft-torch/blob/main/tft_torch/tft.py)
#'
#' @examples
#' library(keras)
#' library(aion)
#'
#' tft <- model_tft(
#'    lookback                = 28,
#'    horizon                 = 14,
#'    past_numeric_size       = 5,
#'    past_categorical_size   = 2,
#'    future_numeric_size     = 4,
#'    future_categorical_size = 2,
#'    vocab_static_size       = c(5, 5),
#'    vocab_dynamic_size      = c(4, 4),
#'    hidden_dim              = 12,
#'    state_size              = 7,
#'    num_heads                 = 10,
#'    dropout_rate            = 0.1,
#'    output_size             = 3
#'    #quantiles               = 0.5
#' )
#'
#' x_static_cat <- array(sample(5, 32 * 2, replace=TRUE), c(32, 2)) - 1
#' x_static_num <- array(runif(32 * 1), c(32, 1))
#'
#' x_past_num <- array(runif(32 * 28 * 2), c(32, 28, 2))
#' x_past_cat <- array(sample(4, 32 * 28 * 2, replace=TRUE), c(32, 28, 5))
#'
#' x_fut_num <- array(runif(32 * 14 * 5), c(32, 28, 1))
#' x_fut_cat <- array(sample(4, 32 * 14 * 2, replace=TRUE), c(32, 28, 5))
#'
#' tft(x_past_num, x_past_cat, x_fut_num, x_fut_cat, x_static_num, x_static_cat)
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
                        hidden_dim = 10,
                        state_size = 5,
                        num_heads = 10,
                        dropout_rate = NULL,
                        output_size = 1, ...){

    super()$`__init__`(...)

    self$lookback     <- lookback
    self$horizon      <- horizon
    self$dropout_rate <- dropout_rate

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

    self$static_projection <- layer_multi_dense(
      units = state_size,
      new_dim = TRUE
    )

    self$past_projection <- layer_multi_dense(
      units = state_size,
      new_dim = TRUE
    )

    self$future_projection <- layer_multi_dense(
      units = state_size,
      new_dim = TRUE
    )

    # Variable selection layers
    self$vsn_static <- layer_vsn(
      hidden_units   = hidden_dim,
      state_size     = state_size,
      dropout_rate   = dropout_rate,
      use_context    = FALSE,
      return_weights = TRUE
    )

    self$vsn_past <- layer_vsn(
      hidden_units   = hidden_dim,
      state_size     = state_size,
      dropout_rate   = dropout_rate,
      use_context    = TRUE,
      return_weights = TRUE
    )

    self$vsn_future <- layer_vsn(
      hidden_units   = hidden_dim,
      state_size     = state_size,
      dropout_rate   = dropout_rate,
      use_context    = TRUE,
      return_weights = TRUE
    )

    # Static context layers
    self$grn_enrichment <- layer_grn(
      hidden_units = state_size,
      output_size  = state_size,
      dropout_rate = dropout_rate
    )

    self$grn_selection <- layer_grn(
      hidden_units = state_size,
      output_size  = state_size,
      dropout_rate = dropout_rate
    )

    self$grn_seq_cell <- layer_grn(
      hidden_units = state_size,
      output_size  = state_size,
      dropout_rate = dropout_rate
    )

    self$grn_hidden <- layer_grn(
      hidden_units = state_size,
      output_size  = state_size,
      dropout_rate = dropout_rate
    )

    # LSTM layers
    self$lstm_past <-
      layer_lstm(units = state_size,
                 return_sequences = TRUE,
                 return_state = TRUE)

    self$lstm_future <-
        layer_lstm(units = state_size,
                   return_sequences = TRUE)

    # After LSTMs
    self$post_lstm_dropout    <- layer_dropout(rate = dropout_rate)
    self$post_lstm_glu        <- layer_glu(units = state_size)
    self$post_lstm_layer_norm <- layer_layer_normalization()

    # Temporal fusion decoder
    self$tfd <- layer_temporal_fusion_decoder(
      hidden_units = hidden_dim,
      state_size   = state_size,
      dropout_rate = dropout_rate,
      use_context  = TRUE,
      num_heads    = num_heads
    )

    # Last gate
    self$last_glu        <- layer_glu(units = state_size)
    self$last_layer_norm <- layer_layer_normalization()

    # Output
    self$output_dense <- layer_dense(units = output_size)
    future_idx        <- (self$lookback + 1):(self$lookback + self$horizon)
    self$output_cut   <- layer_lambda(f = function(x) x[,future_idx,])

  },

  call = function(inputs){

    c(x_past_num, x_past_cat,
      x_fut_num, x_fut_cat,
      x_static_num, x_static_cat) %<-% inputs
    browser()
    # ==========================================================================
    #                       EMBEDDINGS & PROJECTIONS
    # ==========================================================================

    # All the inputs are projected to the common-size space
    static_emb <- self$static_embedding(x_static_cat)
    past_emb   <- self$past_embedding(x_past_cat)
    fut_emb    <- self$future_embedding(x_fut_cat)

    static_proj <- self$static_projection(x_static_num)
    past_proj   <- self$past_projection(x_past_num)
    fut_proj    <- self$future_projection(x_fut_num)

    # ==========================================================================
    #                       STATIC VARIABLE SELECTION
    # ==========================================================================

    # selected_static: [num_samples x state_size]
    # static_weights: [num_samples x num_static_inputs x 1]

    static_features <- layer_concatenate(axis = 1)(list(static_emb, static_proj))

    c(static_selected, static_selection_weights) %<-%
      self$vsn_static(static_features)

    # ============================================================================
    #                     STATIC CONTEXT VECTORS
    # ============================================================================

    # We create four separate static context vectors which are
    # then send to different parts of the network

    c_enrichment <- self$grn_enrichment(static_selected)
    c_selection  <- self$grn_selection(static_selected)
    c_seq_cell   <- self$grn_seq_cell(static_selected)
    c_seq_hidden <- self$grn_hidden(static_selected)

    # ==========================================================================
    #                       PAST VARIABLE SELECTION
    # ==========================================================================

    past_features <- layer_concatenate(axis = 2)(list(past_emb, past_proj))

    c(selected_past, past_selection_weights) %<-%
        self$vsn_past(past_features, c_selection)

    # ==========================================================================
    #                     FUTURE VARIABLE SELECTION
    # ==========================================================================

    fut_features <- layer_concatenate(axis = 2)(list(fut_emb, fut_proj))

    c(selected_future, fut_selection_weights) %<-%
      self$vsn_future(fut_features, c_selection)

    # ==========================================================================
    #                         LSTM FOR PAST FEATURES
    # ==========================================================================

    initial_state <- list(c_seq_hidden, c_seq_cell)

    c(processed_past, h1_past, h2_past) %<-%
      self$lstm_past(selected_past, initial_state = initial_state)

    hidden_state_past <- list(h1_past, h2_past)

    # ==========================================================================
    #                       LSTM FOR FUTURE FEATURES
    # ==========================================================================

    initial_state <- list(c_seq_hidden, c_seq_cell)

    processed_future <-
      self$lstm_future(selected_future, initial_state = hidden_state_past)

    # ==========================================================================
    #                         GATE AFTER LSTM LAYERS
    # ==========================================================================

    # It let us skip the LSTM layers if needed
    lstms_input_features <-
      layer_concatenate(axis = 1)(list(selected_past, selected_future))

    lstms_output_features <-
      layer_concatenate(axis = 1)(list(processed_past, processed_future))

    if (!is.null(self$dropout_rate))
      lstms_input_features <- self$post_lstm_dropout(lstms_input_features)

    lstms_input_features <- self$post_lstm_glu(lstms_input_features)

    combined_lstm_output <- layer_add(
       list(lstms_input_features, lstms_output_features)
    )

    combined_lstm_output <- self$post_lstm_layer_norm(combined_lstm_output)

    # ==========================================================================
    #                   TEMPORAL FUSION TRANSFORMER BLOCK
    # ==========================================================================

    tfd_output <- self$tfd(combined_lstm_output, c_enrichment)

    # ==========================================================================
    #                             LAST GATE
    # ==========================================================================
    # This gate allows the model to minimize TFD block impact if needed
    tfd_output <- self$last_glu(tfd_output)

    gated_output <- layer_add(
      list(tfd_output, combined_lstm_output)
    )

    gated_output <- self$last_layer_norm(gated_output)

    # ==========================================================================
    #                               OUTPUT
    # ==========================================================================
    final_output <- self$output_dense(gated_output)
    final_output <- self$output_cut(final_output)

    final_output
  }


)
