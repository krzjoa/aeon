#' Scaled dot product attention layer
#'
#' Introduced in [Attention Is All You Need](https://arxiv.org/pdf/1706.03762v5.pdf).
#' Defined as:
#'
#' \deqn{Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V}
#'
#' Originally, `dropout` hasn't been specified there. It was added inside the layer
#' in the [Temporal Fusion Transformer](https://github.com/google-research/google-research/blob/4808a726f4b126ea38d49cdd152a6bb5d42efdf0/tft/libs/tft_model.py#L240) implementation by Google.
#' It's a component of the Multi-Head Attention Layers (as well as its interpretable version, available in the `aion` package).
#'
#' @param dropout_rate Dropout rate
#' @inheritSection keras::layer_multi_head_attention Call arguments
#'
#' @references
#' [Attention Is All You Need](https://arxiv.org/pdf/1706.03762v5.pdf).
#'
#' @examples
#'
#' lookback   <- 28
#' horizon    <- 14
#' all_steps  <- lookback + horizon
#' state_size <- 5
#'
#' queries <- layer_input(c(horizon, state_size))
#' keys    <- layer_input(c(all_steps, state_size))
#' values  <- layer_input(c(all_steps, state_size))
#'
#' sdp_attention <- layer_scaled_dot_attention()(queries, keys, values)
#' @export
layer_scaled_dot_attention <- keras::new_layer_class(

  classname = "ScaledDotAttention",

  initialize = function(dropout_rate=0.0, ...){
    super()$`__init__`(...)
    self$dropout_rate <- dropout_rate
  },

  build = function(input_shape){
    self$dropout <- layer_dropout(rate = self$dropout_rate)
    self$softmax <- layer_activation_softmax()
  },

  call = function(query, key, value, mask = NULL, return_attention_scores=FALSE){

    # mask: Masking if required with shape=(?, T, T)
    last_dim  <- tail(dim(key), 1)
    temper    <- k_sqrt(k_cast(last_dim, dtype='float32'))
    attention <- layer_lambda(
      f = function(x) k_batch_dot(x[[1]], x[[2]], axes = c(3, 3)) / temper
    )(list(query, key))

    if (!is.null(mask)) {
      mask <- layer_lambda(
        f = function(x) (-1e+9) * (1 - k_cast(x, 'float32'))
      )(mask)  # setting to infinity
      attention <- layer_add()(list(attention, mask))
    }

    attention <- self$softmax(attention)
    attention <- self$dropout(attention)

    output <- layer_lambda(
      f = function(x) k_batch_dot(x[[1]], x[[2]], axes = NULL)
    )(list(attention, value))

    if (return_attention_scores)
      return(list(output, attention))
    else
      return(output)
  }

)
