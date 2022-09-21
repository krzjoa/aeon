#' Gated Linear Unit
#'
#' In such form introduced in [Language modeling with gated convolutional networks](https://arxiv.org/abs/1612.08083)
#' by Dauphin et al., when it was used in sequence processing tasks and compared with
#' gating mechanism used in LSTM layers. In the context of time series processing explicitly proposed in [Temporal Fusion Transformer](https://arxiv.org/pdf/1912.09363.pdf).
#'
#' Computed according to the equation:
#' \deqn{GLU(\gamma) = \sigma(W\gamma + b) \odot (V\gamma + c)}
#'
#' @inheritParams keras::layer_dense
#' @param return_gate Logical - return gate values. Default: FALSE
#'
#' @inheritSection keras::layer_dense Input and Output Shapes
#'
#' @returns
#' Tensor of shape (batch_size, ..., units). Optionally, it can also return a weights tensor
#' with identical shape.
#'
#' @references
#' 1. Y. N. Dauphin., et al. [Language modeling with gated convolutional networks.](https://arxiv.org/abs/1612.08083).
#' International conference on machine learning. PMLR (2017)
#' 2. B. Lim, S.O. Arik, N. Loeff, T. Pfiste, [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363)(2020)
#' 3. [Implementation PyTorch by Jan Beitner](https://github.com/jdb78/pytorch-forecasting/blob/268121aa9aa732558beefb6d9f95feff955ad08b/pytorch_forecasting/models/temporal_fusion_transformer/sub_modules.py#L71)
#' 4. [Implementation PyTorch by Playtika Research](https://github.com/PlaytikaResearch/tft-torch/blob/9eee6db42b8ec6b6a586e8852e3af6e2d6b18035/tft_torch/tft.py#L11)
#'
#' @examples
#' library(keras)
#' # ================================================================
#' #             SEQUENTIAL MODEL, NO GATE VALUES RETURNED
#' # ================================================================
#' model <-
#'   keras_model_sequential() %>%
#'   layer_glu(10, input_shape = 30)
#' model
#' output <- model(matrix(1, 32, 30))
#' dim(output)
#' output[1,]
#' # ================================================================
#' #                     WITH GATE VALUES RETURNED
#' # ================================================================
#' inp   <- layer_input(30)
#' out   <- layer_glu(units = 10, return_gate = TRUE)(inp)
#' model <- keras_model(inp, out)
#' model
#' c(values, gate) %<-% model(matrix(1, 32, 30))
#' dim(values)
#' dim(gate)
#' values[1,]
#' gate[1,]
#' @export
layer_glu <- keras::new_layer_class(

  # Notation according to the https://math.stackexchange.com/a/601545

  classname = "GLU",

  initialize = function(units,
                        activation = NULL,
                        return_gate = FALSE,
                        ...){

    super()$`__init__`(...)

    self$units       <- as.integer(units)
    self$activation  <- activation
    self$return_gate <- return_gate

  },

  build = function(input_shape){

    self$activation_layer <- layer_dense(units = self$units,
                                         input_shape = input_shape)

    self$gate_layer       <- layer_dense(units = self$units,
                                         input_shape = input_shape,
                                         activation = 'sigmoid')

  },


  call = function(inputs){

    activation_output <- self$activation_layer(inputs)
    gate_weights      <- self$gate_layer(inputs)

    output <-
      layer_multiply(list(
        gate_weights,
        activation_output
      ))

    if (self$return_gate)
      return(list(output, gate_weights))
    else
      return(output)

  }


)

