#' Gated Linear Unit
#'
#' In such form introduced in [Language modeling with gated convolutional networks](https://arxiv.org/abs/1612.08083)
#' by Dauphin et al., when it was used in sequence processing tasks and compared with
#' gating mechanism used in LSTM layers.
#'
#' In the context of time series processing it was used in [Temporal Fusion Transformer](https://arxiv.org/pdf/1912.09363.pdf).
#'
#' @inheritParams keras::layer_dense
#' @param return_gate Logical - return gate values. Default: FALSE
#'
#' @references
#' Dauphin, Yann N., et al. (2017). [Language modeling with gated convolutional networks.](https://arxiv.org/abs/1612.08083)
#' International conference on machine learning. PMLR
#'
#' @examples
#' library(keras)
#'
#' # ================================================================
#' #             SEQUENTIAL MODEL, NO GATE VALUES RETURNED
#' # ================================================================
#'
#' model <-
#'   keras_model_sequential() %>%
#'   layer_glu(10, input_shape = 30)
#'
#' model %>%
#'    compile(optimizer = "adam", loss = "mse")
#'
#' model
#'
#' output <- model(matrix(1, 32, 30))
#' dim(output)
#' output[1,]
#'
#' # ================================================================
#' #                     WITH GATE VALUES RETURNED
#' # ================================================================
#'
#' inp  <- layer_input(30)
#' out  <- layer_glu(units = 10, return_gate = TRUE)(inp)
#'
#' model <- keras_model(inp, out)
#'
#' model
#'
#' model %>%
#'    compile(optimizer = "adam", loss = "mse")
#'
#' c(values, gate) %<-% model(matrix(1, 32, 30))
#' dim(values)
#' dim(gate)
#'
#' values[1,]
#' gate[1,]
#'
#'
#' @export
layer_glu <- keras::new_layer_class(

  classname = "GLU",

  initialize = function(units,
                        activation = NULL,
                        return_gate = FALSE,
                        ...){

    super()$`__init__`(...)

    self$units         <- as.integer(units)
    self$activation    <- activation
    self$return_gate   <- return_gate

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
    gated_output      <- self$gate_layer(inputs)

    output <-
      layer_multiply(list(
        activation_output,
        gated_output
      ))

    if (self$return_gate)
      return(list(output, gated_output))
    else
      return(output)

  }


)

