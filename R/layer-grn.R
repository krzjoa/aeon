#' Gated Residual Network block
#'
#' GRN is one of the elements the TFT model is composed of.
#' The expected benefit from applying such value is a better ability
#' of switching between linear and non-linear processing.
#'
#' Its output is computed as:
#' \deqn{GRN(a,c) = LayerNorm(a + GLU({\eta}_1))}
#' \deqn{{\eta}_1 = W_1\eta_2 + b_1}
#' \deqn{\eta_2 = ELU(W_2a + W_3c + b_2)}
#'
#' ![](img/grn.png)
#'
#' @param hidden_units Size of the hidden layer.
#' @param output_size Dimensionality of the output feature space.
#' @param use_context Use additional (static) context. If TRUE, an additional layer
#' is created to handle context input.
#' @inheritParams layer_glu
#'
#' @include layer-glu.R
#'
#' @inheritSection keras::layer_dense Input and Output Shapes
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
#'   layer_grn(10, input_shape = 30)
#'
#' model
#'
#' output <- model(matrix(1, 32, 30))
#' dim(output)
#' output[1,]
#'
#' #'================================================================
#' #            WITH GATE VALUES AND ADDITIONAL CONTEXT
#' # ================================================================
#'
#' inp  <- layer_input(c(28, 5))
#' ctx  <- layer_input(10)
#' out  <- layer_grn(
#'             hidden_units = 10,
#'             return_gate = TRUE,
#'             use_context = TRUE
#'          )(inp, context = ctx)
#'
#' model <- keras_model(list(inp, ctx), out)
#'
#' model
#'
#' arr_1 <- array(1, dim = c(1, 28, 5))
#' arr_2 <- array(1, dim = c(1, 10))
#'
#' c(values, gate) %<-% model(list(arr_1, arr_2))
#' dim(values)
#' dim(gate)
#'
#' values[1, all_dims()]
#' gate[1, all_dims()]
#'
#' @export
layer_grn <- keras::new_layer_class(

  # https://keras.io/examples/structured_data/classification_with_grn_and_vsn/

  classname = "GRN",

  initialize = function(hidden_units,
                        output_size = hidden_units,
                        dropout_rate = NULL,
                        use_context  = FALSE,
                        return_gate  = FALSE,
                        ...){

    # TODO: consider changing name the output_size argument

    super()$`__init__`(...)

    self$hidden_units <- hidden_units
    self$dropout_rate <- dropout_rate
    self$use_context  <- use_context
    self$return_gate  <- return_gate
    self$output_size  <- output_size

  },

  build = function(input_shape){


    # Setup skip connection
    if (rev(input_shape)[1] != self$output_size) {
      self$change_dim_for_skip <- TRUE
    } else {
      self$change_dim_for_skip <- FALSE
    }

    if (self$change_dim_for_skip)
      self$dim_layer <- layer_dense(units = self$output_size,
                                    input_shape = input_shape)

    self$layer_1 <- layer_dense(units = self$hidden_units,
                                input_shape = input_shape)

    if (self$use_context)
      self$context_layer <- layer_dense(
        units    = self$hidden_units,
        use_bias = FALSE
      )

    self$elu_layer    <- layer_activation(activation = 'elu')
    self$layer_2      <- layer_dense(units = self$output_size)
    self$dropout      <- layer_dropout(rate = self$dropout_rate)
    self$gating_layer <- layer_glu(units = self$output_size,
                                   return_gate = TRUE)
    self$norm         <- layer_layer_normalization()

  },

  call = function(inputs, context = NULL){

    # Skip connection
    if (self$change_dim_for_skip) {
      skip <- self$dim_layer(inputs)
    } else {
      skip <- inputs
    }

    # Apply feedforward network
    hidden <- self$layer_1(inputs)

    if (self$use_context) {
      ctx    <- self$context_layer(context)
      hidden <- layer_add()(list(hidden, ctx))
    }

    hidden <- self$elu_layer(hidden)
    hidden <- self$layer_2(hidden)

    c(out, gate) %<-% self$gating_layer(hidden)

    output <- layer_add()(list(skip, out))
    output <- self$norm(output)

    if (self$return_gate)
      return(list(output, gate))
    else
      return(output)

  }

)



