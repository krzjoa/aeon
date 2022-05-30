#' Temporal Fusion Trannsformer
#'
#' Paper: https://arxiv.org/abs/1912.09363
#' Original implementation: https://github.com/google-research/google-research/blob/master/tft/libs/tft_model.py
#' Alternative: https://github.com/LiamMaclean216/Temporal-Fusion-Transformer


#' @import keras
GLU <- Layer(

  classname = "GLU",

  initialize = function(units,
                        droupout_rate=NULL,
                        activation = NULL,
                        name = NULL,
                        return_gate = FALSE,
                        ...){

      super()$`__init__`(name = name, ...)

      self$units         <- units
      self$droupout_rate <- droupout_rate
      self$activation    <- activation
      self$return_gate   <- return_gate

    },

    build = function(input_shape){

      self$activation_layer <- layer_dense(units = self$units,
                                           input_shape = input_shape)

      self$gated_layer      <- layer_dense(units = self$units,
                                           input_shape = input_shape,
                                           activation = 'sigmoid')

    },


    call = function(inputs, ...){

      # if (!is.null(self$dropout_layer))
      #   inputs <- self$dropout_layer(inputs)

      activation_output <- self$activation_layer(inputs)
      gated_output      <- self$gated_layer(inputs)

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

#' Gated Linear Unit
#'
#' A standard linear ('dense') layer extended with the gating mechanism
#'
#' @examples
#'
#' library(keras)
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
#' model(matrix(1, 32, 30))
#'
#' @export
layer_glu <- function(object, units, droupout_rate=NULL,
                      input_shape = NULL,
                      activation = NULL,
                      return_gate = FALSE,
                      name = NULL){

  args <- list(
    units         = as.integer(units),
    droupout_rate = droupout_rate,
    input_shape   = keras:::normalize_shape(input_shape),
    activation    = activation,
    return_gate   = return_gate,
    name          = name
  )

  create_layer(GLU, object, args)

}

