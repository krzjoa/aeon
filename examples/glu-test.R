library(keras)

# ============================================================================
#                                   GLU 1
# ============================================================================

GLU <- Layer(

  classname = "GLU",

  initialize = function(units,
                        droupout_rate=NULL,
                        return_gate = FALSE,
                        name = NULL,
                        ...){



    super()$`__init__`(name = name, ...)

    self$units         <- units
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

layer_glu_1 <- function(object, units,
                      input_shape = NULL,
                      return_gate = FALSE,
                      name = NULL){

  args <- list(
    units         = as.integer(units),
    input_shape   = keras:::normalize_shape(input_shape),
    return_gate   = return_gate,
    name          = name
  )

  create_layer(GLU, object, args)

}


model <-
  keras_model_sequential() %>%
  layer_glu_1(10, input_shape = 30)

model

inp  <- layer_input(30)
out  <- layer_glu_1(units = 10, return_gate = TRUE)(inp)

model <- keras_model(inp, out)

model


# ============================================================================
#                                   GLU 2
# ============================================================================


layer_glu_2 <- new_layer_class(

  classname = "GLU",

  initialize = function(units,
                        droupout_rate=NULL,
                        activation = NULL,
                        name = NULL,
                        return_gate = FALSE,
                        # input_shape = NULL,
                        # batch_input_shape = NULL,
                        # batch_size = NULL,
                        ...){

    super()$`__init__`(
        name              = name,
        # input_shape       = keras:::normalize_shape(input_shape),
        # batch_input_shape = keras:::normalize_shape(batch_input_shape),
        # batch_size        = keras:::as_nullable_integer(batch_size),
        ...
    )

    self$units         <- as.integer(units)
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

model <-
  keras_model_sequential() %>%
  layer_glu_2(10, input_shape = 30)

model


inp  <- layer_input(30)
out  <- layer_glu_2(units = 10, return_gate = TRUE)(inp)

model <- keras_model(inp, out)
