
GRN <- Layer(
  classname = "GRN",

  initialize = function(hidden_units,
                        output_size,
                        # dropout_rate = NULL,
                        use_additional_context = FALSE,
                        return_gate = FALSE,
                        name = NULL, ...){

    super()$`__init__`(name = name, ...)

    self$hidden_units       <- hidden_units
    self$use_additional_context <- use_additional_context
    self$return_gate        <- return_gate

    # Setup skip connection
    if (is.null(output_size)) {
      self$output_size <- hidden_units
      self$flag_skip   <- TRUE
    } else {
      self$output_size <- output_size
      self$flag_skip   <- FALSE
    }


  },

  build = function(input_shape){

    if (!self$flag_skip)
      self$pre_layer <- layer_dense(units = self$output_size,
                                    input_shape = input_shape)

    self$hidden_layer <- layer_dense(units = self$hidden_units,
                                     input_shape = input_shape)

    if (self$use_additional_context)
      self$context_layer <- layer_dense(
        units    = self$hidden_units,
        use_bias = FALSE
      )

    self$elu_layer    <- layer_activation(activation = 'elu')
    self$final_layer  <- layer_dense(units = self$hidden_units)
    self$gating_layer <- layer_glu(units = self$output_size,
                                   return_gate = TRUE)

  },

  call = function(inputs, ...){

    # Skip connection
    if (self$flag_skip) {
      skip <- inputs
    } else {
      skip <- self$pre_layer(inputs)
    }

    # Apply feedforward network
    hidden <- self$hidden_layer(inputs)

    if (self$use_additional_context) {
      ctx    <- self$context_layer(additional_context)
      hidden <- layer_add()(list(hidden, ctx))
    }

    hidden <- self$elu_layer(hidden)
    hidden <- self$final_layer(hidden)

    c(gated_output, gate) %<-% self$gating_layer(hidden)

    output <- layer_add()(list(skip, gated_output))
    output <- layer_layer_normalization()(output)

    if (self$return_gate)
      return(list(output, gate))
    else
      return(output)

  }

)


#' Gated Residual Network block
layer_grn <- function(object,
                      hidden_units,
                      output_size,
                      #dropout_rate = NULL,
                      use_additional_context = FALSE,
                      input_shape  = NULL,
                      return_gate = FALSE,
                      name = NULL){

  args <- list(
    hidden_units  = as.integer(hidden_units),
    output_size   = output_size,
    #droupout_rate = droupout_rate,
    use_additional_context = use_additional_context,
    input_shape   = keras:::normalize_shape(input_shape),
    return_gate   = return_gate,
    name          = name
  )

  create_layer(GRN, object, args)
}




