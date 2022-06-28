#' Gated Residual Network block
#'
#' @export
layer_grn <- keras::new_layer_class(

  classname = "GRN",

  initialize = function(hidden_units,
                        output_size,
                        dropout_rate = NULL,
                        use_additional_context = FALSE,
                        return_gate = FALSE,
                        name = NULL, ...){

    super()$`__init__`(name = name, ...)

    self$hidden_units           <- hidden_units
    self$dropout_rate           <- dropout_rate
    self$use_additional_context <- use_additional_context
    self$return_gate            <- return_gate

    # Setup skip connection
    if (is.null(output_size)) {
      self$output_size <- hidden_units
      self$change_dim_for_skip   <- FALSE
    } else {
      self$output_size <- output_size
      self$change_dim_for_skip   <- TRUE
    }


  },

  build = function(input_shape){

    # If output_shape is determined, it can differ from the initial one
    # we have to apply additional linear transformation to change dimensionality
    if (self$change_dim_for_skip)
      self$dim_layer <- layer_dense(units = self$output_size,
                                    input_shape = input_shape)

    self$layer_1 <- layer_dense(units = self$hidden_units,
                                input_shape = input_shape)

    if (self$use_additional_context)
      self$context_layer <- layer_dense(
        units    = self$hidden_units,
        use_bias = FALSE
      )

    self$elu_layer    <- layer_activation(activation = 'elu')
    self$layer_2      <- layer_dense(units = self$hidden_units)
    self$dropout      <- layer_dropout(rate = self$dropout_rate)
    self$gating_layer <- layer_glu(units = self$output_size,
                                   return_gate = TRUE)

  },

  call = function(inputs, context = NULL, ...){

    # Skip connection
    if (self$change_dim_for_skip) {
      skip <- self$dim_layer(inputs)
    } else {
      skip <- inputs
    }

    # Apply feedforward network
    hidden <- self$layer_1(inputs)

    if (self$use_additional_context) {
      ctx    <- self$context_layer(context)
      hidden <- layer_add()(list(hidden, ctx))
    }

    hidden <- self$elu_layer(hidden)
    hidden <- self$layer_2(hidden)

    c(gated_output, gate) %<-% self$gating_layer(hidden)

    output <- layer_add()(list(skip, gated_output))
    output <- layer_layer_normalization()(output)

    if (self$return_gate)
      return(list(output, gate))
    else
      return(output)

  }

)




