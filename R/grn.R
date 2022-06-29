#' Gated Residual Network block
#'
#' @inheritParams layer_glu
#' @param use_context Use additional (static) context. If TRUE, an additional layer
#' is created to handle context input.
#'
#' @include glu.R
#' @seealso [https://keras.io/examples/structured_data/classification_with_grn_and_vsn/](Python)
#'
#' @export
layer_grn <- keras::new_layer_class(

  classname = "GRN",

  initialize = function(units,
                        output_size = units,
                        dropout_rate = NULL,
                        use_context = FALSE,
                        return_gate = FALSE,
                        ...){

    super()$`__init__`(...)

    self$units        <- units
    self$dropout_rate <- dropout_rate
    self$return_gate  <- return_gate

    # Setup skip connection
    if (output_size != units) {
      self$output_size         <- units
      self$change_dim_for_skip <- FALSE
    } else {
      self$output_size         <- output_size
      self$change_dim_for_skip <- TRUE
    }


  },

  build = function(input_shape){

    if (self$change_dim_for_skip)
      self$dim_layer <- layer_dense(units = self$output_size,
                                    input_shape = input_shape)

    self$layer_1 <- layer_dense(units = self$units,
                                input_shape = input_shape)

    if (self$use_context)
      self$context_layer <- layer_dense(
        units    = self$units,
        use_bias = FALSE
      )

    self$elu_layer    <- layer_activation(activation = 'elu')
    self$layer_2      <- layer_dense(units = self$units)
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

    c(gated_output, gate) %<-% self$gating_layer(hidden)

    output <- layer_add()(list(skip, gated_output))
    output <- self$norm(output)

    if (self$return_gate)
      return(list(output, gate))
    else
      return(output)

  }

)



