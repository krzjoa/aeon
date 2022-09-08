#' Variable Selection Network block
#'
#' It receives four-dimensional vector as an input in the case of dynamic data
#' (batch_size, timesteps, n_features, feature_dim)
#'
#' @inheritParams layer-grn.R
#' @param return_weights Return weights of the selection.
#' @param state_size Dimensionality of the feature space, common across the model.
#' The name comes from [the original paper](https://arxiv.org/pdf/1912.09363.pdf)
#' where they also refer to as \deqn{d_model}
#'
#' @returns
#' A tensor of shapes:
#'
#' * dynamic data - (batch_size, timesteps, state_size)
#' * static data - (batch_size, state_size)
#'
#' @include layer-grn.R
#'
#' @examples
#'
#' # =========================================================================
#' #               THREE-DIMENSIONAL INPUT (STATIC FEATURES)
#' # =========================================================================
#'
#' # input: (batch_size, n_features, state_size)
#'
#' inp <- layer_input(c(10, 5))
#' out <- layer_vsn(hidden_units = 10, state_size = 5)(inp)
#' dim(out)
#'
#' # =========================================================================
#' #               FOUR-DIMENSIONAL INPUT (DYNAMIC FEATURES)
#' # =========================================================================
#'
#' # input: (batch_size, timesteps, n_features, state_size)
#'
#' inp <- layer_input(c(28, 10, 5))
#' out <- layer_vsn(hidden_units = 10, state_size = 5)(inp)
#' dim(out)
#' @export
layer_vsn <- keras::new_layer_class(

  classname = "VSN",

  initialize = function(
    hidden_units,
    state_size,
    dropout_rate = NULL,
    use_context = FALSE,
    return_weights = FALSE,
    ...){

    super()$`__init__`(...)

    self$hidden_units   <- hidden_units
    self$state_size     <- state_size
    self$dropout_rate   <- dropout_rate
    self$use_context    <- use_context
    self$return_weights <- return_weights

  },

  build = function(input_shape){

    num_features <- as.integer(rev(input_shape)[2])
    state_size   <- as.integer(rev(input_shape)[1])

    # For four-dimenensional space
    if (length(input_shape) == 4)
      self$reshape <- layer_lambda(
        f = function(x) k_reshape(x, c(
          -1, as.numeric(input_shape[2]),
          num_features * state_size
        ))
      )
    else if (length(input_shape) == 3)
      self$reshape <- layer_flatten()

    self$weights_grn <- layer_grn(
      #input_shape  = num_features * state_size,
      hidden_units = self$hidden_units,
      output_size  = num_features,
      dropout_rate = self$dropout_rate,
      use_context  = self$use_context
    )

    # Be careful: Python indexing
    self$softmax      <- layer_activation_softmax(axis = -1)
    self$input_grns   <- as.list(vector(length = num_features))
    self$num_features <- num_features

    # We don't add context here
    for (i in 1:num_features) {

      .grn <- layer_grn(
        hidden_units = self$hidden_units,
        output_size  =  self$state_size,
        dropout_rate = self$dropout_rate,
        use_context  = FALSE
      )

      self[[as.character(i)]] <- .grn
    }

  },

  call = function(inputs, context = NULL){

    inputs_to_weights <- self$reshape(inputs)
    variable_weights  <- self$weights_grn(inputs_to_weights, context)
    variable_weights  <- self$softmax(variable_weights)
    variable_weights  <- layer_lambda(f = function(x) k_expand_dims(x, -1))(variable_weights)

    processed_inputs <- list()

    for (i in 1:self$num_features) {
      inp <- layer_lambda(f = function(x) k_expand_dims(x, -2))(inputs[all_dims(), i,])
      processed <- self[[as.character(i)]](inp)
      processed_inputs <- append(processed_inputs, processed)
    }

    processed_inputs <- layer_concatenate(processed_inputs, axis = -2)
    outputs <- processed_inputs * variable_weights

    outputs <- layer_lambda(f = function(x) k_sum(x, axis = -2))(outputs)

    if (self$return_weights)
      return(list(outputs, variable_weights))
    else
      return(outputs)
  }

)
