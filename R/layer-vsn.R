#' Variable Selection Network block
#'
#' It receives four-dimensional vector as an input in the case of dynamic data
#' (batch_size, timesteps, n_features, feature_dim)
#'
#' @param dropout_rate Droput rate
#' @inheritParams layer-grn.R
#'
#' @returns
#' A t
#'
#' @include layer-grn.R
#'
#' @export
layer_vsn <- keras::new_layer_class(

  classname = "VSN",

  initialize = function(
    hidden_dim,
    state_size,
    dropout_rate = NULL,
    return_weights = FALSE,
    ...){

    super()$`__init__`(...)

    self$hidden_dim   <- hidden_dim
    self$state_size   <- state_size
    self$dropout_rate <- dropout_rate
    self$use_context  <- use_context

  },

  build = function(input_shape){

    # GRN do generowania wag
    # na wejściu (batch_size, num_inputs * state_size)
    # Wejście do GRN od wag ma inny kształt, niż wejście do indywidualnych GRNów
    # Softmax daje wartoœści per feature wejściowy:
    # np. 0.56 dla ceny, 0.1 dla temepratury itp.
    # Ale żeby to policzyć, warstwa dense musi widzieć "wszystko"


    num_features <- as.integer(rev(input_shape)[2])
    state_size   <- as.integer(rev(input_shape)[1])

    self$reshape <- layer_lambda(
      f = function(x) k_reshape(x, c(
        -1, as.numeric(input_shape[2]),
        num_features * state_size
      ))
    )

    self$weights_grn <- layer_grn(
      input_shape  = num_features * state_size,
      hidden_units = self$hidden_dim,
      output_size  = num_features,
      dropout_rate = self$dropout_rate,
      use_context  = self$use_context
    )

    # Be careful: Python indexing
    self$softmax <- layer_activation_softmax(axis = -1)

    # Dodatkowo, każda zmienna dostaje własną warstwę GRN
    self$input_grns <- as.list(vector(length = num_features))

    self$num_features <- num_features


    # Do tych GRN-ów nie dodajemy kontekstu
    for (i in 1:num_features) {

      .grn <- layer_grn(
        hidden_units = self$hidden_dim,
        output_size  =  self$state_size, # Wydaje mi się, że tutaj i tak musi być 1
        dropout_rate = self$dropout_rate
      )

      self[[as.character(i)]] <- .grn
    }

  },

  call = function(inputs, context = NULL){

    # Wejście (batch_size, n_features, state_size)
    # Liczymy wagi dla każdego feature'u wejściowego
    # Pamiętajmy, że każda zmienna została zrzutowana krok wcześniej na wiele ficzerów
    # browser()

    inputs_to_weights <- self$reshape(inputs)
    variable_weights  <- self$weights_grn(inputs_to_weights, context)
    variable_weights  <- self$softmax(variable_weights)
    variable_weights  <- layer_lambda(f = function(x) k_expand_dims(x, -1))(variable_weights)

    processed_inputs <- list()

    # Aplikujemy indywidualne wartwy GRN
    for (i in 1:self$num_features) {
      # layer_lambda do utrzymania wymiarów
      inp <- layer_lambda(f = function(x) k_expand_dims(x, 3))(inputs[,,i,])
      processed <- self[[as.character(i)]](inp)
      processed_inputs <- append(processed_inputs, processed)
    }

    # Składamy wszystko z powrotem
    processed_inputs <- layer_concatenate(processed_inputs, axis = 2)
    outputs <- processed_inputs * variable_weights

    # Likwidujemy drugi wymiar (liczba zmiennych kategorycznych)
    # Zostaje wymiar oznaczający state size
    outputs <- layer_lambda(f = function(x) k_sum(x, axis = 3))(outputs)

    # Dimensions:
    # outputs: [num_samples x state_size]

    if (self$return_weights)
      return(list(outputs, variable_weights))
    else
      return(outputs)
  }

)
