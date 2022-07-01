#'  Variable Selection Network block
#'
#' @param dropout_rate Droput rate
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
    ...){

    super()$`__init__`(name = name, ...)

    self$hidden_dim   <- hidden_dim
    self$state_size   <- state_size
    self$dropout_rate <- dropout_rate

  },

  build = function(input_shape){

    num_features <- as.integer(input_shape[2])
    state_size   <- as.integer(input_shape[3])

    self$weights_grn <- layer_grn(
      input_shape  = num_features * state_size,
      hidden_units = self$hidden_dim,
      output_size  = num_features,
      dropout_rate = self$dropout_rate
    )

    # Tutaj jest indeksowanie z Pythona, trzeba uważać
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

    inputs_to_weights <- layer_flatten()(inputs)
    variable_weights  <- self$weights_grn(inputs_to_weights)
    variable_weights  <- self$softmax(variable_weights)
    variable_weights  <- layer_lambda(f = function(x) k_expand_dims(x, -1))(variable_weights)

    processed_inputs <- list()

    # Aplikujemy indywidualne wartwy GRN
    for (i in 1:self$num_features) {
      # layer_lambda do utrzymania wymiarów
      inp <- layer_lambda(f = function(x) k_expand_dims(x, 2))(inputs[,i,])
      processed <- self[[as.character(i)]](inp)
      processed_inputs <- append(processed_inputs, processed)
    }

    # Składamy wszystko z powrotem
    processed_inputs <- layer_concatenate(processed_inputs, axis = 1)
    outputs <- processed_inputs * variable_weights

    # Likwidujemy drugi wymiar (liczba zmiennych kategorycznych)
    # Zostaje wymiar oznaczający state size
    outputs <- layer_lambda(f = function(x) k_sum(x, axis = 2))(outputs)

    # Dimensions:
    # outputs: [num_samples x state_size]
    list(outputs, variable_weights)
  }


)
