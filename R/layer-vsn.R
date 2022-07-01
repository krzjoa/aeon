#'  Variable Selection Network block
#'
#'
#' @export
layer_vsn <- keras::new_layer_class(

  classname = "VSN",

  initialize = function(
    input_dim,
    num_inputs,
    hidden_dim,
    dropout_rate = NULL,
    use_context = FALSE,
    name = NULL, ...){

    super()$`__init__`(name = name, ...)

    # Każda zmienna (jest `num_inputs` zmiennych)
    # jest reprezentowana przez wektor o długośc input_dim
    # W praktyce możemy więc uzyskać tensor czterowymiarowy
    # (batch_size, timesteps, n_features, feature_dim)
    # Przykładowo: cenę zamieniamy na wektor o długości 4,
    # podbnie jak sprzedaż czy indeks sklepu
    # 4 będzie wówczas ostatnim wymiarem tensora

    self$input_dim    <- input_dim
    self$num_inputs   <- num_inputs
    self$hidden_dim   <- hidden_dim
    self$dropout_rate <- dropout_rate
    self$use_context  <- use_context

  },

  build = function(input_shape){

    # GRN do generowania wag
    self$weights_grn <- layer_grn(
      # input_shape  = self$input_dim * self$num_inputs,
      hidden_units = self$hidden_dim,
      output_size  = self$num_inputs,
      use_additional_context = self$use_context,
      dropout_rate = self$dropout_rate
    )

    self$softmax <- layer_activation_softmax(axis = 2)

    # Dodatkowo, każda zmienna dostaje własną warstwę GRN
    self$input_grns <- list()

    # Do tych GRN-ów nie dodajemy kontekstu
    for (i in 1:self$num_inputs) {

      .grn <- layer_grn(
        # input_shape  = self$input_dim,
        hidden_units = self$hidden_dim,
        output_size  =  self$hidden_dim, # Wydaje mi się, że tutaj i tak musi być 1
        use_additional_context = FALSE,
        dropout_rate = self$dropout_rate
      )

      self$input_grns <- append(
        self$input_grns,
        .grn
      )

    }

  },

  call = function(inputs, context = NULL){

    # Zakładamy, że inputs to spłaszczone wejście?
    # Na razie tak, dla uproszczenia

    # Będziemy iterowac po liście, stąd to przekształcenie
    # if (keras:::is_keras_tensor(inputs))
    #  inputs <- list(inputs)
    # albo zakładamy, że inputs to tensor trójwymiarowy (po spłaszczeniu z czterowymiarowego)

    # Liczymy wagi dla każdego feature'u wejściowego
    # Pamiętajmy, że każda zmienna została zrzutowana krok wcześniej na wiele ficzerów
    variable_weights <- self$weights_grn(inputs, context)
    variable_weights <- self$softmax(weights)
    variable_weights <- layer_lambda(
      f = function(x) k_expand_dims(x, axis = -1)
    )(variable_weights)
    # After that step "sparse_weights" is of shape [(num_samples * num_temporal_steps) x num_inputs x 1]
    # variable_weights <- layer_permute()

    processed_inputs <- list()

    # Aplikujemy indywidualne wartwy GRN
    for (i in 0:(length(self$input_grns) - 1)) {
      processed <- self$input_grns[[i+1]](
        inputs[, (i * self$input_dim + 1):((i + 1) * self$input_dim)]
      )
      processed_inputs <- append(processed_inputs, processed)
    }

    # Dimensions:
    # processed_inputs: [(num_samples * num_temporal_steps) x state_size x num_inputs]
    # wydaje mi się, że poiwnno być
    # [(num_samples * num_temporal_steps) x num_inputs x state_size]

    # Składamy wszystko z powrotem
    processed_inputs <- layer_concatenate(processed_inputs, axis = -1)

    # weigh them by multiplying with the weights tensor viewed as
    # [(num_samples * num_temporal_steps) x 1 x num_inputs]
    # so that the weight given to each input variable (for each time-step/observation) multiplies the entire state
    # vector representing the specific input variable on this specific time-step
    permuted_weights <-

      outputs <- processed_inputs * sparse_weights.transpose(1, 2)

    # Dimensions:
    # outputs: [(num_samples * num_temporal_steps) x state_size x num_inputs]

    # and finally sum up - for creating a weighted sum representation of width state_size for every time-step
    outputs <- layer_lambda(f = function(x) k_sum(x, axis = -1))(outputs)

    # Dimensions:
    # outputs: [(num_samples * num_temporal_steps) x state_size]


    outputs
  }

)
