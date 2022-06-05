
VSN <- Layer(

  classname = "VSN",

  initialize = function(
    n_var_cont,
    n_var_disc,
    dim_model,
    dropout_rate = NULL,
    name = NULL, ...){

    super()$`__init__`(name = name, ...)

    self$dim_model   <- dim_model
    self$n_var_cont  <- n_var_cont
    self$n_var_total <- n_var_cont + length(n_var_disc)

  },


  build = function(input_shape){

    #Linear transformation of inputs into dmodel vector
    self$linearise <- list()

    for (i in 1:self$n_var_cont)
      self$linearise <-
        append(self$linearise, layer_dense(units = self$dim_model, use_bias = FALSE))
      self$fc <- layer_dense(units = self$dim_model, use_bias = FALSE)

    # entity embedings for discrete inputs
    self$entiy_embed <- list()
    for (i in self$n_var_disc)
      self$entiy_embed <-
        append(self$entiy_embed, layer_dense(units = self$dim_model, use_bias = FALSE))

    self$input_grn  <- layer_grn(
      input_shape  = input_shape,
      dropout_rate = dropout_rate
      )

    self$vs_grn    <- layer_grn(
      input_shape  = self$n_var_total * input_shape,
      output_size  = self$n_var_total,
      dropout_rate = dropout_rate
    )

  },



  call = function(x_cont = NULL, x_disc = NULL){

    # Linear transformation of the numeric nputs
    linearised <- list()

    for (i in 1:length(x_cont)) {
      slice      <- layer_lambda(f = function(x)x[,,i])(x_cont)
      linearised <- append(linearised, self$fc(slice))
    }

    # Enity embeddings for the discrete inputs
    embedded <- list()

    for (i in 1:length(x_disc)) {
      fc <- self$entiy_embed[[i]]
      embedded <- append(
        embedded, fc(x_disc[[i]])
      )
    }

    # Stacking variables
    stacked <- layer_concatenate(axis = 3)(c(linearised, embedded))

    #flatten everything except accross batch for variable selection weights
    vs_weights <- self$vs_grn(
      layer_reshape(target_shape = c())(stacked)
    )






  }

)

# Variable Selection Network block
layer_vsn <- function(object,
                      n_var_cont,
                      n_var_disc,
                      dim_model,
                      input_shape,
                      dropout_rate = NULL,
                      name = NULL, ...){

  args <- list(
    units         = as.integer(units),
    input_shape   = keras:::normalize_shape(input_shape),
    activation    = activation,
    return_gate   = return_gate,
    name          = name
  )

  create_layer(VSN, object, args)

}
