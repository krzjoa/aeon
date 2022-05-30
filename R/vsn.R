
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
    self$entiy_embed <- lsit()
    for (i in self$n_var_disc)
      self$entiy_embed <-
        append(self$entiy_embed, layer_dense(units = self$dim_model, use_bias = FALSE))

    self$input_grn  <- layer_grn(
      input_shape  = input_shape,
      dropout_rate = dropout_rate
      )

    self$output_grn <- layer_grn(
      input_shape  = self$n_var_total * input_shape,
      output_size  = self$n_var_total,
      dropout_rate = dropout_rate
    )

  },

  call = function(x_cont, x_disc){

    linearised <- list()

    for (i in 1:length(self$linearise)) {
      slice      <- layer_lambda(f = function(x)x[,,i])(x_cont)
      linearised <- append(linearised, self$fc(slice))
    }

    for

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

}
