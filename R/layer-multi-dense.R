#' Multiple mdeddings in one layer
#'
#' @inheritParams keras::layer_dense
#'
#' @examples
#'
#' # ==========================================================================
#' #                          SIMPLE CONCATENATION
#' # ==========================================================================
#'
#' inp <- layer_input(c(28, 3))
#' md <- layer_multi_dense(units = c(4, 6, 8))(inp)
#'
#' md_model <- keras_model(inp, md)
#'
#' dummy_input <- array(1, dim = c(1, 28, 3))
#'
#' out <- md_model(dummy_input)
#' dim(out)
#'
#' # ==========================================================================
#' #                          NEW DIMESNION
#' # ==========================================================================
#'
#' inp <- layer_input(c(28, 3))
#' md <- layer_multi_dense(units = 5, new_dim = TRUE)(inp)
#'
#' md_model <- keras_model(inp, md)
#'
#' dummy_input <- array(1, dim = c(1, 28, 3))
#'
#' out <- md_model(dummy_input)
#' dim(out)
#'
#' @export
layer_multi_dense <- keras::new_layer_class(

  classname = "MultiDense",

  initialize = function(units, new_dim = FALSE, ...){

    super()$`__init__`(...)

    self$units   <- as.integer(units)
    self$new_dim <- new_dim
    self$len     <- length(units)
  },

  build = function(input_shape){

    # Unexpected behaviour of seq(length(self$input_dims))
    last_dim <- as.integer(tail(input_shape, 1))

    if (last_dim > 0 & (last_dim != self$len)) {
      .units <<- as.integer(rep(self$units[[1]], last_dim))
      self$units <- .units
      self$len  <- length(.units)
    }

    if (self$new_dim & (var(.units) != 0))
      stop("You canot add a new dimension since output spaces differ!")

    for (i in seq(self$len)) {
      self[[as.character(i)]] <-
        layer_dense(
          units  = self$units[[i-1]]
        )
    }

  },

  call = function(inputs){

    outputs   <- list()
    orig_dims <- dim(inputs)

    for (i in seq(self$len)) {

      inp <-
        k_expand_dims(inputs[tensorflow::all_dims(), i])

      outputs[[i]] <-
        self[[as.character(i)]](inp)
    }

    concat_dim <- -1

    if (self$new_dim) {

      # It's because keras ignores first dimension
      new_dim <- length(orig_dims)

      outputs <- Map(
        f = \(inp) layer_lambda(f = \(x) k_expand_dims(x, axis = new_dim))(inp),
        outputs
      )

      concat_dim <- -2
    }

    layer_concatenate(outputs, axis = concat_dim)
  }

)
