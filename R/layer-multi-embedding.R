#' Multiple embeddings in one layer
#'
#' @inheritParams keras::layer_embedding
#' @param new_dim If TRUE, new dimension is created instead of stacking outputs
#' in the same dimension
#'
#' @examples
#'
#' # ==========================================================================
#' #                          SIMPLE CONCATENATION
#' # ==========================================================================
#'
#' inp <- layer_input(c(28, 3))
#' emb <- layer_multi_embedding(input_dims = c(4, 6, 8), output_dims = c(3, 4, 5))(inp)
#'
#' emb_model <- keras_model(inp, emb)
#'
#' dummy_input <- array(1, dim = c(1, 28, 3))
#' dummy_input[,,1] <- sample(4,size = 28, replace = TRUE)
#' dummy_input[,,2] <- sample(6,size = 28, replace = TRUE)
#' dummy_input[,,3] <- sample(8,size = 28, replace = TRUE)
#'
#' out <- emb_model(dummy_input)
#' dim(out)
#'
#' # ==========================================================================
#' #                          NEW DIMESNION
#' # ==========================================================================
#'
#' inp <- layer_input(c(28, 3))
#' emb <- layer_multi_embedding(input_dims = c(4, 6, 8), output_dims = 5, new_dim = TRUE)(inp)
#'
#' emb_model <- keras_model(inp, emb)
#'
#' dummy_input <- array(1, dim = c(1, 28, 3))
#' dummy_input[,,1] <- sample(4,size = 28, replace = TRUE)
#' dummy_input[,,2] <- sample(6,size = 28, replace = TRUE)
#' dummy_input[,,3] <- sample(8,size = 28, replace = TRUE)
#'
#' out <- emb_model(dummy_input)
#' dim(out)
#' @export
layer_multi_embedding <- keras::new_layer_class(

  classname = "MultiEmbedding",

  initialize = function(input_dims, output_dims, new_dim = FALSE, ...){

    if (length(output_dims) == 1)
      output_dims <- rep(output_dims, length(input_dims))

    if (new_dim & (var(output_dims) != 0))
      stop("You canot add a new dimension since output spaces differ!")

    super()$`__init__`(...)

    self$input_dims  <- as.integer(input_dims)
    self$output_dims <- as.integer(output_dims)
    self$new_dim     <- new_dim
    self$len         <- length(input_dims)
  },

  build = function(input_shape){

    # Unexpected behaviour of seq(length(self$input_dims))

    for (i in seq(self$len)) {
      self[[as.character(i)]] <-
        layer_embedding(
          input_dim  = self$input_dims[[i-1]],
          output_dim = self$output_dims[[i-1]]
        )
    }

  },

  call = function(inputs){

    outputs   <- list()
    orig_dims <- dim(inputs)

   for (i in seq(self$len)) {

      outputs[[i]] <-
        self[[as.character(i)]](inputs[tensorflow::all_dims(), i])
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
