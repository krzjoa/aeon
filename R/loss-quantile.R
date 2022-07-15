#' Quantile loss function
#'
#' A generalized version of the quantile loss. The model can predict multiple
#' quantiles at once. Loss value for a single sample-timestep-quantile is computed as:
#' \deqn{QL(y_t, y^{\hat}_t, q) = max(q(y - y^{\hat}))}
#'
#' @importFrom keras keras_array
#' @import tensorflow
#'
#' @examples
#' y_pred <- array(runif(60), c(2, 10, 3))
#' y_true <- array(runif(20), c(2, 10, 1))
#'
#' # As an object
#' loss_quantile(quantiles = c(0.1, 0.5, 0.9), reduction = 'auto')(y_true, y_pred)
#' loss_quantile(quantiles = c(0.1, 0.5, 0.9), reduction = 'sum')(y_true, y_pred)
#'
#' # As a function
#' loss_quantile(y_true, y_pred)
#'
#' @references
#' https://www.evergreeninnovations.co/blog-quantile-loss-function-for-machine-learning/
#'
#' @export
loss_quantile <- keras::new_loss_class(

  classname = "QuantileLoss",

  # Report a bug

  initialize = function(quantiles=NULL, reduction = 'auto', ...){
    super()$`__init__`(reduction = reduction, ...)
    self$quantiles <- self$.validate_quantiles(quantiles)
  },

  call = function(y_true, y_pred, quantiles, reduction = 'auto'){

    if (missing(quantiles))
      quantiles <- self$quantiles
    else
      quantiles <- self$.validate_quantiles(quantiles)

    quantiles <- array(quantiles, c(1, 1, length(quantiles)))
    quantiles <- keras_array(quantiles)

    errors <- tf$subtract(y_pred, y_true)
    errors <- k_cast(errors, 'float32')

    loss   <- tf$maximum(tf$subtract(quantiles, 1) * errors,
                         quantiles * errors)
    loss
  },

  .validate_quantiles = function(quantiles){
    if (any(quantiles > 1) | any(quantiles < 0)) {
      stop("It contains quatiles out of the [0, 1] range!")
    }
    quantiles
  }

)

# loss_negative_log_likelihood
