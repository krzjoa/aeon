#' General negative log likelihood loss function
#'
#' @param distribution A probability distribution function from `tfprobability` package.
#'
#'
#' @examples
#' y_pred <- array(runif(60), c(2, 10, 2))
#' y_true <- array(runif(20), c(2, 10, 1))
#'
#' # As a callable object
#' loss_negative_log_likelihood(reduction = 'auto')(y_true, y_pred)
#' loss_negative_log_likelihood(reduction = 'sum')(y_true, y_pred)
#'
#' # As a function
#' loss_negative_log_likelihood(y_true, y_pred)
#'
#' @export
loss_negative_log_likelihood <- keras::new_loss_class(

  classname = "NegativeLogLikelihood",

  initialize = function(distribution = tfprobability::tfd_normal, ...){
    super()$`__init__`(...)
    self$distribution <- distribution
  },

  call = function(y_true, y_pred, distribution){

    if (missing(distribution))
      distribution <- self$distribution

    args <- Map(function(n) y_pred[all_dims(), n], tail(dim(y_pred), 1))

    distr <- do.call(self$distribution, args)

    # https://rstudio.github.io/tfprobability/articles/layer_dense_variational.html
    - distr$log_prob(y_true)
  }

)

# colour cdad75 c7bd65
