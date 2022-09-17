#' General negative log likelihood loss function
#'
#' Bear in mind, that the number of the model outputs must reflect the number
#' of distribution parameters. For example, if you use normal distribution ([tfprobability::tfd_normal()]),
#' which is described with two parameters (mean and standard deviation), the model
#' should return two values per each timestep. In othr words, it produces a distribution
#' as a forecast rather than a point estimate. When the model is trained, we have two options
#' to generate the final forecast:
#' * use the expected value of the distribution (e.g. mean for normal distribution)
#' * sample a value from the distribution
#' Additionally, having the distribution we can compute prediction intervals.
#' Remeber also about the constraints imposed on the parameter values, e.g. standard deviation must be positive.
#'
#'
#' @param distribution A probability distribution function from `tfprobability` package.
#' Default: [tfprobability::tfd_normal()]
#'
#' @references
#' 1. [Cross-Entropy, Negative Log-Likelihood, and All That Jazz](https://towardsdatascience.com/cross-entropy-negative-log-likelihood-and-all-that-jazz-47a95bd2e81)
#' 2. D. Salinas, V. Flunkert, J. Gasthaus, T. Januschowski, [DeepAR: Probabilistic forecasting with autoregressive recurrent networks, International Journal of Forecasting](https://arxiv.org/abs/1704.04110)(2019)
#'
#' @examples
#' y_pred <- array(runif(60), c(2, 10, 2))
#' y_true <- array(runif(20), c(2, 10, 1))
#'
#' loss_negative_log_likelihood(
#'     distribution = tfprobability::tfd_normal,
#'     reduction = 'auto'
#'  )(y_true, y_pred)
#' loss_negative_log_likelihood(reduction = 'sum')(y_true, y_pred)
#' @export
loss_negative_log_likelihood <- keras::new_loss_class(

  classname = "NegativeLogLikelihood",

  initialize = function(distribution = tfprobability::tfd_normal, ...){
    super()$`__init__`(...)
    self$distribution <- distribution
  },

  call = function(y_true, y_pred){

    y_pred <- keras::k_cast(tensorflow::tf$constant(y_pred), 'float32')
    y_true <- keras::k_cast(y_true, 'float32')

    last_dim <- tail(dim(y_pred), 1)

    args <-
      Map(function(n) y_pred[tensorflow::all_dims(), n, tensorflow::tf$newaxis],
          1:last_dim)

    distr <- do.call(self$distribution, args)

    # https://rstudio.github.io/tfprobability/articles/layer_dense_variational.html
    - distr$log_prob(y_true)
  }

)

# colour cdad75 c7bd65
# 0054AD
