#' Tweedie Loss (negative log likelihood)
#'
#' @inheritParams loss-functions
#' @param p Power parameter from the [0, 2] range. It allows to choose a desired distribution from the Tweedie distributions family.
#'
#' @references
#' * [Tweedie Loss Function](https://towardsdatascience.com/tweedie-loss-function-for-right-skewed-data-2c5ca470678f)
#' * [p parameter](https://stats.stackexchange.com/questions/123598/tweedie-p-parameter-interpretation)
#'
#' @note
#' For the moment, you can only use `loss_quantile` to instantiate a class
#' and not call directly like other loss in `keras`. Please see:
#' [#1342 issue](https://github.com/rstudio/keras/issues/1342)
#'
#' @examples
#' y_pred <- array(runif(60), c(2, 10, 1))
#' y_true <- array(runif(20), c(2, 10, 1))
#'
#' # As a callable object
#' loss_tweedie(p = 1.5, reduction = 'auto')(y_true, y_pred)
#' loss_tweedie(p = 1.5, reduction = 'sum')(y_true, y_pred)
#'
#' @export
loss_tweedie <- keras::new_loss_class(

  classname = "TweedieLoss",

  initialize = function(p = 1.5, ...){
    super()$`__init__`(...)
    self$p <- self$.validate_p(p)
  },

  call = function(y_true, y_pred){
    a <- y_true * (y_pred ** (1-self$p)) / (1-self$p)
    b <- (y_pred ** (2-self$p))/(2-self$p)
    -a + b
  },

  .validate_p = function(p){
    if (p < 0 | p > 2) {
      stop("Cannot define p parameter out of the [0, 2] range!")
    }
    p
  }

)

