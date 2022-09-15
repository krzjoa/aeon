#' Quantile (Pinball) loss function
#'
#' A generalized version of the quantile loss. The model can predict multiple
#' quantiles at once.
#'
#' Loss value for a single sample-timestep-quantile is computed as:
#' \deqn{QL(y_t, \hat{y}_t, q) = max(q(y_t - \hat{y}_t), (q - 1)(y_t - \hat{y}_t))}
#' or equivalently as :
#' \deqn{QL(y_t, \hat{y}_t, q) = max(q(y_t - \hat{y}_t), 0) +  max((1 - q)(\hat{y}_t - y_t), 0)}
#'
#' When multiple quantiles are defined, the generalized, averaged loss is computed according to the equation:
#' \deqn{\mathcal{L}(\Omega, W) = \Sigma_{y_t \in \Omega}\Sigma_{q \in \mathcal{Q}}\Sigma^{\tau_{max}}_{\tau=1} = \frac{QL(y_t, \hat{y}(q, t - \tau, \tau), q)}{M_{\tau_{max}}}}
#'
#' The loss function is computed as above when `reduction = 'auto` or `reduction = 'mean'`.
#'
#' @note
#' For the moment, you can only use `loss_quantile` to instantiate a class
#' and not call directly like other loss in `keras`. Please see:
#' [#1342 issue](https://github.com/rstudio/keras/issues/1342)
#'
#' @inheritParams loss-functions
#' @param quantiles List of quantiles (numeric vector with values between 0 and 1).
#'
#' @importFrom keras keras_array
#' @import tensorflow
#'
#' @examples
#' y_pred <- array(runif(60), c(2, 10, 3))
#' y_true <- array(runif(20), c(2, 10, 1))
#'
#' loss_quantile(quantiles = c(0.1, 0.5, 0.9), reduction = 'auto')(y_true, y_pred)
#' loss_quantile(quantiles = c(0.1, 0.5, 0.9), reduction = 'sum')(y_true, y_pred)
#' @references
#'  * [Quantile loss function for machine learning](https://www.evergreeninnovations.co/blog-quantile-loss-function-for-machine-learning/)
#'  * [Pinball loss function (Lokad)](https://www.lokad.com/pinball-loss-function-definition)
#'  * [Temporal Fusion Transformer](https://arxiv.org/pdf/1912.09363.pdf)
#'  * [Original TFT implementation by Google](https://github.com/google-research/google-research/blob/f87f702c0f78f1cdc19bf167123c43304d01ee08/tft/libs/tft_model.py#L1059)
#'
#' @export
loss_quantile <- keras::new_loss_class(

  classname = "QuantileLoss",

  initialize = function(quantiles=NULL, ...){
    super()$`__init__`( ...)
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

#' @rdname loss_quantile
#' @export
loss_pinball <- loss_quantile
