#' q-Risk metric
#'
#' Also referred to as:
#'  * \eqn{\pi-risk} (2)
#'  * \eqn{p*-loss} (3)
#'  * \eqn{ρ-quantile loss R_{ρ}} (4).
#' It's a metric based on the quantile loss.
#' Loss value for a single sample-timestep-quantile is computed as:
#' \deqn{QL(y_t, \hat{y}_t, q) = max(q(y_t - \hat{y}_t), (q - 1)(y_t - \hat{y}_t))}
#' or equivalently as :
#' \deqn{QL(y_t, \hat{y}_t, q) = max(q(y_t - \hat{y}_t), 0) +  max((1 - q)(\hat{y}_t - y_t), 0)}
#' The final form of the metric looks as follows:
#' \deqn{q-Risk = \frac{2\Sigma_{y_t \in \Omega}\Sigma^{\tau_{max}}_{\tau=1}QL(y_t, \hat{y}(q, t - \tau, \tau), q)}{\Sigma_{y_t \in \Omega}\Sigma^{\tau_{max}}_{\tau=1}{|y_t|}}}
#'
#' @param quantile A desired quantile expressed as a numeric in range [0, 1].
#'
#' @seealso [loss_quantile()]
#'
#' @references
#' 1. B. Lim, S.O. Arik, N. Loeff, T. Pfiste, [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363)(2020)
#' 2. D. Salinas, V. Flunkert, J. Gasthaus, T. Januschowski, [DeepAR: Probabilistic forecasting with autoregressive recurrent networks, International Journal of Forecasting](https://arxiv.org/abs/1704.04110)(2019)
#' 3. S. S. Rangapuram, et al., [Deep state space models for time series forecasting, in: NIPS](https://papers.nips.cc/paper/2018/hash/5cf68969fb67aa6082363a6d4e6468e2-Abstract.html)(2018)
#' 4. S. Li, et al., [Enhancing the locality and breaking the memory bottleneck of transformer on time series forecasting, in: NeurIPS](https://arxiv.org/abs/1907.00235)(2019)
#'
#' @examples
#' y_pred <- array(runif(60), c(2, 10, 1))
#' y_true <- array(runif(20), c(2, 10, 1))
#'
#' metric_q_risk(quantile=0.5)(y_pred, y_true)
#' metric_q_risk(quantile=0.9)(y_pred, y_true)
#' @export
metric_q_risk <- keras::new_metric_class(

  classname = "qRisk",

  initialize = function(quantile, ...){
    super()$`__init__`( ...)
    self$quantile <- self$.validate_quantile(quantile)
    self$y_sum  <- 0
    self$errors <- 0
  },

  update_state = function(y_true, y_pred, ...){

    quantile <- array(self$quantile, c(1, 1, 1))
    quantile <- keras::keras_array(quantile)

    errors <- tensorflow::tf$subtract(y_pred, y_true)
    errors <- keras::k_cast(errors, 'float32')

    partial_y_sum <- sum(abs(y_true))

    partial_errors   <-
      tensorflow::tf$maximum(
        tensorflow::tf$subtract(quantile, 1) * errors,
        quantile * errors
      )

    partial_errors <- tensorflow::tf$reduce_sum(partial_errors)

    self$y_sum  <- self$y_sum + partial_y_sum
    self$errors <- self$errors + partial_errors

  },

  result = function(){
    2 * self$errors / self$y_sum
  },

  .validate_quantile = function(quantile){
    if (quantile > 1 | quantile < 0) {
      stop("The sepcified quantile is out of the [0, 1] range!")
    }
    quantile
  }

)

