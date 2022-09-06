#' q-Risk metric
#'
#' @export
metric_q_risk <- keras::new_metric_class(

  classname = "qRisk",

  initialize = function(quantile, ...){
    super()$`__init__`( ...)
    self$quantile <- quantile
  }

)
