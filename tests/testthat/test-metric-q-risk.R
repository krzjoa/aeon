test_that("Test metric_q_risk", {

  y_pred <- array(runif(60), c(2, 10, 1))
  y_true <- array(runif(20), c(2, 10, 1))

  expect_error(metric_q_risk(quantile=1.5)(y_pred, y_true))
  expect_s3_class(
    metric_q_risk(quantile=0.5)(y_pred, y_true),
    "tensorflow.tensor"
  )
})
