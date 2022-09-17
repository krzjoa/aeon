test_that("Test loss_negative_log_likelihood", {

  y_pred <- array(runif(60), c(2, 10, 2))
  y_true <- array(runif(20), c(2, 10, 1))

  expect_s3_class(
    loss_negative_log_likelihood(
      distribution = tfprobability::tfd_normal,
      reduction = 'auto'
    )(y_true, y_pred),
    "tensorflow.tensor"
  )

  # Wrong number of outputs
  y_pred_2 <- array(runif(60), c(2, 10, 1))
  y_true_2 <- array(runif(20), c(2, 10, 1))

  expect_error(
    loss_negative_log_likelihood(
      distribution = tfprobability::tfd_normal,
      reduction = 'auto'
    )(y_true_2, y_pred_2)
  )

})
