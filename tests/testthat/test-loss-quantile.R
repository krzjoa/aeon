test_that("Test loss_quantile", {

  y_pred <- array(runif(60), c(2, 10, 3))
  y_true <- array(runif(20), c(2, 10, 1))

  expect_error(loss_quantile(quantiles=1.5)(y_pred, y_true))
  expect_s3_class(
    loss_quantile(quantiles=0.5)(y_pred, y_true),
    "tensorflow.tensor"
  )

  y_pred_2 <- array(runif(60), c(2, 10, 2))
  y_true_2 <- array(runif(20), c(2, 10, 1))

  # Num of quantiles != num of outputs
  expect_error(
    loss_quantile(quantiles=c(0.1, 0.5, 0.9))(y_pred_2, y_true_2)
  )

})
