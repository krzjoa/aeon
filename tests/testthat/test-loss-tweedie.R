test_that("Test loss_tweedie", {

  y_pred <- array(runif(60), c(2, 10, 1))
  y_true <- array(runif(20), c(2, 10, 1))

  expect_error(loss_tweedie(p=-1.5)(y_pred, y_true))
  expect_error(loss_tweedie(p=123)(y_pred, y_true))

  expect_s3_class(
    loss_tweedie(p=0.5)(y_pred, y_true),
    "tensorflow.tensor"
  )

})
