library(keras)

test_that("Test layer_lmu", {

  # Return last vector only
  inp         <- layer_input(c(28, 3))
  hidden_cell <- layer_lstm_cell(10)
  lmu         <- layer_lmu(
    memory_d    = 10,
    order       = 3,
    theta       = 28,
    hidden_cell = hidden_cell,
    return_sequences = FALSE
  )(inp)
  model       <- keras_model(inp, lmu)

  out <- model(array(1, c(32, 28, 3)))
  expect_equal(dim(out), c(32, 10))

  # Return sequences
  inp         <- layer_input(c(28, 3))
  hidden_cell <- layer_lstm_cell(10)
  lmu         <- layer_lmu(
    memory_d    = 10,
    order       = 3,
    theta       = 28,
    hidden_cell = hidden_cell,
    return_sequences = TRUE
  )(inp)
  model       <- keras_model(inp, lmu)

  out <- model(array(1, c(32, 28, 3)))
  expect_equal(dim(out), c(32, 28, 10))

})
