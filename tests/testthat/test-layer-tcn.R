library(keras)

test_that("Test layer_lmu", {

  NB_FILTERS <- 64

  # Last vector only
  inp   <- layer_input(c(28, 3))
  tcn   <- layer_tcn(
    nb_filters  = NB_FILTERS,
    kernel_size = 3,
    nb_stacks   = 1,
    dilations   = c(1, 7, 14)
  )(inp)
  model <- keras_model(inp, tcn)

  out <- model(array(1, c(32, 28, 3)))
  expect_equal(dim(out), c(32, NB_FILTERS))

  # Return sequences
  inp   <- layer_input(c(28, 3))
  tcn   <- layer_tcn(
    nb_filters  = NB_FILTERS,
    kernel_size = 3,
    nb_stacks   = 1,
    dilations   = c(1, 7, 14),
    return_sequences = TRUE
  )(inp)
  model <- keras_model(inp, tcn)

  out <- model(array(1, c(32, 28, 3)))
  expect_equal(dim(out), c(32, 28, NB_FILTERS))

})
