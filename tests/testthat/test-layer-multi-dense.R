
test_that("Test MultiDense with simple concat", {

  dim_out <- c(4, 6, 8)

  inp <- layer_input(c(28, 3))
  md  <- layer_multi_dense(units = dim_out)(inp)

  md_model <- keras_model(inp, md)

  dummy_input <- array(1, dim = c(1, 28, 3))

  out <- md_model(dummy_input)

  expect_true(dim(out)[[3]] == sum(dim_out))
  expect_true(length(dim(out)) ==  3)

})


test_that("Test MultiDense with new dimension", {

  inp <- layer_input(c(28, 3))
  md  <- layer_multi_dense(units = 5, new_dim = TRUE)(inp)

  md_model <- keras_model(inp, md)

  dummy_input <- array(1, dim = c(1, 28, 3))

  out <- md_model(dummy_input)
  dim(out)

  expect_true(length(dim(out)) ==  4)
  expect_true(dim(out)[[3]] == 3)
  expect_true(dim(out)[[4]] == 5)

})

test_that("Test MultiDense with wrong args", {
  dim_out <- c(3, 4, 5)
  inp <- layer_input(c(28, 3))
  expect_error(
    layer_multi_dense(units = dim_out, new_dim = TRUE)(inp)
  )
})

