
test_that("Test GLU with return_gate = TRUE", {
  inp  <- layer_input(30)
  out  <- layer_glu(units = 10, return_gate = TRUE)(inp)

  model <- keras_model(inp, out)

  c(values, gate) %<-% model(matrix(1, 32, 30))

  expect_true(all(dim(values) == dim(gate)))
  expect_true(all(as.array(gate) <= 1))
  expect_true(all(as.array(gate) >= 0))

})
