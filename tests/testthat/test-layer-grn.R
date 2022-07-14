
test_that("Test GRN with return_gate = TRUE and context", {


  inp  <- layer_input(c(28, 5))
  ctx  <- layer_input(10)
  out  <- layer_grn(
              hidden_units = 10,
              return_gate  = TRUE,
              use_context  = TRUE
           )(inp, context = ctx)

  model <- keras_model(list(inp, ctx), out)

  model

  model %>%
     compile(optimizer = "adam", loss = "mse")

  arr_1 <- array(1, dim = c(1, 28, 5))
  arr_2 <- array(1, dim = c(1, 10))

  c(values, gate) %<-% model(list(arr_1, arr_2))

  expect_true(all(dim(values) == dim(gate)))
  expect_true(all(as.array(gate) <= 1))
  expect_true(all(as.array(gate) >= 0))

})

