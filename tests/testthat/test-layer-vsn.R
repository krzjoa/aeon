test_that("Test layer VSN", {

  inp <- layer_input(c(10, 5))
  out <- layer_vsn(hidden_units = 10, state_size = 5)(inp)

  expect_equal(length(dim(out)), 2)
  expect_equal(dim(out)[2], 5)


  inp <- layer_input(c(28, 10, 5))
  out <- layer_vsn(hidden_units = 10, state_size = 5)(inp)

  expect_equal(length(dim(out)), 3)
  expect_equal(dim(out)[3], 5)

})
