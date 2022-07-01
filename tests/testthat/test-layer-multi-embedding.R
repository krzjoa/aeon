
test_that("Test MultiEmbedding with simple concat", {

  dim_out <- c(3, 4, 5)

  inp <- layer_input(c(28, 3))
  emb <- layer_multi_embedding(input_dims = c(4, 6, 8), output_dims = dim_out)(inp)

  emb_model <- keras_model(inp, emb)

  dummy_input <- array(1, dim = c(1, 28, 3))
  dummy_input[,,1] <- sample(4,size = 28, replace = TRUE)
  dummy_input[,,2] <- sample(6,size = 28, replace = TRUE)
  dummy_input[,,3] <- sample(8,size = 28, replace = TRUE)

  out <- emb_model(dummy_input)
  dim(out)


  expect_true(dim(out)[[3]] == sum(dim_out))
  expect_true(length(dim(out)) ==  3)

})


test_that("Test MultiEmbedding with new dimension", {

  inp <- layer_input(c(28, 3))
  emb <- layer_multi_embedding(input_dims = c(4, 6, 8), output_dims = 5, new_dim = 5)(inp)

  emb_model <- keras_model(inp, emb)

  dummy_input <- array(1, dim = c(1, 28, 3))
  dummy_input[,,1] <- sample(4,size = 28, replace = TRUE)
  dummy_input[,,2] <- sample(6,size = 28, replace = TRUE)
  dummy_input[,,3] <- sample(8,size = 28, replace = TRUE)

  out <- emb_model(dummy_input)
  dim(out)

  expect_true(length(dim(out)) ==  4)
  expect_true(dim(out)[[3]] == 3)
  expect_true(dim(out)[[4]] == 5)

})

test_that("Test MultiEmbedding with wrong args", {
  dim_out <- c(3, 4, 5)
  inp <- layer_input(c(28, 3))
  expect_error(
    layer_multi_embedding(input_dims = c(4, 6, 8), output_dims = dim_out, new_dim = TRUE)(inp)
  )
})

