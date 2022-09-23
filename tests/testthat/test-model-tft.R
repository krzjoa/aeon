library(keras)


test_that("Test model_tft with static features", {

  horizon     <- 14
  output_size <- 3
  batch_size  <- 32

  tft <- model_tft(
     lookback                = 28,
     horizon                 = horizon,
     past_numeric_size       = 5,
     past_categorical_size   = 2,
     future_numeric_size     = 4,
     future_categorical_size = 2,
     vocab_static_size       = c(5, 5),
     vocab_dynamic_size      = c(4, 4),
     hidden_dim              = 12,
     state_size              = 7,
     num_heads               = 10,
     dropout_rate            = 0.1,
     output_size             = output_size
  )

  x_static_cat <- array(sample(5, 32 * 2, replace=TRUE), c(32, 2)) - 1
  x_static_num <- rand_array(32, 1)

  x_past_num <- rand_array(32, 28, 2)
  x_past_cat <- array(sample(4, 32 * 28 * 2, replace=TRUE), c(32, 28, 5))

  x_fut_num <- rand_array(32, 28, 1)
  x_fut_cat <- array(sample(4, 32 * 14 * 2, replace=TRUE), c(32, 28, 5))

  inputs <-
    list(x_past_num, x_past_cat,
         x_fut_num, x_fut_cat,
         x_static_num, x_static_cat)

  output <- tft(inputs)

  expect_equal(dim(output), c(batch_size, horizon, output_size))

})


test_that("Test model_tft with partial static features", {

  horizon     <- 14
  output_size <- 3
  batch_size  <- 32

  tft <- model_tft(
    lookback                = 28,
    horizon                 = horizon,
    past_numeric_size       = 5,
    past_categorical_size   = 2,
    future_numeric_size     = 4,
    future_categorical_size = 2,
    vocab_static_size       = c(5, 5),
    vocab_dynamic_size      = c(4, 4),
    hidden_dim              = 12,
    state_size              = 7,
    num_heads               = 10,
    dropout_rate            = 0.1,
    output_size             = output_size
  )

  x_static_cat <- array(sample(5, 32 * 2, replace=TRUE), c(32, 2)) - 1

  x_past_num <- rand_array(32, 28, 2)
  x_past_cat <- array(sample(4, 32 * 28 * 2, replace=TRUE), c(32, 28, 5))

  x_fut_num <- rand_array(32, horizon, 1)
  x_fut_cat <- array(sample(4, 32 * horizon * 2, replace=TRUE), c(32, horizon, 5))

  inputs <-
    list(x_past_num, x_past_cat,
         x_fut_num, x_fut_cat,
         NULL, x_static_cat)

  output <- tft(inputs)

  expect_equal(dim(output), c(batch_size, horizon, output_size))

})



test_that("Test model_tft without static features", {

  horizon     <- 14
  output_size <- 3
  batch_size  <- 32

  tft <- model_tft(
    lookback                = 28,
    horizon                 = horizon,
    past_numeric_size       = 5,
    past_categorical_size   = 2,
    future_numeric_size     = 4,
    future_categorical_size = 2,
    vocab_static_size       = c(5, 5),
    vocab_dynamic_size      = c(4, 4),
    hidden_dim              = 12,
    state_size              = 7,
    num_heads               = 10,
    dropout_rate            = 0.1,
    output_size             = output_size,
    use_context             = FALSE
  )

  x_past_num <- rand_array(32, 28, 2)
  x_past_cat <- array(sample(4, 32 * 28 * 2, replace=TRUE), c(32, 28, 5))

  x_fut_num <- rand_array(32, 28, 1)
  x_fut_cat <- array(sample(4, 32 * 14 * 2, replace=TRUE), c(32, 28, 5))

  inputs <-
    list(x_past_num, x_past_cat,
         x_fut_num, x_fut_cat,
         NULL, NULL)

  output <- tft(inputs)

  expect_equal(dim(output), c(batch_size, horizon, output_size))

})

