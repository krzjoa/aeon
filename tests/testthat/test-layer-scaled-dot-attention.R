test_that("Test layer_scaled_dot_attention", {

  samples    <- 32
  lookback   <- 28
  horizon    <- 14
  all_steps  <- lookback + horizon
  state_size <- 5

  queries <- layer_input(c(horizon, state_size))
  keys    <- layer_input(c(all_steps, state_size))
  values  <- layer_input(c(all_steps, state_size))

  sdp_attention <- layer_scaled_dot_attention()(queries, keys, values)

  model <- keras_model(
    inputs  = list(queries, keys, values),
    outputs = sdp_attention
  )

  out <- model(
    list(rand_array(samples, horizon, state_size),
         rand_array(samples, all_steps, state_size),
         rand_array(samples, all_steps, state_size))
  )

  expect_equal(dim(out), c(samples, horizon, state_size))

})
