library(keras)

test_that("Test layer_temporal_fusion_decoder", {

  lookback   <- 28
  horizon    <- 14
  all_steps  <- lookback + horizon
  state_size <- 5
  num_heads  <- 10

  lstm_output <- layer_input(c(all_steps, state_size))
  context     <- layer_input(state_size)

  # No attention scores returned
  tdf <- layer_temporal_fusion_decoder(
     hidden_units = 30,
     state_size   = state_size,
     use_context  = TRUE,
     num_heads    = num_heads
  )(lstm_output, context)

  expect_equal(dim(tdf), c(NA, all_steps, state_size))

  # With attention scores
  c(tfd, attention_scores) %<-%
     layer_temporal_fusion_decoder(
        hidden_units = 30,
        state_size   = state_size,
        use_context  = TRUE,
        num_heads    = num_heads
     )(lstm_output, context, return_attention_scores=TRUE)

  expect_equal(dim(tdf), c(NA, all_steps, state_size))
  expect_equal(dim(attention_scores), c(num_heads, NA, all_steps, all_steps))

})
