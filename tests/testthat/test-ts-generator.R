library(data.table)
library(dplyr, warn.conflicts = FALSE)
library(aion)
library(keras)
library(tsibbledata)
library(recipes, warn.conflicts = FALSE)

test_that("Test ts_generator", {

  split_year <- 2000

  ge_recipe <-
    recipe(GDP ~ . , data = glob_econ) %>%
    step_mutate(Country_idx = Country) %>%
    step_integer(Country_idx) %>%
    prep()

  train <-
    glob_econ[Year <= split_year] %>%
    bake(ge_recipe, .)

  KEY         <- 'Country'
  INDEX       <- 'Year'
  TARGET      <- 'GDP'
  NUMERIC     <- c('Growth', 'CPI', 'Imports', 'Exports', 'Population')
  CATEGORICAL <- 'Country_idx'
  STATIC      <- 'Country_idx'

  BATCH_SIZE <- 32
  LOOKBACK   <- 10
  HORIZON    <- 5

  c(train_gen, train_n_steps) %<-%
    ts_generator(
      data        = train,
      key         = KEY,
      index       = INDEX,
      lookback    = LOOKBACK,
      horizon     = HOIRZON,
      stride      = 1,
      shuffle     = TRUE,
      target      = TARGET,
      categorical = CATEGORICAL,
      numeric     = NUMERIC,
      static      = STATIC,
      y_past_sep  = TRUE,
      batch_size  = BATCH_SIZE
    )

  batch <- train_gen()

  expect_equal(dim(batch$X_past_num)[1], BATCH_SIZE)
  expect_equal(dim(batch$X_past_num)[2], 10)
  expect_equal(dim(batch$X_past_num)[3], length(NUMERIC))

  expect_equal(dim(batch$X_fut_cat)[1], BATCH_SIZE)
  expect_equal(dim(batch$X_fut_cat)[2], 5)
  expect_equal(dim(batch$X_fut_cat)[3], length(CATEGORICAL))

})
