library(data.table)
library(dplyr, warn.conflicts = FALSE)
library(aion)
library(keras)
library(tsibbledata)
library(recipes, warn.conflicts = FALSE)

test_that("Test make_arrays", {

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

  LOOKBACK <- 10
  HORIZON  <- 5

  train_arrays <-
    make_arrays(
      data        = train,
      key         = KEY,
      index       = INDEX,
      lookback    = LOOKBACK,
      horizon     = HORIZON,
      stride      = 4,
      shuffle     = TRUE,
      target      = TARGET,
      categorical = CATEGORICAL,
      numeric     = NUMERIC
    )

  expect_equal(dim(train_arrays$X_past_num)[2], 10)
  expect_equal(dim(train_arrays$X_past_num)[3], length(NUMERIC) + 1)

  expect_equal(dim(train_arrays$X_fut_cat)[2], 5)
  expect_equal(dim(train_arrays$X_fut_cat)[3], length(CATEGORICAL))

})
