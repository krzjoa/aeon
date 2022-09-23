# install.packages("m5")
# devtools::install_github("krzjoa/aion")

# Dataset
library(m5)

# Neural Networks
library(aion)
library(keras)

# Data wrangling
library(dplyr, warn.conflicts=FALSE)
library(data.table, warn.conflicts=FALSE)
library(recipes, warn.conflicts=FALSE)

library(microbenchmark)

# ==============================================================================
#                           DATA PREPARATION
# ==============================================================================

train <- tiny_m5[date < '2016-01-01']

m5_recipe <-
  recipe(value ~ ., data=train) %>%
  step_mutate(item_id_idx=item_id, store_id_idx=store_id) %>%
  step_integer(item_id_idx, store_id_idx,
               wday, month,
               event_name_1, event_type_1,
               event_name_2, event_type_2,
               zero_based=TRUE) %>%
  step_naomit(all_predictors()) %>%
  prep()

train <- bake(m5_recipe, train)

# ==============================================================================
#                               CONFIG
# ==============================================================================

TARGET      <- 'value'
STATIC_CAT  <- c('store_id_idx')
STATIC_NUM  <- 'item_id_idx'
DYNAMIC_CAT <- c('event_name_1', 'event_type_1')
CATEGORICAL <- c(DYNAMIC_CAT, STATIC_CAT)
NUMERIC     <- c('sell_price', 'sell_price', 'item_id_idx')
KEY         <- c('item_id', 'store_id')
INDEX       <- 'date'
LOOKBACK    <- 28
HORIZON     <- 14
STRIDE      <- LOOKBACK
BATCH_SIZE  <- 32

c(x, y) %<-%  make_arrays_candidate(
  data        = train,
  key         = KEY,
  index       = INDEX,
  lookback    = LOOKBACK,
  horizon     = HORIZON,
  stride      = STRIDE,
  target      = TARGET,
  static      = c(STATIC_CAT, STATIC_NUM),
  categorical = CATEGORICAL,
  numeric     = NUMERIC,
  shuffle     = TRUE
)

x$x_static_num

devtools::document()

# ==============================================================================
#                         PREPARING ARRAYS (TENSORS)
# ==============================================================================

microbenchmark(
  make_arrays_1 = make_arrays(
    data        = train,
    key         = KEY,
    index       = INDEX,
    lookback    = LOOKBACK,
    horizon     = HORIZON,
    stride      = STRIDE,
    target      = TARGET,
    static      = c(STATIC_CAT, STATIC_NUM),
    categorical = CATEGORICAL,
    numeric     = NUMERIC,
    shuffle     = TRUE
  ),

  make_arrays_cand = make_arrays_candidate(
    data        = train,
    key         = KEY,
    index       = INDEX,
    lookback    = LOOKBACK,
    horizon     = HORIZON,
    stride      = STRIDE,
    target      = TARGET,
    static      = c(STATIC_CAT, STATIC_NUM),
    categorical = CATEGORICAL,
    numeric     = NUMERIC,
    shuffle     = TRUE
  ),
  times = 20
)

