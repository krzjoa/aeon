library(keras)
library(tensorflow)

#' Temporal Fusion Trannsformer
#'
#' Paper: https://arxiv.org/abs/1912.09363
#' Original implementation: https://github.com/google-research/google-research/blob/master/tft/libs/tft_model.py
#' Alternative: https://github.com/LiamMaclean216/Temporal-Fusion-Transformer

input   <- layer_input(30)

dense_1 <- layer_dense(units = 10)(input)
dense_2 <- layer_dense(units = 10)(input)

layer_concatenate(inputs = list(dense_1, dens))


# ============================================================================
#                         SIMPLE DENSE FOR COMPARISON
# ============================================================================

model <-
  keras_model_sequential() %>%
  layer_dense(10, input_shape = 30)

model %>%
  compile(optimizer = "adam", loss = "mse")

model

# ============================================================================
#                                   GLU
# ============================================================================

model <-
  keras_model_sequential() %>%
  layer_glu(10, input_shape = 30)

model %>%
   compile(optimizer = "adam", loss = "mse")

model

model(matrix(1, 32, 30))

# With gate
inp  <- layer_input(30)
out  <- layer_glu(units = 10, return_gate = TRUE)(inp)

model <- keras_model(inp, out)

model %>%
  compile(optimizer = "adam", loss = "mse")

model

double <- model(matrix(1, 32, 30))
class(double)
double[[1]]
double[[2]]


# ============================================================================
#                                   GRN
# ============================================================================


model <-
  keras_model_sequential() %>%
  layer_grn(hidden_units = 10, output_size = 2,
            input_shape = 30)

model %>%
  compile(optimizer = "adam", loss = "mse")

model
model(matrix(1, 32, 30))


# With gate
inp  <- layer_input(30)
out  <- layer_grn(hidden_units = 10, output_size = 2,
                  input_shape = 30, return_gate = TRUE)(inp)

model <- keras_model(inp, out)

model %>%
  compile(optimizer = "adam", loss = "mse")

model

double <- model(matrix(1, 32, 30))
class(double)
double[[1]]
double[[2]]
