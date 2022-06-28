library(keras)
library(tensorflow)

#' Temporal Fusion Trannsformer
#'
#' Paper: https://arxiv.org/abs/1912.09363
#' Original implementation: https://github.com/google-research/google-research/blob/master/tft/libs/tft_model.py
#' Alternatives:
#' * https://github.com/LiamMaclean216/Temporal-Fusion-Transformer
#' * https://github.com/PlaytikaResearch/tft-torch/blob/main/tft_torch/tft.py

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
  layer_glu_2(10, input_shape = 30)

model %>%
   compile(optimizer = "adam", loss = "mse")

model

model(matrix(1, 32, 30))

# With gate
inp  <- layer_input(30)
out  <- layer_glu_2(units = 10, return_gate = TRUE)(inp)

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

tensor_shape <- c(28, 30, 10)

model <-
  keras_model_sequential() %>%
  layer_grn(hidden_units = 10, output_size = 2,
            input_shape = tensor_shape)

model %>%
  compile(optimizer = "adam", loss = "mse")

model
out <- model(array(1, dim = c(2, tensor_shape)))
dim(out)

# With gate
tensor_shape <- c(28, 30, 10)

inp  <- layer_input(tensor_shape)
out  <- layer_grn(hidden_units = 10, output_size = 2,
                  input_shape = 30, return_gate = TRUE)(inp)

model <- keras_model(inp, out)

model %>%
  compile(optimizer = "adam", loss = "mse")

model

double <- model(array(1, dim = c(2, tensor_shape)))
class(double)
double[[1]]
double[[2]]

# With additional context
tensor_shape  <- c(28, 30, 10)
context_shape <- 20

inp     <- layer_input(tensor_shape)
context <- layer_input(context_shape)

out  <- layer_grn(hidden_units = 10, output_size = 2,
                  input_shape = 30, return_gate = FALSE,
                  use_additional_context = TRUE)(inp, context)

model <- keras_model(
  inputs  = list(inp, context),
  outputs = out
)

output <-
  model(list(
    array(1, dim = c(2, tensor_shape)),
    array(1, dim = c(2, context_shape))
  ))


