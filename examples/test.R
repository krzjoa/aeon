library(keras)
library(tensorflow)

# ============================================================================
#                                   GLU
# ============================================================================

library(keras)

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
