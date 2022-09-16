#' Legendre Memory Unit layer
#'
#' A layer of trainable low-dimensional delay systems.
#' Each unit buffers its encoded input
#' by internally representing a low-dimensional
#' (i.e., compressed) version of the sliding window.
#' Nonlinear decodings of this representation,
#' expressed by the A and B matrices, provide
#' computations across the window, such as its
#' derivative, energy, median value, etc ([1]_, [2]_).
#' Note that these decoder matrices can span across
#' all of the units of an input sequence.
#'
#' @param memory_d Dimensionality of input to memory component.
#' @param order The number of degrees in the transfer function of the LTI system used to
#' represent the sliding window of history. This parameter sets the number of
#' Legendre polynomials used to orthogonally represent the sliding window.
#' @param theta The number of timesteps in the sliding window that is represented using the
#' LTI system. In this context, the sliding window represents a dynamic range of
#' data, of fixed size, that will be used to predict the value at the next time
#' step. If this value is smaller than the size of the input sequence, only that
#' number of steps will be represented at the time of prediction, however the
#' entire sequence will still be processed in order for information to be
#' projected to and from the hidden layer. If `trainable_theta` is enabled, then
#' theta will be updated during the course of training.
#' @param hidden_cell Keras Layer/RNNCell implementing the hidden component.
#' @param trainable_theta If TRUE, theta is learnt over the course of training. Otherwise, it is kept
#' constant.
#' @param hidden_to_memory If TRUE, connect the output of the hidden component back to the memory
#' component (default FALSE).
#' @param memory_to_memory If TRUE, add a learnable recurrent connection (in addition to the static
#' @param input_to_hidden If TRUE, connect the input directly to the hidden component (in addition to
#' @param discretizer The method used to discretize the A and B matrices of the LMU. Current
#' options are "zoh" (short for Zero Order Hold) and "euler".
#' "zoh" is more accurate, but training will be slower than "euler" if
#' ``trainable_theta=TRUE``. Note that a larger theta is needed when discretizing
#' using "euler" (a value that is larger than ``4*order`` is recommended).
#' @param kernel_initializer Initializer for weights from input to memory/hidden component. If ``NULL``,
#' no weights will be used, and the input size must match the memory/hidden size.
#' @param recurrent_initializer Initializer for ``memory_to_memory`` weights (if that connection is enabled).
#' @param kernel_regularizer Regularizer for weights from input to memory/hidden component.
#' @param recurrent_regularizer Regularizer for ``memory_to_memory`` weights (if that connection is enabled).
#' @param use_bias If TRUE, the memory component includes a bias term.
#' @param bias_initializer Initializer for the memory component bias term. Only used if ``use_bias=TRUE``.
#' @param bias_regularizer Regularizer for the memory component bias term. Only used if ``use_bias=TRUE``.
#' @param dropout Dropout rate on input connections.
#' @param recurrent_dropout Dropout rate on ``memory_to_memory`` connection.
#' @param return_sequences If TRUE, return the full output sequence. Otherwise, return just the last
#' output in the output sequence.
#'
#' @include utils.R
#'
#' @references
#' 1. [Voelker, Kajic I. and Eliasmith, Legendre Memory Units: Continuous-Time Representation in Recurrent Neural Networks](http://compneuro.uwaterloo.ca/files/publications/voelker.2019.lmu.pdf)
#' 2. [Voelker and Eliasmith (2018). Improving spiking dynamical networks: Accurate delays, higher-order synapses, and time cells. Neural Computation, 30(3): 569-609.](http://compneuro.uwaterloo.ca/files/publications/voelker.2018.pdf)
#' 3. [Voelker and Eliasmith. "Methods and systems for implementing dynamic neural networks." U.S. Patent Application No. 15/243,223.](https://patents.google.com/patent/US20180053090A1/en)
#' 4. [Is LSTM (Long Short-Term Memory) dead?, CrossValidated](https://stats.stackexchange.com/questions/472822/is-lstm-long-short-term-memory-dead)
#'
#' @examples
#' \donttest{
#' library(keras)
#' inp <- layer_input(c(28, 3))
#' hidden_cell <- layer_lstm_cell(10)
#' lmu <- layer_lmu(memory_d=10, order=3, theta=28, hidden_cell=hidden_cell)(inp)
#' model <- keras_model(inp, lmu)
#' model(array(1, c(32, 28, 3)))
#' }
#' @export
layer_lmu <- function(object,
                      memory_d,
                      order,
                      theta,
                      hidden_cell,
                      trainable_theta=FALSE,
                      hidden_to_memory=FALSE,
                      memory_to_memory=FALSE,
                      input_to_hidden=FALSE,
                      discretizer="zoh",
                      kernel_initializer="glorot_uniform",
                      recurrent_initializer="orthogonal",
                      kernel_regularizer=NULL,
                      recurrent_regularizer=NULL,
                      use_bias=FALSE,
                      bias_initializer="zeros",
                      bias_regularizer=NULL,
                      dropout=0,
                      recurrent_dropout=0,
                      return_sequences=FALSE,
                      ...){

  args <- list(
    memory_d              = as.integer(memory_d),
    order                 = as.integer(order),
    theta                 = theta,
    hidden_cell           = hidden_cell,
    trainable_theta       = trainable_theta,
    hidden_to_memory      = hidden_to_memory,
    memory_to_memory      = memory_to_memory,
    input_to_hidden       = input_to_hidden,
    discretizer           = discretizer,
    kernel_initializer    = kernel_initializer,
    recurrent_initializer = recurrent_initializer,
    kernel_regularizer    = kernel_regularizer,
    recurrent_regularizer = recurrent_regularizer,
    use_bias              = use_bias,
    bias_initializer      = bias_initializer,
    bias_regularizer      = bias_regularizer,
    dropout               = dropout,
    recurrent_dropout     = recurrent_dropout,
    return_sequences      = return_sequences
  )

  args <- append(args, list(...))

  keras.lmu <- try_import(
    name = 'keras_lmu',
    site = 'https://github.com/nengo/keras-lmu'
  )

  create_layer(keras.lmu$LMU, object, args)
}

