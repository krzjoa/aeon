#' Residual block for the WaveNet TCN
#'
#' This is a block composed i.a. from causal convolutional layers.
#' It may be considered as a replacement for the recurrent layers.
#'
#' @param dilation_rate The dilation power of 2 we are using for this residual block
#' @param nb_filters The number of convolutional filters to use in this block
#' @param kernel_size The size of the convolutional kernel
#' @param padding The padding used in the convolutional layers, 'same' or 'causal'.
#' @param activation The final activation used in o = Activation(x + F(x))
#' @param dropout_rate Float between 0 and 1. Fraction of the input units to drop.
#' @param kernel_initializer Initializer for the kernel weights matrix (Conv1D).
#' @param use_batch_norm Whether to use batch normalization in the residual layers or not.
#' @param use_layer_norm Whether to use layer normalization in the residual layers or not.
#' @param use_weight_norm Whether to use weight normalization in the residual layers or not.
#'
#' @include utils.R
#'
#' @references
#' 1. [Keras TCN library by Philippe Rémy](https://github.com/philipperemy/keras-tcn)
#' 2. [Bai Sh., Kolter J.Z., Koltun V., An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling, 2018](https://arxiv.org/abs/1803.01271)
#'
#' @examples
#' \donttest{
#' inp <- layer_input(c(28, 3))
#' tcn <- layer_tcn()(inp)
#' model <- keras_model(inp, tcn)
#' model(array(1, c(32, 28, 3)))
#' }
#' @export
layer_tcn <- function(object,
                     nb_filters=64,
                     kernel_size=3,
                     nb_stacks=1,
                     dilations=c(1, 7, 14),
                     padding='causal',
                     use_skip_connections=TRUE,
                     dropout_rate=0.0,
                     return_sequences=FALSE,
                     activation='relu',
                     kernel_initializer='he_normal',
                     use_batch_norm=FALSE,
                     use_layer_norm=FALSE,
                     use_weight_norm=FALSE,
                     input_shape = NULL,
                     ...){

  args <- list(
    input_shape          = keras:::normalize_shape(input_shape),
    nb_filters           = as.integer(nb_filters),
    kernel_size          = as.integer(kernel_size),
    nb_stacks            = as.integer(nb_stacks),
    dilations            = as.integer(dilations),
    padding              = padding,
    use_skip_connections = use_skip_connections,
    dropout_rate         = dropout_rate,
    return_sequences     = return_sequences,
    activation           = activation,
    kernel_initializer   = kernel_initializer,
    use_batch_norm       = use_batch_norm,
    use_layer_norm       = use_layer_norm,
    use_weight_norm      = use_weight_norm
  )

  args <- append(args, list(...))

  tcn <- try_import(
    name = 'tcn',
    site = 'https://github.com/philipperemy/keras-tcn'
  )

  create_layer(tcn$TCN, object, args)
}
