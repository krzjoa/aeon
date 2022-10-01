#' Temporal Fusion Decoder layer
#'
#' One of blocks, the TFT model consists of. During the call, it accepts two
#' parameters:
#' * output of the LSTM
#' * static context
#' The LSTM output is enriched with the static context features and additionally
#' processed ath the end.
#'
#' @include layer-interpretable-mh-attention.R layer-grn.R
#'
#' @examples
#' lookback   <- 28
#' horizon    <- 14
#' all_steps  <- lookback + horizon
#' state_size <- 5
#'
#' lstm_output <- layer_input(c(all_steps, state_size))
#' context     <- layer_input(state_size)
#'
#' # No attentiion scores returned
#' tdf <- layer_temporal_fusion_decoder(
#'    hidden_units = 30,
#'    state_size = state_size,
#'    use_context = TRUE,
#'    num_heads = 10
#' )(lstm_output, context)
#'
#' # With attention scores
#' c(tfd, attention_scores) %<-%
#'    layer_temporal_fusion_decoder(
#'       hidden_units = 30,
#'       state_size = state_size,
#'       use_context = TRUE,
#'       num_heads = 10
#'    )(lstm_output, context, return_attention_scores=TRUE)
#' @export
layer_temporal_fusion_decoder <- keras::new_layer_class(

 classname = "TemporalFusionDecoder",

 initialize = function(hidden_units, state_size, dropout_rate = 0.,
                       use_context, num_heads, ...){

     super()$`__init__`(...)

     self$hidden_units <- hidden_units
     self$state_size   <- state_size
     self$dropout_rate <- dropout_rate
     self$use_context  <- use_context
     self$num_heads    <- num_heads

   },

   build = function(input_shape){

     # Enrichment GRN
     self$enrichment_grn <- layer_grn(
       hidden_units = self$hidden_units,
       output_size  = self$state_size,
       dropout_rate = self$dropout_rate,
       use_context  = self$use_context
     )

     # Interpretable Multi-Head Attention
     self$imh_attention <-
       layer_interpretable_mh_attention(
         state_size   = self$state_size,
         num_heads    = self$num_heads,
         dropout_rate = self$dropout_rate
       )

     # Final GRN
     self$final_grn <- layer_grn(
       hidden_units = self$hidden_units,
       output_size  = self$state_size,
       dropout_rate = self$dropout_rate
     )
   },

   call = function(inputs, context=NULL,
                   return_attention_scores=FALSE){

     enriched_lstm_output <-
       self$enrichment_grn(inputs, context)

     query  <- enriched_lstm_output
     keys   <- enriched_lstm_output
     values <- enriched_lstm_output

     # mask <- get_decoder_mask(enriched_lstm_output)
     mask <- NULL

     c(gated_post_attention, attention_scores) %<-%
       self$imh_attention(query, keys, values, mask = mask,
                          return_attention_scores=TRUE)

     # Position-wise feed-forward
     post_poswise_ff_grn <- self$final_grn(gated_post_attention)

     if (return_attention_scores)
       return(list(post_poswise_ff_grn, attention_scores))
     else
       return(post_poswise_ff_grn)

   }

)
