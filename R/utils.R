#' @keywords internal
resolve_variables <- function(period_set, type_set){
  if (is.null(period_set))
    return(type_set)
  base::intersect(period_set, type_set)
}

#' Var function, which accepts one-element vectors
#' @keywords internal
safe_var <- function(x){
  if (length(x) == 1)
    return(0)
  else
    return(var(x))
}

#' @keywords internal
try_import <- function(name, site){
  if (!reticulate::py_module_available(name))
    stop(glue::glue("Module {name} is not available and must be installed first. ",
                    "See: {site}"))
  reticulate::import(name)
}

#' @keywords internal
remove_nulls <- function(l){
  Filter(\(x)!is.null(x), l)
}

#' Get cardinalities of the categorical variables
#'
#' Needed as an input to the `[layer_multi_embedding()]`
#'
#' @param data A `data.frame` object
#' @param categorical List of the categorical variables
#'
#' @examples
#' dict_size(m5::tiny_m5, c('event_name_1', 'event_type_1'))
#' @export
dict_size <- function(data, categorical){
  sapply(as.data.frame(data)[categorical], dplyr::n_distinct)
}
