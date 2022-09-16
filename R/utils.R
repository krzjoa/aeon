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
