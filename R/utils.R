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
