#include <Rcpp.h>

using namespace Rcpp;

// [[Rcpp::export]]
List get_arrays(DataFrame data, DataFrame ts_starts,
                int lookback, int horizon, List past_var, List fut_var){

  // ===========================================================================
  //                            ALLOCATING ARRAYS
  // ===========================================================================
  List list = List::create();
  CharacterVector past_names = past_var.names();
  CharacterVector fut_names = fut_var.names();

  // Allocate past arrays
  // TODO: these two loops look similar
  // Ingeter if cat
  for (int i = 0; i < past_var.length(); i++){
    int n_vars = ((CharacterVector) past_var[i]).length();
    NumericVector array(Dimension(ts_starts.nrow(), lookback, n_vars));
    list.push_back(array, as<std::string>(past_names[i]));
  }

  // Allocate future arrays
  for (int i = 0; i < fut_var.length(); i++){
    int n_vars = ((CharacterVector) fut_var[i]).length();
    NumericVector array(Dimension(ts_starts.nrow(), horizon, n_vars));
    list.push_back(array, as<std::string>(fut_names[i]));
  }

  // ===========================================================================
  //                            INSERTING DATA
  // ===========================================================================

  // consider using iterator
  // https://teuder.github.io/rcpp4everyone_en/280_iterator.html
  for (int i = 0; i < ts_starts.nrow(); i++) {
    int row_start_idx = as<NumericVector>(ts_starts["row_idx"])[i] - 1;

    for (int var = 0; var < past_var.length(); var++){
      CharacterVector var_set = as<CharacterVector>(past_var[var]);

      for (int col = 0; col < var_set.length(); col++) {
        std::string col_name = as<std::string>(var_set[col]);

        for (int l=0; l < lookback; l++) {
          ((NumericVector)list[var])[i + l * col * ts_starts.nrow()] = as<NumericVector>(data[col_name])[row_start_idx+l];

        }
      }
    }
  }



  // https://rcppcore.github.io/RcppParallel/

  return list;
}

// x <- aion:::get_arrays(m5::tiny_m5, m5::tiny_m5, m5::tiny_m5, 28, 28, list(a=1, b=2, c=3), list(d=1, e=2, f=3))
