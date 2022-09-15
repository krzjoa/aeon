#include <Rcpp.h>

using namespace Rcpp;

// See: https://gist.github.com/krzjoa/7761420abfaa49e00fb176ee4f49e820
// In C++ transformed to the zero-based index
int flat_index(int dim1, int dim2,
               int idx1, int idx2, int idx3){
  idx1++;
  int output = idx1 + idx2 * dim1 + idx3 * dim1 * dim2;
  return output-1;
}

// Allocate required arrays (tensors)
void alloc_arrays(List& list, DataFrame& ts_starts,
                  int& timesteps, List& vars,
                  CharacterVector& names){
  for (int i = 0; i < vars.length(); i++){
    int n_vars = ((CharacterVector) vars[i]).length();
    NumericVector array(Dimension(ts_starts.nrow(), timesteps, n_vars));
    list.push_back(array, as<std::string>(names[i]));
  }
}


void fill_arrays(DataFrame& data, List& list,
                 DataFrame& ts_starts, List& vars,
                 int& timesteps, int list_start_idx){
  for (int i = 0; i < ts_starts.nrow(); i++) {
    int row_start_idx = as<NumericVector>(ts_starts["row_idx"])[i] - 1;

    for (int var = 0; var < vars.length(); var++){
      CharacterVector var_set = as<CharacterVector>(vars[var]);

      for (int col = 0; col < var_set.length(); col++) {
        std::string col_name = as<std::string>(var_set[col]);

        for (int l=0; l < timesteps; l++) {
          int f_idx = flat_index(ts_starts.nrow(), timesteps, i, l, col);
          ((NumericVector)list[list_start_idx + var])[f_idx] = as<NumericVector>(data[col_name])[row_start_idx+l];
        }
      }
    }
  }
}


// [[Rcpp::export]]
List get_arrays(DataFrame data, DataFrame ts_starts,
                int lookback, int horizon,
                List past_var, List fut_var){

  // ===========================================================================
  //                            ALLOCATING ARRAYS
  // ===========================================================================
  // https://rcppcore.github.io/RcppParallel/
  List list = List::create();
  CharacterVector past_names = past_var.names();
  CharacterVector fut_names = fut_var.names();

  // Allocate past arrays
  // Type conversion is not needed, is done by keras
  alloc_arrays(list, ts_starts, lookback, past_var, past_names);
  alloc_arrays(list, ts_starts, horizon, fut_var, fut_names);

  // ===========================================================================
  //                            INSERTING DATA
  // ===========================================================================

  // PAST
  fill_arrays(data, list, ts_starts, past_var, lookback, 0);

  // FUTURE
  fill_arrays(data, list, ts_starts, fut_var, horizon, past_var.length());

  // consider using iterator
  // https://teuder.github.io/rcpp4everyone_en/280_iterator.html
  // for (int i = 0; i < ts_starts.nrow(); i++) {
  //   int row_start_idx = as<NumericVector>(ts_starts["row_idx"])[i] - 1;
  //
  //   for (int var = 0; var < past_var.length(); var++){
  //     CharacterVector var_set = as<CharacterVector>(past_var[var]);
  //
  //     for (int col = 0; col < var_set.length(); col++) {
  //       std::string col_name = as<std::string>(var_set[col]);
  //
  //       for (int l=0; l < lookback; l++) {
  //         int f_idx = flat_index(ts_starts.nrow(), lookback, i, l, col);
  //         ((NumericVector)list[var])[f_idx] = as<NumericVector>(data[col_name])[row_start_idx+l];
  //       }
  //     }
  //   }
  // }

  // ============================== FUTURE =====================================



  // for (int i = 0; i < ts_starts.nrow(); i++) {
  //   int row_start_idx = as<NumericVector>(ts_starts["row_idx"])[i] - 1;
  //
  //   for (int var = 0; var < fut_var.length(); var++){
  //     CharacterVector var_set = as<CharacterVector>(fut_var[var]);
  //
  //     for (int col = 0; col < var_set.length(); col++) {
  //       std::string col_name = as<std::string>(var_set[col]);
  //
  //       for (int h=0; h < horizon; h++) {
  //         int f_idx = flat_index(ts_starts.nrow(), horizon, i, h, col);
  //         ((NumericVector)list[past_var.length() + var])[f_idx] = as<NumericVector>(data[col_name])[row_start_idx+lookback+h];
  //       }
  //     }
  //   }
  // }

  return list;
}

// x <- aion:::get_arrays(m5::tiny_m5, m5::tiny_m5, m5::tiny_m5, 28, 28, list(a=1, b=2, c=3), list(d=1, e=2, f=3))
