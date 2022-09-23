#include <Rcpp.h>

using namespace Rcpp;

// See: https://gist.github.com/krzjoa/7761420abfaa49e00fb176ee4f49e820
// In C++ transformed to the zero-based index
int flat_index2(int dim1, int dim2,
                int idx1, int idx2, int idx3){
  return idx1 + idx2 * dim1 + idx3 * dim1 * dim2;
}

int flat_index_mat(int dim1, int dim2,
                   int idx1, int idx2){
  return idx1 + idx2 * dim1;
}

// Allocate required arrays (tensors)
void alloc_arrays2(List& list, DataFrame& ts_starts,
                   int& timesteps, List& vars,
                   CharacterVector& names){
  for (int i = 0; i < vars.length(); i++){
    int n_vars = ((CharacterVector) vars[i]).length();
    NumericVector array(Dimension(ts_starts.nrow(), timesteps, n_vars));
    list.push_back(array, as<std::string>(names[i]));
  }
}

void allocco(List& list, DataFrame& ts_starts,
                    List& vars, CharacterVector& names){
  for (int i = 0; i < vars.length(); i++){
    int n_vars = ((CharacterVector) vars[i]).length();
    NumericVector array(Dimension(ts_starts.nrow(), n_vars));
    list.push_back(array, as<std::string>(names[i]));
  }
}


// Fill arrays with data from the data.table
void fill_arrays2(DataFrame& data, List& list,
                  DataFrame& ts_starts, List& vars,
                  int& timesteps, int list_start_idx,
                  int row_start_shift){

  for (int i = 0; i < ts_starts.nrow(); i++) {
    int row_start_idx = as<NumericVector>(ts_starts["row_idx"])[i] - 1 + row_start_shift;

    for (int var = 0; var < vars.length(); var++){
      CharacterVector var_set = as<CharacterVector>(vars[var]);

      for (int col = 0; col < var_set.length(); col++) {
        std::string col_name = as<std::string>(var_set[col]);

        for (int l=0; l < timesteps; l++) {
          int f_idx = flat_index2(ts_starts.nrow(), timesteps, i, l, col);
          ((NumericVector)list[list_start_idx + var])[f_idx] = as<NumericVector>(data[col_name])[row_start_idx+l];
        }
      }
    }
  }
}



// Fill arrays with data from the data.table
void fillo(DataFrame& data, List& list,
           DataFrame& ts_starts, List& vars,
           int list_start_idx){

  for (int i = 0; i < ts_starts.nrow(); i++) {
    int row_start_idx = as<NumericVector>(ts_starts["row_idx"])[i] - 1;

    for (int var = 0; var < vars.length(); var++){
      CharacterVector var_set = as<CharacterVector>(vars[var]);

      for (int col = 0; col < var_set.length(); col++) {
        std::string col_name = as<std::string>(var_set[col]);
        int fidx = flat_index_mat(ts_starts.nrow(), 1, i, col);
        ((NumericVector)list[list_start_idx + var])[fidx] = as<NumericVector>(data[col_name])[row_start_idx];

      }
    }
  }
}

// [[Rcpp::export]]
List trollo(DataFrame data, DataFrame ts_starts,
            int lookback, int horizon,
            List past_var, List fut_var, List static_var){

  // ===========================================================================
  //                            ALLOCATING ARRAYS
  // ===========================================================================
  // https://rcppcore.github.io/RcppParallel/
  List list = List::create();
  CharacterVector past_names = past_var.names();
  CharacterVector fut_names = fut_var.names();
  CharacterVector static_names = static_var.names();

  // Allocate past arrays
  // Type conversion is not needed, is done by keras
  alloc_arrays2(list, ts_starts, lookback, past_var, past_names);
  alloc_arrays2(list, ts_starts, horizon, fut_var, fut_names);
  allocco(list, ts_starts, static_var, static_names);

  // ===========================================================================
  //                            INSERTING DATA
  // ===========================================================================

  // TODO: get first loop out of the fill_arrays
  // Test, if it can speed up the function execution

  fill_arrays2(data, list, ts_starts, past_var, lookback, 0, 0);
  fill_arrays2(data, list, ts_starts, fut_var, horizon, past_var.length(), lookback);
  fillo(data, list, ts_starts, static_var, past_var.length() + fut_var.length());

  return list;
}
