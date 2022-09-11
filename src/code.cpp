#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
List get_arrays(DataFrame data, DataFrame data_index, DataFrame slices,
                int lookback, int horizon, List past_var, List fut_var){
  List list = List::create();

  // alocate an array of names

  // Allocate past arrays
  for (int i = 0; i < past_var.length(); i++){
    // list
    list.push_back(1);
  }

  // Allocate future arrays
  for (int i = 0; i < fut_var.length(); i++){
    // list
    // allocate an array
    // add names(?)
    list.push_back(2);
  }

  // assign list of names to the list

  // https://rcppcore.github.io/RcppParallel/

  return list;
}
