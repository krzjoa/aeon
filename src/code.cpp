#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
List get_arrays(DataFrame data, DataFrame data_index, DataFrame slices,
                int lookback, int horizon, List past_var, List fut_var){
  List list = List::create();

  // Allocate past arrays
  for (int i = 0; i < past_var.length(); i++){
    // list
  }

  // Allocate future arrays
  for (int i = 0; i < fut_var.length(); i++){
    // list
  }

  return list;
}
