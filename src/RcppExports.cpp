// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// get_arrays
List get_arrays(DataFrame data, DataFrame data_index, DataFrame slices, int lookback, int horizon, List past_var, List fut_var);
RcppExport SEXP _aion_get_arrays(SEXP dataSEXP, SEXP data_indexSEXP, SEXP slicesSEXP, SEXP lookbackSEXP, SEXP horizonSEXP, SEXP past_varSEXP, SEXP fut_varSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< DataFrame >::type data(dataSEXP);
    Rcpp::traits::input_parameter< DataFrame >::type data_index(data_indexSEXP);
    Rcpp::traits::input_parameter< DataFrame >::type slices(slicesSEXP);
    Rcpp::traits::input_parameter< int >::type lookback(lookbackSEXP);
    Rcpp::traits::input_parameter< int >::type horizon(horizonSEXP);
    Rcpp::traits::input_parameter< List >::type past_var(past_varSEXP);
    Rcpp::traits::input_parameter< List >::type fut_var(fut_varSEXP);
    rcpp_result_gen = Rcpp::wrap(get_arrays(data, data_index, slices, lookback, horizon, past_var, fut_var));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_aion_get_arrays", (DL_FUNC) &_aion_get_arrays, 7},
    {NULL, NULL, 0}
};

RcppExport void R_init_aion(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
