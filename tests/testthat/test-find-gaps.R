test_that("Test find_gaps", {
  glob_econ <- as.data.table(tsibbledata::global_economy)
  expect_false(find_gaps(glob_econ, 'Country', 'Year'))
  glob_econ <- glob_econ[Year != 1970]
  expect_true(find_gaps(glob_econ, 'Country', 'Year'))
})
