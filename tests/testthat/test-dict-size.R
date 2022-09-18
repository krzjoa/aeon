test_that("Test dict_size", {
  expect_equal(
    dict_size(m5::tiny_m5, c('event_name_1', 'event_type_1')),
    c(event_name_1=31, event_type_1=5)
  )
})
