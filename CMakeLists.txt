cmake_minimum_required (VERSION 3.0)

project (onml)

add_library(onml SHARED)
target_sources(onml PRIVATE src/linear_regressor.cpp src/loss.cpp)
target_include_directories(onml PUBLIC include extern)

add_library(catch_main SHARED)
target_sources(catch_main PRIVATE test/catch_main.cpp)
target_include_directories(catch_main PUBLIC extern/catch2)

add_executable(test_linear_regressor test/test_linear_regressor.cpp)
target_link_libraries(test_linear_regressor onml catch_main)
