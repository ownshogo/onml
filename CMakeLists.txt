cmake_minimum_required (VERSION 3.0)

project (onml)

set (CMAKE_CXX_STANDARD 14)
set (CMAKE_BUILD_TYPE Release)
enable_testing()

add_library(onml SHARED
    src/linear_regressor.cpp
    src/loss.cpp
    src/optimizer.cpp
    src/linear_classifier.cpp
    src/file_reader.cpp)
target_include_directories(onml PUBLIC include extern)

add_library(catch_main SHARED test/catch_main.cpp)
target_include_directories(catch_main PUBLIC extern/catch2)

foreach(TEST_NAME linear_regressor linear_classifier)
  add_executable(test_${TEST_NAME} test/test_${TEST_NAME}.cpp)
  target_link_libraries(test_${TEST_NAME} onml catch_main)
  add_test(NAME ${TEST_NAME} COMMAND test_${TEST_NAME} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/test) 
endforeach(TEST_NAME)
