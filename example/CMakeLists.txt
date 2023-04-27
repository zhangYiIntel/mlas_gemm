cmake_minimum_required(VERSION 3.8)
project(ov_test)

add_executable(mlas_test main.cpp)
target_link_libraries(mlas_test ${CMAKE_SOURCE_DIR}/../build/libonnxruntime_mlas.a)
target_include_directories(mlas_test PUBLIC ../inc)