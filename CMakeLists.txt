cmake_minimum_required(VERSION 3.13)
project(ADMM)

set(CMAKE_CXX_STANDARD 14)

find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)
find_package(xtensor-blas REQUIRED)

add_executable(xtensor_lasso src/xtensor_lasso.cpp)

target_link_libraries(xtensor_lasso xtensor xtensor-blas)

