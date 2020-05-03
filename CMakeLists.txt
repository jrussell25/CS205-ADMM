cmake_minimum_required(VERSION 3.13)
project(ADMM)

set(XTENSOR_USE_SIMD 1)
set(XTENSOR_USE_OPENMP 1)

add_definitions(-DHAVE_CBLAS=1)

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

message(STATUS "BLAS VENDOR:    " ${BLA_VENDOR})
message(STATUS "BLAS LIBRARIES: " ${BLAS_LIBRARIES})

set(CMAKE_CXX_STANDARD 14)

find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)
find_package(xtensor-blas REQUIRED)

add_executable(xtensor_lasso src/xtensor_lasso.cpp)

target_link_libraries(xtensor_lasso xtensor xtensor-blas xtensor::optimize ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

