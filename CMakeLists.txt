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
find_package(MPI REQUIRED)

add_executable(xtensor_lasso src/xtensor_lasso.cpp)

#set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
#set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})

target_include_directories(xtensor_lasso PRIVATE ${MPI_CXX_INCLUDE_PATH})
target_compile_options(xtensor_lasso PRIVATE ${MPI_CXX_COMPILE_FLAGS})

target_link_libraries(xtensor_lasso ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS} xtensor xtensor-blas xtensor::optimize ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

