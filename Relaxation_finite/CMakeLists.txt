cmake_minimum_required(VERSION 3.20)

project(relaxation)

find_package(samurai 0.27.0)
find_package(CLI11)
find_package(nlohmann_json REQUIRED)
find_package(MPI)

#include(FindPkgConfig)
#pkg_check_modules(PETSC PETSc)

set(CMAKE_CXX_STANDARD 23)

add_executable(BN_test BN_solver.cpp)
target_link_libraries(BN_test samurai::samurai CLI11::CLI11 nlohmann_json::nlohmann_json)
target_include_directories(BN_test PRIVATE include)
