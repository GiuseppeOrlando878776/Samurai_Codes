cmake_minimum_required(VERSION 3.20)

project(relaxation)

find_package(samurai)
find_package(CLI11)
find_package(nlohmann_json REQUIRED)
find_package(MPI)

include(FindPkgConfig)
pkg_check_modules(PETSC PETSc)


set(CMAKE_CXX_STANDARD 23)

add_executable(BN_test main.cpp)
target_link_libraries(BN_test samurai::samurai CLI11::CLI11 nlohmann_json::nlohmann_json)
