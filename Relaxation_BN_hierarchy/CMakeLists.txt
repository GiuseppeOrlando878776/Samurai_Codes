cmake_minimum_required(VERSION 3.20)

project(relaxation)

find_package(samurai 0.23.0)
find_package(CLI11)
find_package(nlohmann_json REQUIRED)
find_package(MPI)

#include(FindPkgConfig)
#pkg_check_modules(PETSC PETSc)

set(CMAKE_CXX_STANDARD 17)

add_executable(relaxation relaxation.cpp)
target_link_libraries(relaxation samurai::samurai CLI11::CLI11)
