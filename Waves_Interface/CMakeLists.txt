cmake_minimum_required(VERSION 3.20)

project(waves_interface)

find_package(samurai)
find_package(CLI11)
find_package(MPI)

include(FindPkgConfig)
pkg_check_modules(PETSC PETSc)

set(CMAKE_CXX_STANDARD 17)

add_executable(waves_interface main.cpp)
target_link_libraries(waves_interface samurai::samurai CLI11::CLI11)
