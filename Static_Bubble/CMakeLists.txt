cmake_minimum_required(VERSION 3.20)

project(static_bubble)

find_package(samurai)
find_package(CLI11)
find_package(MPI)

include(FindPkgConfig)
pkg_check_modules(PETSC PETSc)

set(CMAKE_CXX_STANDARD 17)

add_executable(static_bubble static_bubble.cpp)
target_link_libraries(static_bubble samurai::samurai CLI11::CLI11)
