cmake_minimum_required(VERSION 3.20)

project(Euler_MR)

find_package(samurai)
find_package(CLI11)

include(FindPkgConfig)
pkg_check_modules(PETSC PETSc)

if(PETSC_FOUND)
    find_package(MPI)
endif()

set(CMAKE_CXX_STANDARD 17)

add_executable(Euler_MR main.cpp)
target_link_libraries(Euler_MR samurai::samurai CLI11::CLI11)
