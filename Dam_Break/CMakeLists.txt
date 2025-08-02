cmake_minimum_required(VERSION 3.20)

project(dam_break)

set(SAMURAI_WITH_MPI On)
find_package(samurai 0.26.0)
find_package(CLI11)
find_package(nlohmann_json REQUIRED)
find_package(MPI)

include(FindPkgConfig)
pkg_check_modules(PETSC PETSc)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_STANDARD 17)

add_executable(dam_break dam_break.cpp)
target_link_libraries(dam_break samurai::samurai CLI11::CLI11 ${PETSC_LINK_LIBRARIES} ${MPI_LIBRARIES})
