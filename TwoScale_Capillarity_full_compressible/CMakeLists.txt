cmake_minimum_required(VERSION 3.20)

project(two_scale_capillarity)

set(SAMURAI_WITH_MPI On)
find_package(samurai 0.26.1)
find_package(CLI11)
find_package(nlohmann_json REQUIRED)
find_package(MPI)

# include(FindPkgConfig)
# pkg_check_modules(PETSC PETSc)

set(CMAKE_CXX_STANDARD 17)

add_executable(two_scale_capillarity two_scale_capillarity.cpp)
target_link_libraries(two_scale_capillarity samurai::samurai CLI11::CLI11 nlohmann_json::nlohmann_json)
