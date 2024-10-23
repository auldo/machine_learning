cmake_minimum_required(VERSION 3.29)
project(auldo_machine_learning)

set(CMAKE_CXX_STANDARD 20)

include_directories(include)

add_executable(linear_regression src/playground.cpp
        src/read/tab_reparated.cpp
        src/read/util.cpp)