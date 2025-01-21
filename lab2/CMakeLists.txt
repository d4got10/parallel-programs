cmake_minimum_required(VERSION 3.24)
project(lab2)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")

add_executable(lab2 main.cpp)
