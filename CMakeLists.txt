cmake_minimum_required(VERSION 2.8)

set(PROJ_NAME edge_aware_surface_normal_calculator)
project(${PROJ_NAME})
add_executable(
    ${PROJ_NAME} 
    src/main.cpp 
    src/toml_reader.cpp
)

#set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_CXX_FLAGS "-O0 -std=c++11 -lprofiler")
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-O3 -std=c++11")

include_directories(include)
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)

pkg_check_modules(OpenCV REQUIRED opencv)
include_directories(${OpenCV_CFLAGS})
target_link_libraries(${PROJ_NAME} ${OpenCV_LDFLAGS})
target_link_libraries(${PROJ_NAME} stdc++fs)
#target_link_libraries (${PROJ_NAME} profiler)

