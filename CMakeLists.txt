cmake_minimum_required(VERSION 2.8)

set(PROJ_NAME edge_aware_surface_normal_calculator)
project(${PROJ_NAME})
add_executable(
    ${PROJ_NAME} 
    src/main.cpp 
    src/toml_reader.cpp
)



include_directories(include)
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(OpenMP)

#set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_CXX_FLAGS "-O0 -std=c++11")
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-O3 -std=c++11")
if (OPENMP_FOUND) 
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

pkg_check_modules(OpenCV REQUIRED opencv)
include_directories(${OpenCV_CFLAGS})
target_link_libraries(${PROJ_NAME} ${OpenCV_LDFLAGS})

pkg_check_modules(eigen3 REQUIRED)
include_directories(${eigen3_INCLUDE_DIRS})
target_link_libraries(${PROJ_NAME} ${eigen3_LIBRARIES})

target_link_libraries(${PROJ_NAME} stdc++fs)
#target_link_libraries (${PROJ_NAME} profiler)

