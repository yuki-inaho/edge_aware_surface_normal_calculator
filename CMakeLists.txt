cmake_minimum_required(VERSION 2.8)

set(PROJ_NAME edge_aware_surface_normal_calculator)
project(${PROJ_NAME})

#set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_CXX_FLAGS "-O0 -std=c++11")
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-O3 -std=c++11")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenMP)
if (OPENMP_FOUND) 
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(PythonLibs 3 REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
MESSAGE("PYTHON_INCLUDE_DIRS : " ${PYTHON_INCLUDE_DIRS})
MESSAGE("PYTHON_LIBRARIES : " ${PYTHON_LIBRARIES})
add_subdirectory(extern/pybind11)

include_directories(
    include
    extern/pybind11/include
)

pybind11_add_module(
    ${PROJ_NAME}
    ${PYTHON_INCLUDE_DIRS}
    src/bindings.cpp 
    src/ndarray_converter.cpp
)

pkg_check_modules(OpenCV REQUIRED opencv)
include_directories(${OpenCV_CFLAGS})
target_link_libraries(${PROJ_NAME} ${OpenCV_LDFLAGS})

pkg_check_modules(eigen3 REQUIRED)
include_directories(${eigen3_INCLUDE_DIRS})
target_link_libraries(${PROJ_NAME} ${eigen3_LIBRARIES})

target_link_libraries(${PROJ_NAME} ${PYTHON_LIBRARIES})
target_link_libraries(${PROJ_NAME} PUBLIC Python3::NumPy)
