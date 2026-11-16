cmake_minimum_required (VERSION 3.8)
project (raster_test)

# unit tests of LiteRS CPU rasterizer (../raster.h) and of tiled LiteImage::Image2D
set (CMAKE_CXX_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP)
include_directories(".." "../../LiteMath")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-attributes")

add_executable(raster_test main.cpp ../../LiteMath/Image2d.cpp)
if(OpenMP_CXX_FOUND)
  target_link_libraries(raster_test LINK_PUBLIC OpenMP::OpenMP_CXX)
endif()
