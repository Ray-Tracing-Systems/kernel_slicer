cmake_minimum_required (VERSION 3.8)
project (test)

add_definitions(-D_CRT_SECURE_NO_WARNINGS)
set (CMAKE_CXX_STANDARD 17)
option(USE_VULKAN "Enable Vulkan implementation" ON)

include_directories(".")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

find_package(OpenMP)
include_directories("../LiteMath" "../utils")

if(USE_VULKAN)
  find_package(Vulkan)
  add_compile_definitions(USE_VOLK)
  add_compile_definitions(USE_VULKAN)
  include_directories(${Vulkan_INCLUDE_DIR})
  include_directories("../vkutils" "../volk")
  link_directories("../volk")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -fPIC -Wno-attributes")

set(MAIN_SOURCE main.cpp
                metaballs.cpp
                ../LiteMath/Image2d.cpp)

set(VKGEN_SOURCE metaballs_generated.cpp
                 metaballs_generated_ds.cpp
                 metaballs_generated_init.cpp)

set(VKUTILS_SOURCE ../vkutils/vk_utils.cpp
                   ../vkutils/vk_copy.cpp
                   ../vkutils/vk_buffers.cpp
                   ../vkutils/vk_images.cpp
                   ../vkutils/vk_context.cpp
                   ../vkutils/vk_alloc_simple.cpp
                   ../vkutils/vk_pipeline.cpp
                   ../vkutils/vk_descriptor_sets.cpp)

set(LIBS OpenMP::OpenMP_CXX)
if(USE_VULKAN)
  list(APPEND MAIN_SOURCE ${VKGEN_SOURCE} ${VKUTILS_SOURCE})
  list(APPEND LIBS ${Vulkan_LIBRARY} volk dl)
endif()

add_executable(testapp ${MAIN_SOURCE})
set_target_properties(testapp PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
target_link_libraries(testapp LINK_PUBLIC ${LIBS})
