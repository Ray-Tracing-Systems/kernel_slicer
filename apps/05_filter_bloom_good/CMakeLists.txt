cmake_minimum_required (VERSION 3.8)
project (test)

# get rid of annoying MSVC warnings.
add_definitions(-D_CRT_SECURE_NO_WARNINGS)
set (CMAKE_CXX_STANDARD 17)
option(USE_ISPC "Enable ISPC implementation" OFF)
option(USE_CUDA "Enable CUDA implementation" OFF)
option(USE_HIP    "Enable HIP implementation" OFF)
option(USE_VULKAN "Enable Vulkan implementation" ON)
option(USE_WGPU   "Enable WGPU code" OFF)

include_directories(".")
if(WIN32)
  link_directories(${ADDITIONAL_LIBRARY_DIRS})
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS ${CMAKE_CURRENT_SOURCE_DIR})

find_package(OpenMP)
if(USE_VULKAN)
  find_package(Vulkan)
  add_compile_definitions(USE_VOLK)
  add_compile_definitions(USE_VULKAN)
endif()
if(USE_CUDA)
  find_package(CUDA REQUIRED)
  add_compile_definitions(CUDA_MATH)
  add_compile_definitions(USE_CUDA)
elseif(USE_HIP)
  find_package(HIP REQUIRED)
  enable_language(HIP)
  add_compile_definitions(CUDA_MATH)
  add_compile_definitions(USE_HIP)
endif()
find_package(FreeImage REQUIRED)
include_directories(${FREEIMAGE_INCLUDE_DIRS})
include_directories("../LiteMath" "../utils")
if(USE_VULKAN)
  include_directories(${Vulkan_INCLUDE_DIR})
  include_directories("../vkutils" "../volk")
  link_directories("../volk")
endif()
if(USE_ISPC)
  add_compile_definitions(USE_ISPC)
endif()

#include_directories("/home/frol/PROG/kernel_slicer/apps/05_filter_bloom_good/cmake-build-debug/_deps/webgpu-backend-wgpu-src/include")
#include_directories("/home/frol/PROG/kernel_slicer/apps/05_filter_bloom_good/cmake-build-debug/_deps/dawn-src/include")
#include_directories("/home/frol/PROG/kernel_slicer/apps/05_filter_bloom_good/cmake-build-debug")

if(USE_WGPU)
  message(STATUS "WEBGPU_BACKEND: ${WEBGPU_BACKEND}")
  add_compile_definitions(USE_WEBGPU)
  include_directories("../wkutils")
  get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
  set(EXTERNAL_DIR ${PARENT_DIR}/external)
  add_subdirectory(${PARENT_DIR}/webgpu ${CMAKE_CURRENT_BINARY_DIR}/webgpu)
  include_directories(${PARENT_DIR}/webgpu/include)

  if(WEBGPU_BACKEND STREQUAL "WGPU")
    add_compile_definitions(-DUSE_WGPU)
    message(STATUS "Using WGPU backend")
  else()
    add_compile_definitions(-DUSE_DAWN)
    message(STATUS "Using DAWN backend (or default)")
  endif()
endif()

#uncomment this to detect broken memory problems via gcc sanitizers
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize-address-use-after-scope -fno-omit-frame-pointer -fsanitize=leak -fsanitize=undefined -fsanitize=bounds-strict")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -Wno-attributes")

set(SOURCE main.cpp 
           test_class.cpp
           ../LiteMath/Image2d.cpp
           imageutils.cpp)

set(ISPC_SOURCE test_class_ispc.cpp
                test_class_kernels.o
                 ../LiteMathAux/ispc_tasksys.cpp)

set(VKGEN_SOURCE test_class_gpu.cpp 
                 test_class_gpu_ds.cpp
                 test_class_gpu_init.cpp)

set(VKUTILS_SOURCE ../vkutils/vk_utils.cpp
                   ../vkutils/vk_copy.cpp
                   ../vkutils/vk_buffers.cpp
                   ../vkutils/vk_images.cpp
                   ../vkutils/vk_context.cpp
                   ../vkutils/vk_alloc_simple.cpp
                   ../vkutils/vk_pipeline.cpp
                   ../vkutils/vk_descriptor_sets.cpp)                

set(WKGEN_SOURCE test_class_wgpu.cpp)    
set(WKUTILS_SOURCE ../wkutils/wk_context.cpp) 

set(LIBS ${FREEIMAGE_LIBRARIES} OpenMP::OpenMP_CXX dl)

if(USE_CUDA)
  cuda_add_library(test_class_cuda test_class_cuda.cu)
  list(APPEND LIBS test_class_cuda)
elseif(USE_HIP)
  add_library(kernel_obj OBJECT test_class_cuda.hip)
  list(APPEND SOURCE $<TARGET_OBJECTS:kernel_obj>)
  list(APPEND LIBS hip::device)
endif()
if(USE_ISPC)
  list(APPEND SOURCE ${ISPC_SOURCE})
endif()
if(USE_VULKAN)
  list(APPEND SOURCE ${VKGEN_SOURCE} ${VKUTILS_SOURCE} )
  list(APPEND LIBS volk dl)
endif()
if(USE_WGPU)
  list(APPEND LIBS webgpu)
  list(APPEND SOURCE ${WKGEN_SOURCE} ${WKUTILS_SOURCE})
endif()

add_executable(testapp ${SOURCE})
set_target_properties(testapp PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
target_link_libraries(testapp LINK_PUBLIC ${LIBS})
if(USE_WGPU)
  target_copy_webgpu_binaries(testapp)
endif()