cmake_minimum_required (VERSION 3.8)
project (test)

# get rid of annoying MSVC warnings.
add_definitions(-D_CRT_SECURE_NO_WARNINGS)
set (CMAKE_CXX_STANDARD 17)
option(USE_ISPC   "Enable ISPC implementation" OFF)
option(USE_CUDA   "Enable CUDA implementation" OFF)
option(USE_HIP    "Enable HIP implementation" OFF)
option(USE_VULKAN "Enable Vulkan implementation" ON)

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
endif()
if(USE_HIP)
  find_package(HIP REQUIRED)
  enable_language(HIP)
  add_compile_definitions(CUDA_MATH)
  add_compile_definitions(USE_HIP)
endif()

include_directories("../../LiteMath" "../../utils")
if(USE_VULKAN)
  include_directories(${Vulkan_INCLUDE_DIR})
  include_directories("../../vkutils" "../../volk")
  link_directories("../../volk")
endif()
if(USE_ISPC)
  add_compile_definitions(USE_ISPC)
endif()

#uncomment this to detect broken memory problems via gcc sanitizers
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize-address-use-after-scope -fno-omit-frame-pointer -fsanitize=leak -fsanitize=undefined -fsanitize=bounds-strict")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -fPIC -Wno-attributes")

set(SOURCE main.cpp test_class.cpp)
set(LIBS OpenMP::OpenMP_CXX)

set(VKGEN_SOURCE test_class_generated.cpp 
                 test_class_generated_ds.cpp
                 test_class_generated_init.cpp )

set(VKUTILS_SOURCE ../../vkutils/vk_utils.cpp
                   ../../vkutils/vk_copy.cpp
                   ../../vkutils/vk_buffers.cpp
                   ../../vkutils/vk_images.cpp
                   ../../vkutils/vk_context.cpp
                   ../../vkutils/vk_alloc_simple.cpp
                   ../../vkutils/vk_pipeline.cpp
                   ../../vkutils/vk_descriptor_sets.cpp)

if(USE_CUDA)
  cuda_add_library(test_class_cuda test_class_generated.cu)
  list(APPEND LIBS test_class_cuda)
endif()

if(USE_HIP)
  add_library(kernel_obj OBJECT test_class_generated.hip)
  list(APPEND SOURCE $<TARGET_OBJECTS:kernel_obj>)
  list(APPEND LIBS hip::device)
endif()

#if(USE_ISPC)
#  list(APPEND SOURCE ${ISPC_SOURCE})
#endif()

if(USE_VULKAN)
  list(APPEND SOURCE ${VKGEN_SOURCE} ${VKUTILS_SOURCE} )
  list(APPEND LIBS volk dl)
endif()                   

add_executable(testapp ${SOURCE})
set_target_properties(testapp PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
target_link_libraries(testapp LINK_PUBLIC ${LIBS})
