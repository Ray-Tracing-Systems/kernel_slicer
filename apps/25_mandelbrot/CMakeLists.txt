cmake_minimum_required (VERSION 3.8)
project (test)

# get rid of annoying MSVC warnings.
add_definitions(-D_CRT_SECURE_NO_WARNINGS)
set (CMAKE_CXX_STANDARD 17)

option(USE_VULKAN "Enable Vulkan code" ON)
option(USE_ISPC   "Enable ISPC code" OFF)
option(USE_WGPU   "Enable WGPU code" OFF)

include_directories(".")
if(WIN32)
  link_directories(${ADDITIONAL_LIBRARY_DIRS})
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS ${CMAKE_CURRENT_SOURCE_DIR})

find_package(OpenMP)
include_directories("../LiteMath" "../utils")

if(USE_VULKAN)
  find_package(Vulkan)
  include_directories(${Vulkan_INCLUDE_DIR})
  include_directories("../vkutils" "../volk")
  link_directories("../volk")
  add_compile_definitions(USE_VOLK)
  add_compile_definitions(USE_VULKAN)
endif()

if(USE_ISPC)
  add_compile_definitions(USE_ISPC)
endif()

if(USE_WGPU)
  add_compile_definitions(USE_WGPU)
  include_directories("../wkutils")
  get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
  add_subdirectory(${PARENT_DIR}/webgpu ${CMAKE_CURRENT_BINARY_DIR}/webgpu)
  ##include_directories(${PARENT_DIR}/webgpu/emdawnwebgpu/include)
  #include_directories(${PARENT_DIR}/webgpu-native/include)
  #link_directories(${PARENT_DIR}/webgpu-native/lib)
endif()

#uncomment this to detect broken memory problems via gcc sanitizers
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize-address-use-after-scope -fno-omit-frame-pointer -fsanitize=leak -fsanitize=undefined -fsanitize=bounds-strict")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wno-attributes")

set(MAIN_SOURCE main.cpp 
                mandelbrot.cpp
                ../LiteMath/Image2d.cpp)

set(ISPC_SOURCE mandelbrot_ispc.cpp
                mandelbrot_kernels.o
                ../LiteMathAux/ispc_tasksys.cpp)

set(VKGEN_SOURCE mandelbrot_generated.cpp
                 mandelbrot_generated_init.cpp
                 mandelbrot_generated_ds.cpp)

set(VKUTILS_SOURCE ../vkutils/vk_utils.cpp
                   ../vkutils/vk_copy.cpp
                   ../vkutils/vk_buffers.cpp
                   ../vkutils/vk_images.cpp
                   ../vkutils/vk_context.cpp
                   ../vkutils/vk_alloc_simple.cpp
                   ../vkutils/vk_pipeline.cpp
                   ../vkutils/vk_descriptor_sets.cpp)    


set(WKGEN_SOURCE mandelbrot_wgpu.cpp)    
set(WKUTILS_SOURCE ../wkutils/wk_context.cpp) # ../wkutils/wk_utils.cpp

set(SOURCE ${MAIN_SOURCE})
set(LINK ${FREEIMAGE_LIBRARIES})

if(USE_VULKAN)
  list(APPEND SOURCE ${VKGEN_SOURCE} ${VKUTILS_SOURCE})
  list(APPEND LINK ${Vulkan_LIBRARY} volk)
endif()
if(USE_ISPC)
  list(APPEND SOURCE ${ISPC_SOURCE})
endif()
if(USE_WGPU)
  list(APPEND LINK webgpu)
  list(APPEND SOURCE ${WKGEN_SOURCE} ${WKUTILS_SOURCE})
endif()


add_executable(testapp ${SOURCE})
set_target_properties(testapp PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
target_link_libraries(testapp LINK_PUBLIC ${LINK} OpenMP::OpenMP_CXX dl)

if(USE_WGPU)
  #target_link_libraries(testapp PRIVATE wgpu_native)
  target_copy_webgpu_binaries(testapp)
endif()
