cmake_minimum_required (VERSION 3.8)
project (textured_cube)

# CPU only build until kslicer supports LiteRS: cmake -DUSE_VULKAN=OFF ..
option(USE_VULKAN "Enable Vulkan implementation" ON)
option(USE_ISPC   "Enable ISPC implementation (instead of Vulkan), run z_build_ispc.sh first" OFF)
set(APPS "${CMAKE_CURRENT_SOURCE_DIR}/..")

add_definitions(-D_CRT_SECURE_NO_WARNINGS)
set(CMAKE_CXX_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP REQUIRED)
include_directories("." "${APPS}/LiteMath" "${APPS}/LiteMathAux" "${APPS}/utils")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wno-attributes")

set(MAIN_SOURCE main.cpp textured_cube.cpp ${APPS}/LiteMath/Image2d.cpp)
set(LIBS OpenMP::OpenMP_CXX)

if(USE_ISPC)
  add_compile_definitions(USE_ISPC)
  list(APPEND MAIN_SOURCE textured_cube_ispc.cpp textured_cube_kernels.o ${APPS}/LiteMathAux/ispc_tasksys.cpp)
elseif(USE_VULKAN)
  find_package(Vulkan)
  add_compile_definitions(USE_VOLK USE_VULKAN)
  include_directories(${Vulkan_INCLUDE_DIR} "${APPS}/vkutils" "${APPS}/volk")
  link_directories("${APPS}/volk")
  list(APPEND MAIN_SOURCE textured_cube_generated.cpp textured_cube_generated_ds.cpp textured_cube_generated_init.cpp
                          ${APPS}/vkutils/vk_utils.cpp ${APPS}/vkutils/vk_copy.cpp ${APPS}/vkutils/vk_buffers.cpp
                          ${APPS}/vkutils/vk_images.cpp ${APPS}/vkutils/vk_context.cpp ${APPS}/vkutils/vk_alloc_simple.cpp
                          ${APPS}/vkutils/vk_pipeline.cpp ${APPS}/vkutils/vk_descriptor_sets.cpp)
  list(APPEND LIBS ${Vulkan_LIBRARY} volk dl)
endif()

add_executable(testapp ${MAIN_SOURCE})
set_target_properties(testapp PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
target_link_libraries(testapp LINK_PUBLIC ${LIBS})
