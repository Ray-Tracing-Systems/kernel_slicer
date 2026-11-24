cmake_minimum_required (VERSION 3.8)
project (puddles)

option(USE_VULKAN "Enable Vulkan implementation" ON)
option(USE_ISPC   "Enable ISPC implementation (instead of Vulkan), run z_build_ispc.sh first" OFF)
set(APPS "${CMAKE_CURRENT_SOURCE_DIR}/..")

add_definitions(-D_CRT_SECURE_NO_WARNINGS)
set(CMAKE_CXX_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP REQUIRED)
include_directories("." "${APPS}/LiteMath" "${APPS}/LiteMathAux" "${APPS}/LiteScene" "${APPS}/RayTracing" "${APPS}/utils" "${APPS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wno-attributes")

set(MAIN_SOURCE main.cpp puddles.cpp puddles_scene.cpp ${APPS}/LiteMath/Image2d.cpp
                ${APPS}/LiteScene/hydraxml.cpp ${APPS}/LiteScene/cmesh4.cpp ${APPS}/LiteScene/3rd_party/pugixml.cpp)
set(LIBS CrossRT OpenMP::OpenMP_CXX)

if(USE_ISPC)
  add_compile_definitions(USE_ISPC)
  list(APPEND MAIN_SOURCE puddles_ispc.cpp puddles_kernels.o ${APPS}/LiteMathAux/ispc_tasksys.cpp)
  set(USE_VULKAN OFF) # CrossRT: Embree only
elseif(USE_VULKAN)
  find_package(Vulkan)
  add_compile_definitions(USE_VOLK USE_VULKAN)
  include_directories(${Vulkan_INCLUDE_DIR} "${APPS}/vkutils" "${APPS}/volk")
  link_directories("${APPS}/volk")
  list(APPEND MAIN_SOURCE puddles_generated.cpp puddles_generated_ds.cpp puddles_generated_init.cpp
                          ${APPS}/vkutils/vk_utils.cpp ${APPS}/vkutils/vk_copy.cpp ${APPS}/vkutils/vk_buffers.cpp
                          ${APPS}/vkutils/vk_images.cpp ${APPS}/vkutils/vk_context.cpp ${APPS}/vkutils/vk_alloc_simple.cpp
                          ${APPS}/vkutils/vk_pipeline.cpp ${APPS}/vkutils/vk_descriptor_sets.cpp
                          ${APPS}/vkutils/ray_tracing/vk_rt_utils.cpp ${APPS}/vkutils/geom/vk_mesh.cpp ${APPS}/vkutils/geom/cmesh.cpp
                          ${APPS}/LiteScene/scene_mgr.cpp)
  list(APPEND LIBS ${Vulkan_LIBRARY} volk dl)
endif()

add_subdirectory(${APPS}/RayTracing ${CMAKE_CURRENT_BINARY_DIR}/RayTracing) # after include directories of Vulkan

add_executable(testapp ${MAIN_SOURCE})
set_target_properties(testapp PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
target_link_libraries(testapp LINK_PUBLIC ${LIBS})
