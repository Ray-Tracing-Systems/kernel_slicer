cmake_minimum_required (VERSION 3.8)
project (test)

# get rid of annoying MSVC warnings.
add_definitions(-D_CRT_SECURE_NO_WARNINGS)

set (CMAKE_CXX_STANDARD 17)

include_directories(".")
if(WIN32)
  link_directories(${ADDITIONAL_LIBRARY_DIRS})
endif()

find_package(OpenMP)
find_package(Vulkan)
include_directories(${Vulkan_INCLUDE_DIR})
include_directories("../../vkutils"
                    "../../volk" 
                    "../../LiteMath" 
                    "../../LiteScene"
                    "../../RayTracing"
                    "../../utils"
                    "../..")
                    
link_directories("../../volk")
link_directories("../../RayTracing/lib")

add_compile_definitions(USE_VOLK)

if(CMAKE_SYSTEM_NAME STREQUAL Windows)
  link_directories("${CMAKE_SOURCE_DIR}/../../RayTracing/lib_win64")
  set(EMBREE_LIBS ${CMAKE_CURRENT_SOURCE_DIR}/../../RayTracing/lib_win64/embree3.lib)
elseif(CMAKE_SYSTEM_NAME STREQUAL Linux)
  find_package(Threads REQUIRED)
  set(EMBREE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../RayTracing/lib)
  link_directories(${EMBREE_DIR})
  set(EMBREE_LIBS ${EMBREE_DIR}/libembree3.so 
                  ${EMBREE_DIR}/libtbb.so)
endif()

#uncomment this to detect broken memory problems via gcc sanitizers
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize-address-use-after-scope -fno-omit-frame-pointer -fsanitize=leak -fsanitize=undefined -fsanitize=bounds-strict")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -fPIC -Wno-attributes")

add_executable(testapp main.cpp 
                       test_class.cpp
                       ../../LiteMath/Image2d.cpp
                       test_class_generated.cpp
                       test_class_generated_init.cpp
                       test_class_generated_ds.cpp
                       ../../vkutils/geom/cmesh.cpp
                       ../../vkutils/vk_utils.cpp
                       ../../vkutils/vk_copy.cpp
                       ../../vkutils/vk_images.cpp
                       ../../vkutils/vk_buffers.cpp
                       ../../vkutils/vk_images.cpp
                       ../../vkutils/vk_context.cpp
                       ../../vkutils/vk_alloc_simple.cpp
                       ../../vkutils/vk_pipeline.cpp
                       ../../vkutils/vk_descriptor_sets.cpp
                       ../../vkutils/ray_tracing/vk_rt_utils.cpp
                       ../../vkutils/geom/vk_mesh.cpp
                       ../../LiteScene/pugixml.cpp
                       ../../LiteScene/hydraxml.cpp
                       ../../LiteScene/scene_mgr.cpp
                       ../../LiteScene/cmesh4.cpp
                       ../../RayTracing/EmbreeRT.cpp
                       ../../RayTracing/VulkanRTX.cpp)

set_target_properties(testapp PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
target_link_libraries(testapp LINK_PUBLIC ${Vulkan_LIBRARY} ${EMBREE_LIBS} OpenMP::OpenMP_CXX volk dl) 
