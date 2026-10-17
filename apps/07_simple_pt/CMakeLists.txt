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
include_directories("../vkutils"
                    "../volk" 
                    "../LiteMath" 
                    "../LiteScene"
                    "../RayTracing"
                    "../utils"
                    "..")
                    
link_directories("../volk")
#find_package(embree 3.0 REQUIRED)
#uncomment this to detect broken memory problems via gcc sanitizers
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize-address-use-after-scope -fno-omit-frame-pointer -fsanitize=leak -fsanitize=undefined -fsanitize=bounds-strict")

add_compile_definitions(USE_VOLK)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -fPIC -Wno-attributes")

add_subdirectory(../RayTracing ${CMAKE_CURRENT_BINARY_DIR}/RayTracing)

add_executable(testapp main.cpp 
                       test_class.cpp
                       test_class_scene.cpp 
                       ../LiteMath/Image2d.cpp
                       test_class_generated.cpp
                       test_class_generated_init.cpp 
                       test_class_generated_ds.cpp
                       ../vkutils/geom/cmesh.cpp
                       ../vkutils/vk_utils.cpp
                       ../vkutils/vk_copy.cpp
                       ../vkutils/vk_buffers.cpp
                       ../vkutils/vk_images.cpp
                       ../vkutils/vk_context.cpp
                       ../vkutils/vk_alloc_simple.cpp
                       ../vkutils/vk_pipeline.cpp
                       ../vkutils/vk_descriptor_sets.cpp
                       ../vkutils/ray_tracing/vk_rt_utils.cpp
                       ../vkutils/geom/vk_mesh.cpp
                       ../LiteScene/3rd_party/pugixml.cpp
                       ../LiteScene/hydraxml.cpp
                       ../LiteScene/scene_mgr.cpp
                       ../LiteScene/cmesh4.cpp)

set_target_properties(testapp PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
target_link_libraries(testapp LINK_PUBLIC ${Vulkan_LIBRARY} CrossRT OpenMP::OpenMP_CXX volk dl) 
