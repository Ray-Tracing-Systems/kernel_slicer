cmake_minimum_required (VERSION 3.8)
project (test)

# get rid of annoying MSVC warnings.
add_definitions(-D_CRT_SECURE_NO_WARNINGS)

set (CMAKE_CXX_STANDARD 17)

include_directories(".")
if(WIN32)
  link_directories(${ADDITIONAL_LIBRARY_DIRS})
endif()

#list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}") # to find 'FindFreeImage.cmake' script
#find_package(FreeImage REQUIRED)

find_package(OpenMP)
find_package(Vulkan)

include_directories(${Vulkan_INCLUDE_DIR})
#include_directories(${FREEIMAGE_INCLUDE_DIRS})
include_directories("../vkutils"
                    "../volk" 
                    "../LiteMath" 
                    "../loader_utils"
                    "../geo_utils"
                    "../RayTracing"
                    "../utils")
                    
link_directories("../volk")
link_directories("../RayTracing/lib")

add_compile_definitions(USE_VOLK)

#uncomment this to detect broken memory problems via gcc sanitizers
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize-address-use-after-scope -fno-omit-frame-pointer -fsanitize=leak -fsanitize=undefined -fsanitize=bounds-strict")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -mavx2 -fPIC -Wno-attributes")

add_executable(testapp main.cpp 
                       test_class.cpp
                       test_class2.cpp
                       test_class_scene.cpp
                       #imageutils.cpp 
                       Bitmap.cpp
                       ../LiteMath/texture2d.cpp
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
                       ../loader_utils/pugixml.cpp
                       ../loader_utils/hydraxml.cpp
                       ../loader_utils/scene_mgr.cpp
                       ../RayTracing/EmbreeRT.cpp
                       ../RayTracing/VulkanRTX.cpp)

set_target_properties(testapp PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
target_link_libraries(testapp LINK_PUBLIC ${Vulkan_LIBRARY} OpenMP::OpenMP_CXX volk dl    # ${FREEIMAGE_LIBRARIES}
                      embree3 embree_sse42 embree_avx embree_avx2 lexers simd sys tasking # embree shit
                      ) 
