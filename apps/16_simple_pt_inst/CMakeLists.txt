cmake_minimum_required (VERSION 3.8)
project (test)

# get rid of annoying MSVC warnings.
add_definitions(-D_CRT_SECURE_NO_WARNINGS)

set (CMAKE_CXX_STANDARD 17)

include_directories(".")
if(WIN32)
  link_directories(${ADDITIONAL_LIBRARY_DIRS})
endif()

if (CMAKE_SYSTEM_PROCESSOR MATCHES "arm" )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
endif()

find_package(OpenMP)
find_package(Vulkan)
include_directories(${Vulkan_INCLUDE_DIR})
include_directories("../vkfw" 
                    "../volk" 
                    "../LiteMath" 
                    "../loader_utils"
                    "../geo_utils"
                    "../RayTracing")
                    
link_directories("../volk")
link_directories("../RayTracing/lib")

#uncomment this to detect broken memory problems via gcc sanitizers
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize-address-use-after-scope -fno-omit-frame-pointer -fsanitize=leak -fsanitize=undefined -fsanitize=bounds-strict")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2 -fopenmp -fPIC")

add_executable(testapp main.cpp 
                       test_class.cpp
                       test_class_scene.cpp 
                       Bitmap.cpp
                       #test_class_gpu.cpp
                       #test_class_generated.cpp
                       #test_class_generated_init.cpp 
                       #test_class_generated_ds.cpp 
                       ../geo_utils/cmesh.cpp
                       ../vkfw/vk_utils.cpp
                       ../vkfw/vk_copy.cpp
                       ../vkfw/vk_buffer.cpp
                       ../vkfw/vk_compute_pipeline.cpp
                       ../vkfw/vk_program.cpp
                       ../vkfw/vk_rt_utils.cpp
                       ../vkfw/rt_funcs.cpp
                       ../vkfw/vk_mesh.cpp
                       ../loader_utils/pugixml.cpp
                       ../loader_utils/hydraxml.cpp
                       ../loader_utils/scene_mgr.cpp
                       ../RayTracing/EmbreeRT.cpp)

set_target_properties(testapp PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
target_link_libraries(testapp LINK_PUBLIC ${Vulkan_LIBRARY} OpenMP::OpenMP_CXX volk dl 
                      embree3 embree_sse42 embree_avx embree_avx2 embree_avx512 lexers simd sys tasking # embree shit
                      ) 
