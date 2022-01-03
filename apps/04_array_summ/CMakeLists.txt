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

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS ${CMAKE_CURRENT_SOURCE_DIR})

find_package(OpenMP)
find_package(Vulkan)

include_directories(${Vulkan_INCLUDE_DIR})
include_directories("../vkutils" "../volk" "../LiteMath" "../utils")
link_directories("../volk")

add_compile_definitions(USE_VOLK)

#uncomment this to detect broken memory problems via gcc sanitizers
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize-address-use-after-scope -fno-omit-frame-pointer -fsanitize=leak -fsanitize=undefined -fsanitize=bounds-strict")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -fPIC -Wno-attributes")

add_executable(testapp main.cpp 
                       test_class.cpp
                       test_class_generated.cpp 
                       test_class_generated_ds.cpp
                       test_class_generated_init.cpp 
                       ../vkutils/vk_utils.cpp
                       ../vkutils/vk_copy.cpp
                       ../vkutils/vk_buffers.cpp
                       ../vkutils/vk_images.cpp
                       ../vkutils/vk_context.cpp
                       ../vkutils/vk_alloc_simple.cpp
                       ../vkutils/vk_pipeline.cpp
                       ../vkutils/vk_descriptor_sets.cpp)

set_target_properties(testapp PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
target_link_libraries(testapp LINK_PUBLIC ${Vulkan_LIBRARY} OpenMP::OpenMP_CXX volk dl)
