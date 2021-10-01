cmake_minimum_required (VERSION 3.8)
project (c_image_processing)

set(COMPILER_WARNINGS "-Wall")
list(APPEND COMPILER_WARNINGS "-Werror")
list(APPEND COMPILER_WARNINGS "-Wextra -Wshadow -Wnull-dereference -Wlogical-op")
list(APPEND COMPILER_WARNINGS "-Wmisleading-indentation -Wduplicated-cond")
list(APPEND COMPILER_WARNINGS "-fsanitize=undefined")

list(JOIN COMPILER_WARNINGS " " LIST_OF_COMPILER_FLAGS)

set(CMAKE_C_FLAGS ${LIST_OF_COMPILER_FLAGS})

#find_package(OpenMP)
#find_package(Vulkan)
#include_directories(${Vulkan_INCLUDE_DIR})
#include_directories("../vkutils"
#                    "../volk" 
#                    "../LiteMath" 
#                    "../loader_utils"
#                    "../geo_utils")
                    
#link_directories("../volk")

add_executable(c_image_proc main.c image_proc.c)

target_link_libraries(c_image_proc LINK_PUBLIC m) # ${Vulkan_LIBRARY} volk dl
