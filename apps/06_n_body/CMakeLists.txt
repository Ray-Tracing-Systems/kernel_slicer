cmake_minimum_required (VERSION 3.8)
project (test)

# get rid of annoying MSVC warnings.
add_definitions(-D_CRT_SECURE_NO_WARNINGS)
set (CMAKE_CXX_STANDARD 17)
option(USE_ISPC "Enable ISPC code" OFF)

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
include_directories("../volk" "../LiteMath" "../utils")
link_directories("../volk")

include_directories(../vkutils)

find_package(glfw3 REQUIRED)
include_directories(${GLFW_INCLUDE_DIRS})

add_compile_definitions(USE_VOLK)
if(USE_ISPC)
  add_compile_definitions(USE_ISPC)
endif()

#uncomment this to detect broken memory problems via gcc sanitizers
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize-address-use-after-scope -fno-omit-frame-pointer -fsanitize=leak -fsanitize=undefined -fsanitize=bounds-strict")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")


set(MAIN_SOURCE main.cpp 
                test_class.cpp)

set(ISPC_SOURCE test_class_ispc.cpp
                test_class_kernels.o
                ../LiteMathAux/ispc_tasksys.cpp)

set(VKGEN_SOURCE test_class_generated.cpp 
                 test_class_generated_ds.cpp
                 test_class_generated_init.cpp
                 test_class_gpu.cpp)
       
set(VKUTILS_NEW_SRC ../vkutils/vk_copy.cpp
                    ../vkutils/vk_utils.cpp
                    ../vkutils/vk_buffers.cpp
                    ../vkutils/vk_descriptor_sets.cpp
                    ../vkutils/vk_images.cpp
                    ../vkutils/vk_swapchain.cpp
                    ../vkutils/vk_pipeline.cpp
                    ../vkutils/geom/vk_mesh.cpp
                    ../vkutils/geom/cmesh.cpp)

set(NBODY_VIS render/create_render.cpp
              render/points_render.cpp
              render/points_render_compute.cpp
              generated_userfix.cpp
              glfw_window.cpp)

if(USE_ISPC)
add_executable(testapp ${MAIN_SOURCE}
                       ${VKGEN_SOURCE}
                       ${NBODY_VIS}
                       ${VKUTILS_NEW_SRC}
                       ${ISPC_SOURCE})
else()
add_executable(testapp ${MAIN_SOURCE}
                       ${VKGEN_SOURCE}
                       ${NBODY_VIS}
                       ${VKUTILS_NEW_SRC})
endif()

set_target_properties(testapp PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
target_link_libraries(testapp LINK_PUBLIC ${Vulkan_LIBRARY} OpenMP::OpenMP_CXX volk dl glfw)
