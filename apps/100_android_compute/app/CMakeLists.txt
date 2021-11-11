cmake_minimum_required(VERSION 3.6)

# build native_app_glue as a static lib
set(APP_GLUE_DIR ${ANDROID_NDK}/sources/android/native_app_glue)
include_directories(${APP_GLUE_DIR})
add_library(app-glue STATIC ${APP_GLUE_DIR}/android_native_app_glue.c)
add_library(cpufeatures STATIC ${ANDROID_NDK}/sources/android/cpufeatures/cpu-features.c)

# build vulkan app
set(VKUTILS_DIR    ../../vkutils)
set(MATH_DIR       ../../LiteMath)

set(ALL_APPS_DIR    ../../)
set(MAIN_APP_DIR_1  ../../04_array_summ)
set(MAIN_APP_DIR_2  ../../06_n_body)

set(SRC_DIR src/main/jni)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -DVK_USE_PLATFORM_ANDROID_KHR")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -u ANativeActivity_onCreate")

set(VOLK_STATIC_DEFINES VK_USE_PLATFORM_ANDROID_KHR)
add_subdirectory(../../volk ../../volk)


include_directories(../../volk)
link_directories(../../volk)
include_directories(${APP_GLUE_DIR})
include_directories(${VKUTILS_DIR})
include_directories(${MATH_DIR})
include_directories(${SRC_DIR})
include_directories(${ALL_APPS_DIR})

include_directories(${MAIN_APP_DIR_1})
include_directories(${MAIN_APP_DIR_2})

set(VKUTILS_NEW_SRC ${VKUTILS_DIR}/vk_copy.cpp
                    ${VKUTILS_DIR}/vk_utils.cpp
                    ${VKUTILS_DIR}/vk_buffers.cpp
                    ${VKUTILS_DIR}/vk_descriptor_sets.cpp
                    ${VKUTILS_DIR}/vk_images.cpp
                    ${VKUTILS_DIR}/vk_swapchain.cpp
                    ${VKUTILS_DIR}/vk_pipeline.cpp
                    ${VKUTILS_DIR}/geom/vk_mesh.cpp
                    ${VKUTILS_DIR}/geom/cmesh.cpp)


set(APP_SRC_1 ${MAIN_APP_DIR_1}/test_class.cpp
              ${MAIN_APP_DIR_1}/test_class_gpu.cpp
              ${MAIN_APP_DIR_1}/test_class_generated.cpp
              ${MAIN_APP_DIR_1}/test_class_generated_ds.cpp
              ${MAIN_APP_DIR_1}/test_class_generated_init.cpp)

set(APP_SRC_2 ${MAIN_APP_DIR_2}/test_class.cpp
              ${MAIN_APP_DIR_2}/test_class_gpu.cpp
              ${MAIN_APP_DIR_2}/test_class_generated.cpp
              ${MAIN_APP_DIR_2}/test_class_generated_ds.cpp
              ${MAIN_APP_DIR_2}/test_class_generated_init.cpp)

add_library(VkCompute SHARED
                ${VKUTILS_NEW_SRC}
                ${APP_SRC_1}
                ${APP_SRC_2}
                ${SRC_DIR}/AndroidMain.cpp
                ${SRC_DIR}/lodepng.cpp)

#MESSAGE(WARNING "Android ABI is ${ANDROID_ABI}")

if (${ANDROID_ABI} STREQUAL "arm64-v8a")
    set_target_properties(VkCompute PROPERTIES COMPILE_FLAGS -mfpu=neon)
    add_definitions("-DHAVE_NEON=1")
elseif (${ANDROID_ABI} STREQUAL "x86_64")
    set_target_properties(VkCompute PROPERTIES COMPILE_FLAGS -mssse3)
    set_target_properties(VkCompute PROPERTIES COMPILE_FLAGS -mavx)
    add_definitions(-DHAVE_NEON_X86=1 -DHAVE_NEON=1)
endif()

target_link_libraries(VkCompute app-glue cpufeatures log android volk)

