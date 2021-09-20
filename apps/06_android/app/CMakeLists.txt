cmake_minimum_required(VERSION 3.6)

# build native_app_glue as a static lib
set(APP_GLUE_DIR ${ANDROID_NDK}/sources/android/native_app_glue)
include_directories(${APP_GLUE_DIR})
add_library(app-glue STATIC ${APP_GLUE_DIR}/android_native_app_glue.c)
add_library(cpufeatures STATIC ${ANDROID_NDK}/sources/android/cpufeatures/cpu-features.c)

# build vulkan app
set(VKUTILS_DIR    ../../vkutils)
set(MAIN_APP_DIR   ../../06_n_body)
set(MATH_DIR       ../../LiteMath)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -DVK_USE_PLATFORM_ANDROID_KHR")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -u ANativeActivity_onCreate")

set(VOLK_STATIC_DEFINES VK_USE_PLATFORM_ANDROID_KHR)
add_subdirectory(../../volk ../../volk)

include_directories(../../volk)
link_directories(../../volk)
include_directories(${APP_GLUE_DIR})
include_directories(${VKUTILS_DIR})
include_directories(${MAIN_APP_DIR})
include_directories(${MATH_DIR})

set(VK_BASE_SAMPLE_DIR ../../vk_base)
set(SAMPLE_SRC_DIR ../../loadSceneFromXML)

set(SRC_DIR src/main/jni)

set(VKUTILS_NEW_SRC ${VKUTILS_DIR}/vk_copy.cpp
                    ${VKUTILS_DIR}/vk_utils.cpp
                    ${VKUTILS_DIR}/vk_buffers.cpp
                    ${VKUTILS_DIR}/vk_descriptor_sets.cpp
                    ${VKUTILS_DIR}/vk_images.cpp
                    ${VKUTILS_DIR}/vk_swapchain.cpp
                    ${VKUTILS_DIR}/vk_pipeline.cpp
                    ${VKUTILS_DIR}/geom/vk_mesh.cpp
                    ${VKUTILS_DIR}/geom/cmesh.cpp)

set(NBODY_VIS ${MAIN_APP_DIR}/render/create_render.cpp
              ${MAIN_APP_DIR}/render/points_render.cpp
              ${MAIN_APP_DIR}/render/points_render_compute.cpp
              ${MAIN_APP_DIR}/generated_userfix.cpp)

set(APP_SRC ${MAIN_APP_DIR}/test_class.cpp
            ${MAIN_APP_DIR}/test_class_gpu.cpp
            ${MAIN_APP_DIR}/test_class_generated.cpp
            ${MAIN_APP_DIR}/test_class_generated_ds.cpp
            ${MAIN_APP_DIR}/test_class_generated_init.cpp)

add_library(NBodySample SHARED
                ${VKUTILS_NEW_SRC}
                ${NBODY_VIS}
                ${APP_SRC}
                ${SRC_DIR}/AndroidMain.cpp
                ${SRC_DIR}/android_input.cpp)

#MESSAGE(WARNING "Android ABI is ${ANDROID_ABI}")

if (${ANDROID_ABI} STREQUAL "arm64-v8a")
    set_target_properties(NBodySample PROPERTIES COMPILE_FLAGS -mfpu=neon)
    add_definitions("-DHAVE_NEON=1")
elseif (${ANDROID_ABI} STREQUAL "x86_64")
    set_target_properties(NBodySample PROPERTIES COMPILE_FLAGS -mssse3)
    set_target_properties(NBodySample PROPERTIES COMPILE_FLAGS -mavx)
    add_definitions(-DHAVE_NEON_X86=1 -DHAVE_NEON=1)
endif()

target_link_libraries(NBodySample app-glue cpufeatures log android volk)

