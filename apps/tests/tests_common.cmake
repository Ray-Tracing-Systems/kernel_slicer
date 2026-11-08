
include_guard(GLOBAL)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP REQUIRED)

add_compile_definitions(
    _CRT_SECURE_NO_WARNINGS
    USE_STB_IMAGE
    USE_VULKAN
    USE_VOLK
)
add_compile_options(-Wno-attributes)

add_subdirectory(../../volk volk)

include_directories(
    .
    ../..
    ../../utils
    ../../LiteMath
    ../../LiteMath/external
    ../../vkutils
)

link_libraries(volk::volk)

add_subdirectory(../../RayTracing RayTracing)

link_libraries(CrossRT OpenMP::OpenMP_CXX)

file(GLOB sources CONFIGURE_DEPENDS LIST_DIRECTORIES OFF *.cpp)

add_executable(testapp
    ${sources}

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
    ../../vkutils/external/samplers_vk.cpp

    ../../LiteScene/hydraxml.cpp
    ../../LiteScene/scene_mgr.cpp
    ../../LiteScene/cmesh4.cpp
    ../../LiteScene/3rd_party/pugixml.cpp
    ../../LiteScene/3rd_party/tinyexr/miniz.c

    ../../LiteMath/Image2d.cpp
)
set_target_properties(testapp PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
