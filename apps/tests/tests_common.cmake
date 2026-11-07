
include_guard(GLOBAL)

set(CMAKE_CXX_STANDARD 17)
add_compile_definitions(_CRT_SECURE_NO_WARNINGS)

add_subdirectory(../../volk volk)

add_library(vkutils STATIC
    ../../vkutils/vk_utils.cpp
    ../../vkutils/vk_copy.cpp
    ../../vkutils/vk_buffers.cpp
    ../../vkutils/vk_images.cpp
    ../../vkutils/vk_context.cpp
    ../../vkutils/vk_alloc_simple.cpp
    ../../vkutils/vk_pipeline.cpp
    ../../vkutils/vk_descriptor_sets.cpp
)
target_include_directories(vkutils PUBLIC ../../vkutils)
target_compile_definitions(vkutils PUBLIC USE_VOLK USE_VULKAN)
target_link_libraries(vkutils PUBLIC volk::volk)

file(GLOB sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS *.cpp)

add_executable(testapp ${sources})
target_include_directories(testapp PRIVATE . ../../utils ../../LiteMath)
target_compile_options(testapp PRIVATE -Wno-attributes)
target_link_libraries(testapp PRIVATE vkutils)

set_target_properties(testapp PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
