cmake_minimum_required(VERSION 3.8)
project(kernel_slicer)

set(CMAKE_CXX_STANDARD 17)

option(USE_LLVM22 "Build with LLVM/Clang 22.1 instead of 17" OFF)

if(USE_LLVM22)
    find_package(LLVM 22.1 REQUIRED CONFIG)
    find_package(Clang 22.1 REQUIRED CONFIG)
else()
    find_package(LLVM 17 REQUIRED CONFIG)
    find_package(Clang 17 REQUIRED CONFIG)
endif()

add_executable(kslicer
    kslicer_main.cpp
    kslicer_utils.cpp
    kslicer_warnings.cpp
    hashmap.cpp
    kslicer_def.cpp
    attributes.cpp
    block_expansion.cpp
    initial_pass.cpp
    initial_pass_compos.cpp
    extractor.cpp
    ast_matchers.cpp
    class_gen.cpp
    class_gen_ipv.cpp
    class_gen_rtv.cpp
    class_gen_vfh.cpp
    class_data.cpp
    template_rendering.cpp
    template_rendering_host.cpp
    rewrite_functions.cpp
    rewrite_kernel.cpp
    rewrite_kernel2.cpp
    rewrite_kernel_info.cpp
    setter_process.cpp
    reduction.cpp
    feature_status.cpp
    neural_nets.cpp
    shaders_clspv.cpp
    shaders_glsl.cpp
    shaders_ispc.cpp
    shaders_slang.cpp
    shaders_cuda.cpp
    tr_host_vulkan.cpp
    tr_host_cuda.cpp
    tr_host_ispc.cpp
    tr_host_wgpu.cpp
)

target_compile_definitions(kslicer PRIVATE ${LLVM_DEFINITIONS} ${CLANG_DEFINITIONS})
if(USE_LLVM22)
  target_compile_definitions(kslicer PRIVATE KSLICER_USE_LLVM22=1)
endif()
target_include_directories(kslicer PRIVATE include ${LLVM_INCLUDE_DIRS} ${CLANG_INCLUDE_DIRS})
target_link_directories(kslicer PRIVATE ${LLVM_LIBRARY_DIRS})
target_link_libraries(kslicer PRIVATE clangTooling)

if(MSVC)
    # This option is not ON by default because Microslop is stupid
    target_compile_options(kslicer PRIVATE /bigobj)
endif()
