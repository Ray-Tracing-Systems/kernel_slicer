cmake_minimum_required(VERSION 3.20)
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
    graphics_pipeline.cpp
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

    # Imported target LLVMDebugInfoPDB wants to link with a library which probably wouldn't be present.
    # Clearing INTERFACE_LINK_LIBRARIES does not break linking, so that library is not need anyway.
    set_target_properties(LLVMDebugInfoPDB PROPERTIES INTERFACE_LINK_LIBRARIES "")

    if(WIN32 AND(NOT MINGW))
        # When linking with clang static libraries on Windows this macro should be defined for macro CLANG_ABI to be defined correctly.
        # Not doing so can cause linking warnings (and possible silent errors)
        # Normally, this macro is added during CMake configuration of clang/llvm project and then propagated through imported targets in ClangTargets.cmake,
        # but if clang/llvm is built on Windows not by msvc but by clang then this macro is not defined.
        # It is a bug and it is already fixed in llvm 23.0.
        target_compile_definitions(kslicer PRIVATE CLANG_BUILD_STATIC)
    endif()
endif()

target_include_directories(kslicer SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS} ${CLANG_INCLUDE_DIRS})
target_include_directories(kslicer PRIVATE include)
target_link_directories(kslicer PRIVATE ${LLVM_LIBRARY_DIRS})
target_link_libraries(kslicer PRIVATE clangTooling)

if(MSVC)
    # This option is not ON by default because Microslop is stupid
    target_compile_options(kslicer PRIVATE /bigobj)
endif()
