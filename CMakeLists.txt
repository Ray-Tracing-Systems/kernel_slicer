cmake_minimum_required(VERSION 3.8)
project(kernel_slicer)

set(CMAKE_CXX_STANDARD 17)
set(LLVM_LINK_COMPONENTS support)

include_directories("include")

# file(GLOB_RECURSE SRC_DIR "include/*.h" "include/nlohmann/*.h" )
# set_source_files_properties(${SRC_DIR} 
#                             PROPERTIES COMPILE_FLAGS "-w")

add_clang_executable(kslicer 
                     kslicer_main.cpp
                     kslicer_utils.cpp
                     kslicer_warnings.cpp
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
                     rewrite_control.cpp
                     setter_process.cpp
                     reduction.cpp
                     shaders_clspv.cpp
                     shaders_glsl.cpp
                     shaders_ispc.cpp
                     )

target_compile_options(kslicer PRIVATE -fexceptions)

target_link_libraries(kslicer PRIVATE 
                      clang
                      clangTooling
                      clangBasic
                      clangASTMatchers
                      ) 
