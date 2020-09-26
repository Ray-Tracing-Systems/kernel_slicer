cmake_minimum_required (VERSION 3.8)
project (kernel_slicer)
set (CMAKE_CXX_STANDARD 17)

set(LLVM_LINK_COMPONENTS support)

add_clang_executable(kslicer
                     kslicer_main.cpp
                     )

target_link_libraries(kslicer
                      PRIVATE
                      clangTooling
                      clangBasic
                      clangASTMatchers
                      )
