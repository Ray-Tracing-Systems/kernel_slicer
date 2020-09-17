cmake_minimum_required (VERSION 3.8)
project (kernel_slicer)
set (CMAKE_CXX_STANDARD 17)

#set(LibClang_DIR "/home/frol/PROG/VEXCC/clspv/third_party/llvm/clang/lib")
#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules")
#
#find_package(LibClang REQUIRED)
#
#add_definitions(${LibClang_DEFINITIONS})
#
#include_directories(
#  ${LibClang_INCLUDE_DIRS}
#)
#
#add_executable(kslicer kslicer_main.cpp)
#target_link_libraries(kslicer ${LibClang_LIBRARIES} )
#
#add_subdirectory(test)

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
