cmake_minimum_required (VERSION 3.8)
project (kernel_slicer)
set (CMAKE_CXX_STANDARD 17)

find_package(LLVM 11 REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

include_directories("include")
include_directories(${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_INCLUDE_DIRS}/../lib)
add_definitions(${LLVM_DEFINITIONS})

message("${LLVM_INCLUDE_DIRS}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_executable(kslicer 
               kslicer_main.cpp
               ast_matchers.cpp
               class_gen.cpp
               class_gen_ipv.cpp
               class_gen_rtv.cpp
               class_data.cpp
               extractor.cpp
               initial_pass.cpp
               template_rendering.cpp
               rewrite_kernel.cpp
               rewrite_control.cpp
               reduction.cpp
               )

target_link_libraries(kslicer PRIVATE -lclang -lLLVM-11 
                     -lclangAST -lclangTooling -lclangBasic -lclangASTMatchers 
                     -lclangFrontendTool -lclangFrontend -lclangDriver -lclangSerialization 
                     -lclangCodeGen -lclangParse -lclangSema -lclangStaticAnalyzerFrontend 
                     -lclangStaticAnalyzerCheckers -lclangStaticAnalyzerCore -lclangAnalysis
                     -lclangIndex -lclangARCMigrate -lclangRewrite -lclangAST -lclangLex -lclangEdit
                     ) 
