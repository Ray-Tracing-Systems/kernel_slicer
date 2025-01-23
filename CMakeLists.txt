cmake_minimum_required(VERSION 3.8)
project(kernel_slicer)

set(CMAKE_CXX_STANDARD 17)
set(LLVM_LINK_COMPONENTS support)

set(KSLICER_SOURCE_FILES
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
    rewrite_kernel_info.cpp
    rewrite_control.cpp
    setter_process.cpp
    reduction.cpp
    shaders_clspv.cpp
    shaders_glsl.cpp
    shaders_ispc.cpp)

include_directories("include")

option(KSLICER_LLVM_BUILD "Build kernel_slicer as a LLVM project module" OFF)

if(NOT KSLICER_LLVM_BUILD)
  set(LLVM_VER "17")
  set(LLVM_LIB "LLVM-${LLVM_VER}")

  if(WIN32)
    if(MSVC)
      add_compile_options(/bigobj)
    endif()
  else()
    set(PLATFORM_DEPENDENT_LIBS clang ${LLVM_LIB})
  endif()

  find_package(LLVM ${LLVM_VER} REQUIRED CONFIG)
  find_package(Clang ${LLVM_VER} REQUIRED)

  message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

  include_directories(${LLVM_INCLUDE_DIRS})
  include_directories(${CLANG_INCLUDE_DIRS})
  link_directories(${LLVM_LIBRARY_DIRS})
  add_definitions(${LLVM_DEFINITIONS})
  add_definitions(${CLANG_DEFINITIONS})

  message("LLVM_INCLUDE_DIRS = ${LLVM_INCLUDE_DIRS}")
  message("LLVM_LIBRARY_DIRS = ${LLVM_LIBRARY_DIRS}")
  # message("CLANG_INCLUDE_DIRS = ${CLANG_INCLUDE_DIRS}")
  # message("CLANG_LIBRARY_DIRS = ${CLANG_LIBRARY_DIRS}") set(CMAKE_MODULE_PATH
  # ${CMAKE_MODULE_PATH} "${LLVM_CMAKE_DIR}")

  add_executable(kslicer ${KSLICER_SOURCE_FILES})

  target_link_libraries(
    kslicer
    PRIVATE ${PLATFORM_DEPENDENT_LIBS}
            clangAST
            clangTooling
            clangBasic
            clangASTMatchers
            clangFrontendTool
            clangFrontend
            clangDriver
            clangSerialization
            clangCodeGen
            clangParse
            clangSema
            clangStaticAnalyzerFrontend
            clangStaticAnalyzerCheckers
            clangStaticAnalyzerCore
            clangAnalysis
            clangIndex
            clangARCMigrate
            clangRewrite
            clangLex
            clangEdit)
else() # NOT DEFINED KSLICER_LLVM_BUILD
  message(STATUS "kernel_slicer build with LLVM")

  # file(GLOB_RECURSE SRC_DIR "include/*.h" "include/nlohmann/*.h" )
  # set_source_files_properties(${SRC_DIR} PROPERTIES COMPILE_FLAGS "-w")

  add_clang_executable(kslicer ${KSLICER_SOURCE_FILES})

  target_compile_options(kslicer PRIVATE -fexceptions)

  target_link_libraries(kslicer PRIVATE clang clangTooling clangBasic
                                        clangASTMatchers)
endif() # DEFINED KSLICER_LLVM_BUILD
