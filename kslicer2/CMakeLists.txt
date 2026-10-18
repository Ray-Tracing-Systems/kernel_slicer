
add_executable(kslicer2
    main.cpp
    utils.cpp
    errors.cpp
    codegen_host.cpp
    class_info.cpp
    codegen.cpp
    compiler.cpp
    command_line.cpp
    config.cpp
)

target_compile_definitions(kslicer2 PRIVATE ${LLVM_DEFINITIONS} ${CLANG_DEFINITIONS})
target_include_directories(kslicer2 PRIVATE ../include ${LLVM_INCLUDE_DIRS} ${CLANG_INCLUDE_DIRS})
target_link_directories(kslicer2 PRIVATE ${LLVM_LIBRARY_DIRS})
target_link_libraries(kslicer2 PRIVATE clangTooling)
