
add_executable(common_rewriter
    main.cpp
    utils.cpp
    errors.cpp
    codegen_host.cpp
    class_info.cpp
    codegen.cpp
    compiler.cpp
)

target_compile_definitions(common_rewriter PRIVATE ${LLVM_DEFINITIONS} ${CLANG_DEFINITIONS})
target_include_directories(common_rewriter PRIVATE ../include ${LLVM_INCLUDE_DIRS} ${CLANG_INCLUDE_DIRS})
target_link_directories(common_rewriter PRIVATE ${LLVM_LIBRARY_DIRS})
target_link_libraries(common_rewriter PRIVATE clangTooling)
