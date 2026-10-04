
add_executable(common_rewriter
    main.cpp
    class.cpp
    common_rewriter.cpp
    utils.cpp
    errors.cpp
)

target_compile_definitions(common_rewriter PRIVATE ${LLVM_DEFINITIONS} ${CLANG_DEFINITIONS})
target_include_directories(common_rewriter PRIVATE include ${LLVM_INCLUDE_DIRS} ${CLANG_INCLUDE_DIRS})
target_link_directories(common_rewriter PRIVATE ${LLVM_LIBRARY_DIRS})
target_link_libraries(common_rewriter PRIVATE clangTooling)
