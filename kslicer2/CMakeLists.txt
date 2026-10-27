
add_executable(kslicer2
    main.cpp
    io.cpp
    utils.cpp
    errors.cpp
    codegen_host.cpp
    class_info.cpp
    codegen.cpp
    compiler.cpp
    command_line.cpp
    command_line_parser.cpp
    config.cpp
    kslicer_cpp_attributes.cpp
)

target_compile_options(kslicer2 PRIVATE -include ${CMAKE_CURRENT_SOURCE_DIR}/do_not_use_printf_and_cout_use_kslicer_outs_intead.h)
target_compile_definitions(kslicer2 PRIVATE ${LLVM_DEFINITIONS} ${CLANG_DEFINITIONS})
target_include_directories(kslicer2 SYSTEM PRIVATE ../include ${LLVM_INCLUDE_DIRS} ${CLANG_INCLUDE_DIRS})
target_link_directories(kslicer2 PRIVATE ${LLVM_LIBRARY_DIRS})
target_link_libraries(kslicer2 PRIVATE clangTooling)
target_compile_options(kslicer2 PRIVATE -Wall -Wextra -Werror)
