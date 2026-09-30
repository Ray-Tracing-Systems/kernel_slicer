
add_executable(common_rewriter main.cpp common_rewriter.cpp)

target_link_libraries(common_rewriter PRIVATE clangTooling)
