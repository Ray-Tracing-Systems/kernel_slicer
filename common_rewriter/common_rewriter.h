#pragma once
#include <string>

namespace clang
{
    class CompilerInstance;
}

namespace common_rewriter
{
    extern const clang::CompilerInstance *compiler_instance;
    void rewrite_kernel(std::string class_name, std::string kernel_name);

}
