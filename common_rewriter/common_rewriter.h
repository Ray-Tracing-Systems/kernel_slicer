#pragma once
#include <string>

namespace clang
{
    class CompilerInstance;
}

namespace common_rewriter
{
    extern const clang::CompilerInstance *compiler_instance;
    void rewrite_class(std::string class_name);

}
