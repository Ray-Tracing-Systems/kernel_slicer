#pragma once
#include <filesystem>
#include <functional>
#include <vector>

namespace clang {
    class CompilerInstance;
}

namespace kslicer {

    clang::CompilerInstance* compiler();

    struct CompilerParams {
        std::filesystem::path source;                           // *.cpp
        std::vector<std::filesystem::path> include_directories; // -I
    };

    int invoke_compiler(const CompilerParams& params, std::function<void()> callback);

} // namespace kslicer
