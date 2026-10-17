#pragma once
#include <filesystem>
#include <functional>
#include <vector>

namespace clang {
    class CompilerInstance;
}

namespace kslicer {

    clang::CompilerInstance* compiler();

    struct InvocationParams {
        std::vector<std::filesystem::path> sources;             // *.cpp
        std::vector<std::filesystem::path> include_directories; // -I
    };

    int invoke_compiler(const InvocationParams& params, std::function<void()> callback);
    int invoke_compiler(const std::vector<std::string>& sources, const std::vector<std::string>& params,
                        std::function<void()> callback);

} // namespace kslicer
