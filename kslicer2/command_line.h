#pragma once
#include <filesystem>
#include <optional>

namespace kslicer {

    struct CommandLine {
        bool from_args(int argc, char** argv);
        void dump() const;

        std::filesystem::path kslicer_directory;    // -selfdir
        std::filesystem::path config_path;          // -config
        bool verbose = false;                       // --verbose
        std::optional<std::string> shader_compiler; // -shaderCC
    };

} // namespace kslicer
