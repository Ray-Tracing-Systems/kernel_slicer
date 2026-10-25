#pragma once
#include <filesystem>
#include <vector>

namespace kslicer {
    struct Config {
        explicit Config(std::filesystem::path config_path);
        void dump() const;

        std::string class_name;
        std::string suffix;
        std::string shaders_directory_prefix;
        std::vector<std::filesystem::path> sources;
        std::vector<std::filesystem::path> include_directories;
        std::filesystem::path shaders_directory;
    };

} // namespace kslicer
