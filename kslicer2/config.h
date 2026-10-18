#pragma once
#include <filesystem>
#include <vector>

namespace kslicer {
    struct Config {
        std::filesystem::path config_path;
        std::filesystem::path kslicer_directory;
        std::string main_class;
        std::string suffix;
        std::filesystem::path base_directory;
        std::filesystem::path shaders_directory;
        std::vector<std::filesystem::path> sources;
        std::vector<std::filesystem::path> include_directories;
    };

    Config read_config_from_file(std::filesystem::path path);

} // namespace kslicer
