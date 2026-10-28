#include "config.h"
#include "io.h"
#include <nlohmann/json.hpp>
namespace kslicer {

    bool Config::from_file(std::filesystem::path config_path) {
        std::string config_text;
        if (!read_entire_text_file(config_path, config_text))
            return false;

        nlohmann::json config = nlohmann::json::parse(config_text, nullptr, true, /* ignore_comments */ true);
        class_name = config.at("mainClass");
        suffix = config.at("options").at("-suffix");
        shaders_directory_prefix = config.at("options").at("-shaderFolderPrefix");
        shader_compiler = config.at("options").at("-shaderCC");

        std::filesystem::path config_directory = config_path.parent_path();
        std::filesystem::path base_directory = config_directory / config.at("baseDirectory");

        std::vector<std::filesystem::path> source = config.at("source");
        std::vector<std::filesystem::path> include_process = config.at("includeProcess");
        std::vector<std::filesystem::path> include_ignore = config.at("includeIgnore");

        for (auto i : source) {
            sources.push_back(base_directory / i);
        }
        for (auto i : include_process) {
            include_directories.push_back(base_directory / i);
        }
        for (auto i : include_ignore) {
            include_directories.push_back(base_directory / i);
        }

        return true;
    }

    void Config::dump() const {

        outs() << "class_name: " << class_name << "\n";
        outs() << "suffix: " << suffix << "\n";
        outs() << "shaders_directory_prefix: " << shaders_directory_prefix << "\n";
        outs() << "shader_compiler: " << shader_compiler << "\n";
        for (size_t i = 0; i < sources.size(); ++i) {
            outs() << "sources[" << i << "]: " << sources[i] << "\n";
        }
        for (size_t i = 0; i < include_directories.size(); ++i) {
            outs() << "include_directories[" << i << "]: " << include_directories[i] << "\n";
        }
        outs() << "shaders_directory: " << shaders_directory << "\n";
    }

} // namespace kslicer
