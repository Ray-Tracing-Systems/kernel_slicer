#include "config.h"
#include "utils.h"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
namespace kslicer {

    Config::Config(std::filesystem::path config_path) {
        std::ifstream file(config_path);
        auto data = nlohmann::json::parse(file);
        class_name = data["mainClass"];
        suffix = data["options"]["-suffix"];
        shaders_directory_prefix = data["options"]["-shaderFolderPrefix"].get<std::string>();

        std::filesystem::path config_directory = config_path.parent_path();
        std::filesystem::path base_directory = config_directory / data["baseDirectory"].get<std::string>();
        for (auto i : data["source"]) {
            sources.push_back((base_directory / i.get<std::string>()).lexically_normal());
        }
        for (auto i : data["includeProcess"]) {
            include_directories.push_back((base_directory / i.get<std::string>()).lexically_normal());
        }
        for (auto i : data["includeIgnore"]) {
            include_directories.push_back((base_directory / i.get<std::string>()).lexically_normal());
        }
        shaders_directory = (base_directory / shaders_directory_prefix / ("shaders" + suffix)).lexically_normal();
    }

    void Config::dump() const {

        std::cout << "class_name: " << class_name << std::endl;
        std::cout << "suffix: " << suffix << std::endl;
        std::cout << "shaders_directory_prefix: " << shaders_directory_prefix << std::endl;
        for (size_t i = 0; i < sources.size(); ++i) {
            std::cout << "sources[" << i << "]: " << sources[i] << std::endl;
        }
        for (size_t i = 0; i < include_directories.size(); ++i) {
            std::cout << "include_directories[" << i << "]: " << include_directories[i] << std::endl;
        }
        std::cout << "shaders_directory: " << shaders_directory << std::endl;
    }

} // namespace kslicer
