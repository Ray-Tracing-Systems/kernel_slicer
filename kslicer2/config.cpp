#include "config.h"
#include "utils.h"
#include <fstream>
#include <nlohmann/json.hpp>
namespace kslicer {

    bool Config::from_file(std::filesystem::path config_path) {
        std::ifstream file(config_path);
        auto data = nlohmann::json::parse(file);
        class_name = data["mainClass"];
        suffix = data["options"]["-suffix"];
        shaders_directory_prefix = data["options"]["-shaderFolderPrefix"].get<std::string>();
        shader_compiler = data["options"]["-shaderCC"];

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
