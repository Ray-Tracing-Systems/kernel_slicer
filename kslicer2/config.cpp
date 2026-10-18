#include "config.h"
#include <fstream>
#include <nlohmann/json.hpp>

namespace kslicer {
    Config read_config_from_file(std::filesystem::path path) {
        std::ifstream file(path);
        auto data = nlohmann::json::parse(file);
        Config out{};
        out.config_path = path;
        out.main_class = data["mainClass"];
        out.suffix = data["options"]["-suffix"];
        out.base_directory = data["baseDirectory"].get<std::string>();
        out.shaders_directory = data["options"]["-shaderFolderPrefix"].get<std::string>();
        for (auto i : data["source"]) {
            out.sources.push_back(i.get<std::string>());
        }
        for (auto i : data["includeProcess"]) {
            out.include_directories.push_back(i.get<std::string>());
        }
        for (auto i : data["includeIgnore"]) {
            out.include_directories.push_back(i.get<std::string>());
        }
        return out;
    }
} // namespace kslicer