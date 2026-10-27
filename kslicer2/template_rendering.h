#pragma once
#include <filesystem>
#include <nlohmann/json.hpp>

namespace kslicer {
    bool render_template(nlohmann::json data, std::filesystem::path template_path, std::filesystem::path output_path);
}
