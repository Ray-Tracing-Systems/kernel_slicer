#pragma once
#include <filesystem>
#include <nlohmann/json_fwd.hpp>

namespace kslicer {
    bool render_template(const nlohmann::json& data, std::filesystem::path template_path, std::filesystem::path output_path);
}
