#pragma once
#include "class_info.h"
#include "config.h"

namespace kslicer {

    bool codegen(std::filesystem::path kslicer_directory, const Config& config, const ClassInfo& info);

}
