#pragma once
#include <filesystem>

namespace kslicer {

    struct CommandLine {
        std::filesystem::path selfdir;
        std::filesystem::path config;
    };

    CommandLine parse_command_line(int argc, char** argv);

} // namespace kslicer
