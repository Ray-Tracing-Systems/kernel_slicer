#pragma once
#include <filesystem>

namespace kslicer {

    struct CommandLine {
        static CommandLine from_args(int argc, char** argv);
        void dump() const;

        std::filesystem::path kslicer_directory;
        std::filesystem::path config_path;
    };

} // namespace kslicer
