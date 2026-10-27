#include "command_line.h"
#include "command_line_parser.h"
#include "io.h"

namespace kslicer {

    bool CommandLine::from_args(int argc, char** argv) {
        CommandLineParser parser;
        parser.add_option("-selfdir", kslicer_directory);
        parser.add_option("-config", config_path);
        parser.add_flag("--verbose", verbose);
        parser.add_option("-shaderCC", shader_compiler);
        return parser.parse(argc, argv);
    }

    void CommandLine::dump() const {
        outs() << "kslicer_directory: " << kslicer_directory << "\n";
        outs() << "config_path: " << config_path << "\n";
        outs() << "verbose: " << verbose << "\n";
        outs() << "shader_compiler: " << shader_compiler << "\n";
    }

} // namespace kslicer
