#include "command_line.h"
#include "utils.h"

namespace kslicer {
    CommandLine parse_command_line(int argc, char** argv) {
        CommandLine out{};
        bool selfdir_set = false;
        bool config_set = false;
        bool error = false;

        for (int i = 1; i < argc; i += 1) {
            std::string key = argv[i];
            if (key == "-selfdir") {
                if (i + 1 >= argc) {
                    error = true;
                    fprintf(stderr, "error: Expected value for -selfdir.\n");
                    break;
                }
                out.selfdir = argv[i + 1];
                selfdir_set = true;
                i++;
            } else if (key == "-config") {
                if (i + 1 >= argc) {
                    error = 1;
                    fprintf(stderr, "error: Expected value for -config.\n");
                }
                out.config = argv[i + 1];
                config_set = true;
                i++;
            }
        }

        if (!selfdir_set) {
            error = true;
            fprintf(stderr, "error: -selfdir is not provided.\n");
        }
        if (!config_set) {
            error = true;
            fprintf(stderr, "error: -config is not provided.\n");
        }
        if (error) {
            exit(1);
        }

        return out;
    }
} // namespace kslicer