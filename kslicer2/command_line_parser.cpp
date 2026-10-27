#include "command_line_parser.h"
#include "io.h"

#ifdef _WIN32
#include <Windows.h>
#include <llvm/Support/WindowsError.h>
#endif

namespace kslicer {

#ifdef _WIN32
    bool CommandLineParser::parse(int argc, char**) const
#else
    bool CommandLineParser::parse(int argc, char** argv) const
#endif
    {
#ifdef _WIN32
        std::unique_ptr<LPWSTR[], void (*)(LPWSTR*)> argv(::CommandLineToArgvW(::GetCommandLineW(), &argc),
                                                          [](LPWSTR* s) { LocalFree(s); });
        if (!argv) {
            std::error_code e = llvm::mapWindowsError(::GetLastError());
            kslicer_error() << e.message() << "\n";
            return false;
        }
#endif

        std::vector<bool> set_args(args_.size());

        bool error = false;
        for (int i = 1; i < argc; ++i) {
            bool is_known_option = false;
            std::filesystem::path arg{argv[i]};
            for (size_t j = 0; j < args_.size(); ++j) {
                if (arg == args_[j].name) {
                    is_known_option = true;
                    if (args_[j].store) {
                        if (i + 1 >= argc) {
                            kslicer_error() << "missing value after " << arg << "\n";
                            return false;
                        }
                        set_args[j] = true;
                        args_[j].store(args_[j].destination, argv[i + 1]);
                        ++i;
                    } else {
                        *static_cast<bool*>(args_[j].destination) = true;
                    }
                    break;
                }
            }
            if (!is_known_option) {
                error = true;
                kslicer_error() << "unrecognized command line option '" << arg << "'\n";
            }
        }

        for (size_t i = 0; i < args_.size(); ++i) {
            if (!args_[i].is_optional && !set_args[i]) {
                kslicer_error() << "option " << args_[i].name << " is missing\n";
                error = true;
            }
        }

        return !error;
    }

} // namespace kslicer
