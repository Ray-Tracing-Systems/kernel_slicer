#include "class_info.h"
#include "codegen_host.h"
#include "command_line.h"
#include "compiler.h"
#include "config.h"
#include "errors.h"
#include "utils.h"
namespace kslicer {
    using namespace clang;

    void handle_translation_unit(const Config& config) {
        auto decl = find_global_declaration(config.main_class);
        ClassInfo class_info(dyn_cast<CXXRecordDecl>(decl));
        note_found(decl->getLocation(), "found");
        class_info.dump();
        codegen(config, class_info);
    }
} // namespace kslicer

int main(int argc, char** argv) {
    using namespace kslicer;

    CommandLine command_line = parse_command_line(argc, argv);
    llvm::outs() << "-selfdir: " << command_line.selfdir.string() << "\n";
    llvm::outs() << "-config: " << command_line.config.string() << "\n";

    Config config = read_config_from_file(command_line.config);
    config.kslicer_directory = command_line.selfdir;
    llvm::outs() << "config.main_class: " << config.main_class << "\n";
    llvm::outs() << "config.suffix: " << config.suffix << "\n";
    llvm::outs() << "config.base_directory: " << config.base_directory.string() << "\n";
    for (auto i : config.sources)
        llvm::outs() << "config.sources: " << i.string() << "\n";
    for (auto i : config.include_directories)
        llvm::outs() << "config.include_directories: " << i.string() << "\n";
    llvm::outs() << "config.shaders_directory: " << config.shaders_directory.string() << "\n";

    auto config_dir = command_line.config.parent_path();
    auto base = (config_dir / config.base_directory).lexically_normal();

    llvm::outs() << "base: " << base.string() << "\n";

    CompilerParams params;
    params.source = base / config.sources[0];
    params.include_directories.push_back(command_line.selfdir / "TINYSTL");
    for (auto i : config.include_directories) {
        params.include_directories.push_back(base / i);
    }

    return invoke_compiler(params, [&]() { handle_translation_unit(config); });
}
