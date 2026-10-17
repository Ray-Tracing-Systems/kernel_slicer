#include "class_info.h"
#include "codegen_host.h"
#include "compiler.h"
#include "errors.h"
#include "utils.h"
namespace kslicer {
    using namespace clang;
    using namespace common_rewriter;
    void handle_translation_unit() {
        auto decl = find_global_declaration("VolumeRenderer");
        ClassInfo class_info(dyn_cast<CXXRecordDecl>(decl));
        note_found(decl->getLocation(), "found");
        class_info.dump();
        codegen(class_info);
    }
} // namespace kslicer

int main(int argc, const char** argv) {
    kslicer::InvocationParams params;
    params.sources = {"D:/gml_private/VolumeRenderer/VolumeRenderer.cpp"};
    params.include_directories = {"D:/gml_private/", "D:/gml_private/VolumeRenderer/",
                                  "D:/gml_private/dependencies/1st-party/LiteMath/", "D:/kernel_slicer/TINYSTL"};
    return kslicer::invoke_compiler(params, kslicer::handle_translation_unit);
}
