#include "class.h"
#include "utils.h"
#include <clang/AST/AST.h>

namespace common_rewriter {
    using namespace clang;
    using llvm::outs;

    std::vector<const clang::CXXMethodDecl*> kernels;
    std::vector<const clang::CXXMethodDecl*> controls;
    std::vector<const clang::CXXMethodDecl*> methods;
    std::vector<const clang::FieldDecl*> buffer_fields;
    std::vector<clang::QualType> buffer_fields_elements_types;
    std::vector<const clang::FieldDecl*> uniform_fields;

    void discover_class(std::string main_class_name) {
        auto main_class = dyn_cast<CXXRecordDecl>(find_global_declaration(main_class_name));

        for (auto i : main_class->methods()) {
            if (i->getNameAsString().substr(0, 6) == "kernel")
                kernels.push_back(i);
        }

        for (auto i : main_class->methods()) {
            if (auto def = i->getDefinition()) {
                bool is_control = false;
                traverse_this_calls(def->getBody(), [&](const CXXMemberCallExpr* expr) {
                    if (contains(kernels, expr->getMethodDecl()))
                        is_control = true;
                });
                if (is_control)
                    controls.push_back(i);
            }
        }

        for (auto i : kernels) {
            traverse_this_calls(i->getBody(), [&](const CXXMemberCallExpr* expr) {
                push_back_unique(methods, dyn_cast<CXXMethodDecl>(expr->getMethodDecl()->getDefinition()));
            });
        }

        for (size_t i = 0; i < methods.size(); i++) {
            traverse_this_calls(methods[i]->getBody(), [&](const CXXMemberCallExpr* expr) {
                push_back_unique(methods, dyn_cast<CXXMethodDecl>(expr->getMethodDecl()->getDefinition()));
            });
        }

        auto add_field = [&](const MemberExpr* expr) {
            auto field = dyn_cast<FieldDecl>(expr->getMemberDecl());
            auto type = field->getType().getCanonicalType();
            if (is_vector_specialization(type)) {
                push_back_unique(buffer_fields, field);
            } else {
                push_back_unique(uniform_fields, field);
            }
        };

        for (auto i : kernels) {
            traverse_this_fields(i->getBody(), add_field);
        }
        for (auto i : methods) {
            traverse_this_fields(i->getBody(), add_field);
        }

        outs() << "controls:\n";
        for (auto i : controls) {
            outs() << i->getNameAsString() << "\n";
        }
        outs() << "\n";

        outs() << "kernels:\n";
        for (auto i : kernels) {
            outs() << i->getNameAsString() << "\n";
        }
        outs() << "\n";

        outs() << "methods:\n";
        for (auto i : methods) {
            outs() << i->getNameAsString() << "\n";
        }
        outs() << "\n";

        outs() << "uniform fields:\n";
        for (auto i : uniform_fields)
            outs() << i->getNameAsString() << "\n";
        outs() << "\n";

        outs() << "buffer fields:\n";
        for (auto i : buffer_fields)
            outs() << i->getNameAsString() << "\n";
        outs() << "\n";
    }

} // namespace common_rewriter
