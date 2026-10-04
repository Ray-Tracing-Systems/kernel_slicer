#include "class.h"
#include "utils.h"
#include <clang/AST/AST.h>

namespace common_rewriter {
    using namespace clang;
    using llvm::outs;

    std::vector<const CXXMethodDecl*> kernels;
    std::vector<const CXXMethodDecl*> controls;
    std::vector<const CXXMethodDecl*> methods;
    std::vector<const FieldDecl*> buffer_fields;
    std::vector<QualType> buffer_fields_elements_types;
    std::vector<const FieldDecl*> uniform_fields;
    std::vector<const FunctionDecl*> global_functions;
    std::vector<const VarDecl*> global_constants;

    void discover_class(std::string main_class_name) {
        auto main_class = dyn_cast<CXXRecordDecl>(find_global_declaration(main_class_name));

        for (auto i : main_class->methods()) {
            if (i->getNameAsString().substr(0, 6) == "kernel")
                kernels.push_back(i);
        }

        for (auto i : main_class->methods()) {
            if (auto def = i->getDefinition()) {
                bool is_control = false;
                traverse_this_method_calls(def->getBody(), [&](const CXXMemberCallExpr* expr) {
                    if (contains(kernels, expr->getMethodDecl()))
                        is_control = true;
                });
                if (is_control)
                    controls.push_back(i);
            }
        }

        for (auto i : kernels) {
            traverse_this_method_calls(i->getBody(), [&](const CXXMemberCallExpr* expr) {
                push_back_unique(methods, dyn_cast<CXXMethodDecl>(expr->getMethodDecl()->getDefinition()));
            });
        }

        for (size_t i = 0; i < methods.size(); i++) {
            traverse_this_method_calls(methods[i]->getBody(), [&](const CXXMemberCallExpr* expr) {
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
            traverse_global_function_calls(i->getBody(), [](const CallExpr* expr) {
                push_back_unique(global_functions, dyn_cast<FunctionDecl>(expr->getCalleeDecl()));
            });
            traverse_this_field_accesses(i->getBody(), add_field);
        }
        for (auto i : methods) {
            traverse_global_function_calls(i->getBody(), [](const CallExpr* expr) {
                push_back_unique(global_functions, dyn_cast<FunctionDecl>(expr->getCalleeDecl()));
            });
            traverse_this_field_accesses(i->getBody(), add_field);
        }

        for (size_t i = 0; i < global_functions.size(); ++i) {
            traverse_global_function_calls(global_functions[i]->getBody(), [](const CallExpr* expr) {
                push_back_unique(global_functions, dyn_cast<FunctionDecl>(expr->getCalleeDecl()));
            });
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

        outs() << "global functions:\n";
        for (auto i : global_functions)
            outs() << i->getNameAsString() << "\n";
        outs() << "\n";
    }

} // namespace common_rewriter
