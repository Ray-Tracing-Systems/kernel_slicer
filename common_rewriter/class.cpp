#include "class.h"
#include "errors.h"
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
    std::map<const CXXMethodDecl*, std::vector<const ParmVarDecl*>> kernels_dimentions;
    std::map<const CXXMethodDecl*, std::vector<const CXXMethodDecl*>> controls_kernel_calls;
    std::map<const CXXMethodDecl*, std::vector<const ValueDecl*>> kernels_bindings;

    const ParmVarDecl* get_for_loop_bound(const ForStmt* stmt) {
        auto cond = stmt->getCond();
        if (auto expr = dyn_cast<BinaryOperator>(cond)) {
            auto op = expr->getOpcodeStr();
            if (op == "<") {
                auto bound = expr->getRHS()->IgnoreParenImpCasts();
                if (auto ref = dyn_cast<DeclRefExpr>(bound)) {
                    if (auto var = dyn_cast<ParmVarDecl>(ref->getDecl()))
                        return var;
                }
            }
        }
        error_invalid_for_loop_condition(cond);
        return nullptr;
    }

    void discover_class(std::string main_class_name) {
        auto main_class = dyn_cast<CXXRecordDecl>(find_global_declaration(main_class_name));

        for (auto i : main_class->methods()) {
            if (i->getNameAsString().substr(0, 6) == "kernel") {
                push_back_unique(kernels, dyn_cast<CXXMethodDecl>(i->getDefinition()));
            }
        }

        for (auto i : main_class->methods()) {
            if (auto def = i->getDefinition()) {
                bool is_control = false;
                traverse_this_method_calls(def->getBody(), [&](const CXXMemberCallExpr* expr) {
                    if (contains(kernels, dyn_cast<CXXMethodDecl>(expr->getMethodDecl()->getDefinition())))
                        is_control = true;
                });
                if (is_control)
                    push_back_unique(controls, dyn_cast<CXXMethodDecl>(def));
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

        for (auto i : kernels) {
            for (auto stmt : i->getBody()->children()) {
                if (auto for1 = dyn_cast<ForStmt>(stmt)) {
                    kernels_dimentions[i].push_back(get_for_loop_bound(for1));
                    for (auto stmt : for1->getBody()->children()) {
                        if (auto for2 = dyn_cast<ForStmt>(stmt)) {
                            kernels_dimentions[i].push_back(get_for_loop_bound(for2));
                            for (auto stmt : for2->getBody()->children()) {
                                if (auto for3 = dyn_cast<ForStmt>(stmt)) {
                                    kernels_dimentions[i].push_back(get_for_loop_bound(for3));
                                }
                            }
                        }
                    }
                }
            }
            REWRITE_ASSERT(kernels_dimentions[i].size() == i->getNameAsString().substr(6, 1)[0] - '0');
        }

        for (auto kernel : kernels) {
            for (size_t i = 0; i < kernel->getNumParams(); i++) {
                auto p = kernel->getParamDecl(i);
                if (p->getType()->isPointerType())
                    kernels_bindings[kernel].push_back(p);
            }
            // outs() << "size: " << kernels_bindings[kernel].size() << "\n";
        }

        outs() << "controls:\n";
        for (auto i : controls) {
            outs() << i->getNameAsString() << "\n";
        }
        outs() << "\n";

        outs() << "kernels:\n";
        for (auto i : kernels) {
            outs() << i->getNameAsString() << "(";
            for (auto j : kernels_dimentions[i]) {
                outs() << j->getNameAsString() << ", ";
            }
            outs() << ")\n";
            outs() << "bindings:\n";
            for (auto binding : kernels_bindings[i])
                outs() << binding->getNameAsString() << "\n";
        }
        outs() << "\n";

        size_t calls = 0;
        for (auto control : controls) {
            traverse_this_method_calls(control->getBody(), [&](const CXXMemberCallExpr* method) {
                if (contains(kernels, dyn_cast<CXXMethodDecl>(method->getMethodDecl()->getDefinition()))) {
                    // controls_kernel_calls[control].push_back(method->getMethodDecl());
                    push_back_unique(controls_kernel_calls[control],
                                     dyn_cast<CXXMethodDecl>(method->getMethodDecl()->getDefinition()));
                    // calls++;
                }
            });
        }
        outs() << "Total calls: " << calls << "\n";

        // outs() << "methods:\n";
        // for (auto i : methods) {
        //     outs() << i->getNameAsString() << "\n";
        // }
        // outs() << "\n";

        // outs() << "uniform fields:\n";
        // for (auto i : uniform_fields)
        //     outs() << i->getNameAsString() << "\n";
        // outs() << "\n";

        // outs() << "buffer fields:\n";
        // for (auto i : buffer_fields)
        //     outs() << i->getNameAsString() << "\n";
        // outs() << "\n";

        // outs() << "global functions:\n";
        // for (auto i : global_functions)
        //     outs() << i->getNameAsString() << "\n";
        // outs() << "\n";
    }

} // namespace common_rewriter
