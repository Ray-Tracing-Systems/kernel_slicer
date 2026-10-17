#include "class_info.h"
#include "errors.h"
#include "utils.h"

namespace common_rewriter {
    using llvm::outs;

    const KernelInfo* ClassInfo::get_kernel(const CXXMethodDecl* decl) const {
        if (auto def = decl->getDefinition()) {
            for (const KernelInfo& kernel : kernels)
                if (kernel.decl == def)
                    return &kernel;
        }
        return nullptr;
    }

    const ControlFunctionInfo* ClassInfo::get_control_function(const CXXMethodDecl* decl) const {
        if (auto def = decl->getDefinition()) {
            for (const ControlFunctionInfo& control_function : control_functions)
                if (control_function.decl == def)
                    return &control_function;
        }
        return nullptr;
    }

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

    void ClassInfo::dump() const {
        outs() << "Kernels (" << kernels.size() << "):\n";
        for (const KernelInfo& i : kernels) {
            outs() << i.decl->getNameAsString() << "\n";
        }

        outs() << "\nControl functions (" << control_functions.size() << "):\n";
        for (const ControlFunctionInfo& i : control_functions) {
            outs() << i.decl->getNameAsString() << "\n";
        }
    }

    ClassInfo::ClassInfo(const CXXRecordDecl* decl) {
        for (auto i : decl->methods()) {
            if (i->getNameAsString().substr(0, 6) == "kernel") {
                add_kernel(i);
            }
        }

        for (auto i : decl->methods()) {
            if (auto def = i->getDefinition()) {
                bool is_control = false;
                traverse_this_method_calls(def->getBody(), [&](const CXXMemberCallExpr* expr) {
                    if (get_kernel(expr->getMethodDecl()))
                        is_control = true;
                });
                if (is_control)
                    add_control_function(i);
            }
        }

        for (auto f : kernels) {
            traverse_this_field_accesses(f.decl->getBody(), [&](const MemberExpr* expr) {
                auto field = dyn_cast<FieldDecl>(expr->getMemberDecl());
                add_field(field);
            });
        }

        for (auto f : control_functions) {
            traverse_this_field_accesses(f.decl->getBody(), [&](const MemberExpr* expr) {
                auto field = dyn_cast<FieldDecl>(expr->getMemberDecl());
                add_field(field);
            });
        }

        for (auto f : used_functions) {
            traverse_this_field_accesses(f->getBody(), [&](const MemberExpr* expr) {
                auto field = dyn_cast<FieldDecl>(expr->getMemberDecl());
                add_field(field);
            });
            traverse_statement(f->getBody(), [&](const Stmt* stmt_) {
                if (auto stmt = dyn_cast<DeclStmt>(stmt_)) {
                    for (auto decl : stmt->decls()) {
                        add_type(dyn_cast<VarDecl>(decl)->getType().getCanonicalType());
                    }
                }
                if (auto expr = dyn_cast<DeclRefExpr>(stmt_)) {
                    auto decl = expr->getDecl();
                    if (auto var = dyn_cast<VarDecl>(decl)) {
                        if (var->isConstexpr()) {
                            note_found(decl->getLocation(), "global");
                            for (auto i : used_global_constants) {
                                if (i == decl) {
                                    return;
                                }
                            }
                            used_global_constants.push_back(var);
                        }
                    }
                }
            });
        }

        for (auto& i : kernels) {
            for (auto j : used_buffer_fields) {
                i.bindings.push_back(j);
            }
        }

        std::sort(used_uniform_fields.begin(), used_uniform_fields.end(), [](const FieldDecl* a, const FieldDecl* b) {
            return get_type_alignment(a->getType()) > get_type_alignment(b->getType());
        });
    }

    void ClassInfo::add_kernel(const CXXMethodDecl* decl) {
        decl = dyn_cast<CXXMethodDecl>(decl->getDefinition());
        if (get_kernel(decl))
            return;
        KernelInfo info{decl};
        for (size_t i = 0; i < decl->getNumParams(); i++) {
            auto p = decl->getParamDecl(i);
            if (p->getType()->isPointerType()) {
                info.bindings.push_back(p);
            }
        }

        for (auto stmt : decl->getBody()->children()) {
            if (auto for1 = dyn_cast<ForStmt>(stmt)) {
                info.index_vars.push_back(dyn_cast<VarDecl>(dyn_cast<DeclStmt>(for1->getInit())->getSingleDecl()));
                info.bounds_args.push_back(get_for_loop_bound(for1));
                info.body = for1->getBody();
                for (auto stmt : for1->getBody()->children()) {
                    if (auto for2 = dyn_cast<ForStmt>(stmt)) {
                        info.index_vars.push_back(
                            dyn_cast<VarDecl>(dyn_cast<DeclStmt>(for2->getInit())->getSingleDecl()));
                        info.bounds_args.push_back(get_for_loop_bound(for2));
                        info.body = for2->getBody();
                        for (auto stmt : for2->getBody()->children()) {
                            if (auto for3 = dyn_cast<ForStmt>(stmt)) {
                                info.index_vars.push_back(
                                    dyn_cast<VarDecl>(dyn_cast<DeclStmt>(for3->getInit())->getSingleDecl()));
                                info.bounds_args.push_back(get_for_loop_bound(for3));
                                info.body = for3->getBody();
                            }
                        }
                    }
                }
            }
        }

        auto add_used_function = [&](auto expr_) {
            const CallExpr* expr = expr_;
            auto decl = dyn_cast<FunctionDecl>(expr->getCalleeDecl());
            if (auto def = decl->getDefinition()) {
                std::string n = def->getNameAsString();
                if (n == "clamp" || n == "max" || n == "min" || n == "floor" || n == "abs" || n == "dot" ||
                    n == "length" || n == "normalize" || n == "exp" || n == "sqrt" || n == "copysign" || n == "sign" || n == "pow")
                    return;
                for (auto i : used_functions)
                    if (i == def)
                        return;
                used_functions.push_back(def);
            }
        };

        traverse_this_method_calls(decl->getBody(), add_used_function);
        traverse_global_function_calls(decl->getBody(), add_used_function);

        for (size_t i = 0; i < used_functions.size(); i++) {
            traverse_this_method_calls(used_functions[i]->getBody(), add_used_function);
            traverse_global_function_calls(used_functions[i]->getBody(), add_used_function);
        }

        // traverse_this_method_calls(decl->getBody(), [&](const CXXMemberCallExpr* expr) {
        //     auto def = expr->getMethodDecl()->getDefinition();
        //     for (auto i : used_functions)
        //         if (i == def)
        //             return;
        //     used_functions.push_back(def);
        // });

        // traverse_global_function_calls(decl->getBody(), [&](const CallExpr* expr) {
        //     auto def = expr->getCalleeDecl();
        //     for (auto i : used_functions)
        //         if (i == def)
        //             return;
        //     used_functions.push_back(def);
        // });

        // for (size_t i = 0; i < used_functions.size(); i++) {
        //     traverse_this_method_calls(used_functions[i]->getBody(), [&](){});
        // }

        kernels.push_back(info);
    }

    void ClassInfo::add_control_function(const CXXMethodDecl* decl) {
        decl = dyn_cast<CXXMethodDecl>(decl->getDefinition());
        if (get_control_function(decl))
            return;
        ControlFunctionInfo info{decl};

        traverse_this_method_calls(decl->getBody(), [&](const CXXMemberCallExpr* expr) {
            if (get_kernel(expr->getMethodDecl())) {
                info.kernel_calls.push_back(expr);
            }
        });

        control_functions.push_back(info);
    }

    void ClassInfo::add_field(const FieldDecl* decl) {
        if (is_vector_specialization(decl->getType())) {
            for (auto i : used_buffer_fields) {
                if (i == decl)
                    return;
            }
            used_buffer_fields.push_back(decl);
            add_type(get_vector_specialization_type(decl->getType()));
        } else {
            for (auto i : used_uniform_fields) {
                if (i == decl) {
                    return;
                }
            }
            used_uniform_fields.push_back(decl);
            add_type(decl->getType());
        }
    }

    void ClassInfo::add_type(QualType type) {
        if (auto r = dyn_cast<RecordDecl>(type->getAsRecordDecl())) {
            add_struct(r);
        }
    }

    void ClassInfo::add_struct(const RecordDecl* decl) {
        decl = dyn_cast<RecordDecl>(decl->getDefinition());
        {
            auto n = decl->getNameAsString();
            if (n[0] == 'f' && n[1] == 'l' && n[2] == 'o' && n[3] == 'a' && n[4] == 't')
                return;
            if (n[0] == 'i' && n[1] == 'n' && n[2] == 't')
                return;
            if (n[0] == 'u' && n[1] == 'i' && n[2] == 'n' && n[3] == 't')
                return;
        }
        outs() << "Struct: " << decl->getNameAsString() << "\n";
        for (auto i : used_structs) {
            if (i == decl) {
                return;
            }
        }
        used_structs.push_back(decl);

        for (auto i : decl->fields()) {
            add_type(i->getType());
        }
    }

} // namespace common_rewriter
