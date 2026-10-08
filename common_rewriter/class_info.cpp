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
                info.bounds_args.push_back(get_for_loop_bound(for1));
                for (auto stmt : for1->getBody()->children()) {
                    if (auto for2 = dyn_cast<ForStmt>(stmt)) {
                        info.bounds_args.push_back(get_for_loop_bound(for2));
                        for (auto stmt : for2->getBody()->children()) {
                            if (auto for3 = dyn_cast<ForStmt>(stmt)) {
                                info.bounds_args.push_back(get_for_loop_bound(for3));
                            }
                        }
                    }
                }
            }
        }

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

} // namespace common_rewriter
