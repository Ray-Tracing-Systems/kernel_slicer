#pragma once
#include <clang/AST/AST.h>
#include <map>
#include <vector>

namespace common_rewriter {
    using namespace clang;

    struct KernelInfo {
        const CXXMethodDecl* decl = nullptr;
        std::vector<const ParmVarDecl*> bounds_args;
        std::vector<const ValueDecl*> bindings;
        std::string shader;
    };

    struct ControlFunctionInfo {
        const CXXMethodDecl* decl = nullptr;
        std::vector<const CXXMemberCallExpr*> kernel_calls;
        std::string body;
    };

    struct ClassInfo {

        ClassInfo(const CXXRecordDecl* class_decl);
        void dump() const;

        std::vector<KernelInfo> kernels;
        const KernelInfo* get_kernel(const CXXMethodDecl* decl) const;

        std::vector<ControlFunctionInfo> control_functions;
        const ControlFunctionInfo* get_control_function(const CXXMethodDecl* decl) const;

        std::vector<const FieldDecl*> used_uniform_fields;
        std::vector<const FieldDecl*> used_buffer_fields;

    private:
        void add_kernel(const CXXMethodDecl* decl);
        void add_control_function(const CXXMethodDecl* decl);
    };

} // namespace common_rewriter
