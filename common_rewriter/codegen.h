#pragma once
#include "class_info.h"
#include <clang/AST/AST.h>

namespace common_rewriter {
    using namespace clang;

    class Codegen {
    public:
        Codegen() = default;
        virtual ~Codegen() = default;

        std::string buffer;

        void render_type(QualType type);
        void render_expression(const Expr* expr);
        void render_statement(const Stmt* stmt, size_t indent, bool newline = true);
        void render_function(const FunctionDecl* function);

    protected:
        virtual void render_function_param(const ParmVarDecl* param);
        virtual void render_field_access(const MemberExpr* expr_);
        virtual void render_method_call(const CXXMemberCallExpr* expr_);

        void emit_indent(size_t indent);
        void emit(std::string str);
        void emit(llvm::StringRef str);
        void emit(const char*);
    };

    class ControlFunctionCodegen : public Codegen {
    public:
        ControlFunctionCodegen(const ClassInfo& info, const CXXMethodDecl* decl, size_t ds_index)
            : info_(info), decl_(decl), ds_index_(ds_index) {}

    private:
        void render_method_call(const CXXMemberCallExpr* expr) override;

        const ClassInfo& info_;
        const CXXMethodDecl* decl_;
        size_t ds_index_;
    };

} // namespace common_rewriter
