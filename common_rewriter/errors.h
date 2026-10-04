#pragma once
#include <clang/Basic/SourceLocation.h>

namespace clang {
    class Stmt;
    class ExprWithCleanups;
    class CXXOperatorCallExpr;
    class Expr;
    class FunctionDecl;
} // namespace clang

namespace common_rewriter {
    using namespace clang;

    void note_found(SourceLocation loc, std::string what);

    void error_unknown_stmt_class(const Stmt* stmt);

    void error_expr_needs_cleanups(const ExprWithCleanups* expr);

    void error_unknown_overloaded_operator(const CXXOperatorCallExpr* expr);

    void error_can_only_access_locals(const Expr* expr);

    void error_can_only_call_methods_of_this(const Expr* expr);

    void error_function_without_definition(const FunctionDecl* function);

} // namespace common_rewriter
