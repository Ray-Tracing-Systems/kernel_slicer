#include "errors.h"
#include "utils.h"
#include <clang/Frontend/CompilerInstance.h>

namespace common_rewriter {
    using namespace clang;

    void note_found(SourceLocation loc, std::string what) {
        DiagnosticsEngine& diagnostic_engine = compiler_instance->getDiagnostics();
        uint32_t id = diagnostic_engine.getCustomDiagID(DiagnosticsEngine::Remark, "Found %0");
        diagnostic_engine.Report(loc, id) << what;
    }

    void error_unknown_stmt_class(const Stmt* stmt) {
        DiagnosticsEngine& diagnostic_engine = compiler_instance->getDiagnostics();
        uint32_t id = diagnostic_engine.getCustomDiagID(DiagnosticsEngine::Error, "Unknown statement kind: %0");
        diagnostic_engine.Report(stmt->getBeginLoc(), id) << stmt->getStmtClassName();
    }

    void error_expr_needs_cleanups(const ExprWithCleanups* expr) {
        DiagnosticsEngine& diagnostic_engine = compiler_instance->getDiagnostics();
        uint32_t id = diagnostic_engine.getCustomDiagID(DiagnosticsEngine::Error, "Expression needs cleanups");
        diagnostic_engine.Report(expr->getExprLoc(), id);
    }

    void error_unknown_overloaded_operator(const CXXOperatorCallExpr* expr) {
        DiagnosticsEngine& diagnostic_engine = compiler_instance->getDiagnostics();
        uint32_t id = diagnostic_engine.getCustomDiagID(DiagnosticsEngine::Error, "Unknown operator overload '%0'");
        diagnostic_engine.Report(expr->getExprLoc(), id) << getOperatorSpelling(expr->getOperator());
    }

    void error_can_only_access_locals(const Expr* expr) {
        DiagnosticsEngine& diagnostic_engine = compiler_instance->getDiagnostics();
        uint32_t id =
            diagnostic_engine.getCustomDiagID(DiagnosticsEngine::Error, "Trying to access non local variable");
        diagnostic_engine.Report(expr->getExprLoc(), id);
    }

    void error_can_only_call_methods_of_this(const Expr* expr) {
        DiagnosticsEngine& diagnostic_engine = compiler_instance->getDiagnostics();
        uint32_t id = diagnostic_engine.getCustomDiagID(DiagnosticsEngine::Error, "Can only call methods of this");
        diagnostic_engine.Report(expr->getExprLoc(), id);
    }

    void error_function_without_definition(const FunctionDecl* function) {
        DiagnosticsEngine& diagnostic_engine = compiler_instance->getDiagnostics();
        uint32_t id = diagnostic_engine.getCustomDiagID(DiagnosticsEngine::Error, "Function without definition");
        diagnostic_engine.Report(function->getLocation(), id);
    }

} // namespace common_rewriter
