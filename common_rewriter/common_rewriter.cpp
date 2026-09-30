#include "common_rewriter.h"
#include <clang/Frontend/CompilerInstance.h>
#include <clang/AST/ASTContext.h>
#include <clang/Lex/Lexer.h>
#include <fstream>

#define REWRITE_ASSERT(cond) \
    (!!(cond) ? (void)0 : (fprintf(stderr, "Assertion failed: %s\n", #cond), abort()));

namespace common_rewriter
{

    using namespace clang;
    using namespace llvm;

    const CompilerInstance *compiler_instance;

    std::vector<const CXXMethodDecl *> called_methods;

    void add_method(const CXXMethodDecl *method)
    {
        if (std::find(called_methods.begin(), called_methods.end(), method) == called_methods.end())
            called_methods.push_back(method);
    }

    std::vector<const FunctionDecl *> called_functions;

    void add_function(const FunctionDecl *function)
    {
        if (std::find(called_functions.begin(), called_functions.end(), function) == called_functions.end())
            called_functions.push_back(function);
    }

    std::string buffer_;
    bool do_emit_ = false;

    void emit(std::string s)
    {
        if (do_emit_)
            buffer_ += s;
    }

    void emit_indent(size_t indent)
    {
        for (size_t i = 0; i < indent; i++)
            emit("    ");
    }

    void emit(const char *s)
    {
        emit(std::string(s));
    }

    void emit(const llvm::StringRef &str)
    {
        emit(std::string(str));
    }

    const ASTContext &context()
    {
        return compiler_instance->getASTContext();
    }

    std::string string_from_source_range(SourceRange range)
    {
        // https://stackoverflow.com/questions/11083066/getting-the-source-behind-clangs-ast
        const SourceManager &source_manager = compiler_instance->getSourceManager();
        clang::SourceLocation true_end(clang::Lexer::getLocForEndOfToken(range.getEnd(), 0, source_manager, {}));
        return std::string(source_manager.getCharacterData(range.getBegin()), source_manager.getCharacterData(true_end) - source_manager.getCharacterData(range.getBegin()));
    }

    void note_found(SourceLocation loc, std::string what)
    {
        DiagnosticsEngine &diagnostic_engine = compiler_instance->getDiagnostics();
        uint32_t id = diagnostic_engine.getCustomDiagID(DiagnosticsEngine::Remark, "Found %0");
        diagnostic_engine.Report(loc, id) << what;
    }

    void error_unknown_stmt_class(const Stmt *stmt)
    {
        DiagnosticsEngine &diagnostic_engine = compiler_instance->getDiagnostics();
        uint32_t id = diagnostic_engine.getCustomDiagID(DiagnosticsEngine::Error, "Unknown statement kind: %0");
        diagnostic_engine.Report(stmt->getBeginLoc(), id) << stmt->getStmtClassName();
    }

    void error_expr_needs_cleanups(const ExprWithCleanups *expr)
    {
        DiagnosticsEngine &diagnostic_engine = compiler_instance->getDiagnostics();
        uint32_t id = diagnostic_engine.getCustomDiagID(DiagnosticsEngine::Error, "Expression needs cleanups");
        diagnostic_engine.Report(expr->getExprLoc(), id);
    }

    void error_unknown_overloaded_operator(const CXXOperatorCallExpr *expr)
    {
        DiagnosticsEngine &diagnostic_engine = compiler_instance->getDiagnostics();
        uint32_t id = diagnostic_engine.getCustomDiagID(DiagnosticsEngine::Error, "Unknown operator overload '%0'");
        diagnostic_engine.Report(expr->getExprLoc(), id) << getOperatorSpelling(expr->getOperator());
    }

    void error_can_only_access_locals(const Expr *expr)
    {
        DiagnosticsEngine &diagnostic_engine = compiler_instance->getDiagnostics();
        uint32_t id = diagnostic_engine.getCustomDiagID(DiagnosticsEngine::Error, "Trying to access non local variable");
        diagnostic_engine.Report(expr->getExprLoc(), id);
    }

    void error_can_only_call_methods_of_this(const Expr *expr)
    {
        DiagnosticsEngine &diagnostic_engine = compiler_instance->getDiagnostics();
        uint32_t id = diagnostic_engine.getCustomDiagID(DiagnosticsEngine::Error, "Can only call methods of this");
        diagnostic_engine.Report(expr->getExprLoc(), id);
    }

    void error_multiple_declarations(const DeclStmt *stmt)
    {
        DiagnosticsEngine &diagnostic_engine = compiler_instance->getDiagnostics();
        uint32_t id = diagnostic_engine.getCustomDiagID(DiagnosticsEngine::Error, "Multiple declarations are not supported for now");
        diagnostic_engine.Report(stmt->getBeginLoc(), id);
    }

    NamedDecl *find_decl_by_name(std::string name)
    {
        TranslationUnitDecl *translation_unit_declaration_context = context().getTranslationUnitDecl();
        IdentifierInfo &identifier_info = context().Idents.get(name);
        DeclContext::lookup_result res = translation_unit_declaration_context->lookup(&identifier_info);

        if (res.empty())
        {
            outs() << name << " is not found\n";
            exit(1);
        }
        if (!res.isSingleResult())
        {
            outs() << name << " is not unique\n";
            exit(1);
        }
        return res.front();
    }

    void rewrite_type(QualType type)
    {
        std::string s;
        llvm::raw_string_ostream ss(s);
        type.print(ss, {{}});
        emit(s);
    }

    void rewrite_expression(const Expr *expr_)
    {
        if (auto expr = dyn_cast<ExprWithCleanups>(expr_))
        {
            // Lets not do destroying of temporary variables
            if (expr->getNumObjects() > 0)
                error_expr_needs_cleanups(expr);
            rewrite_expression(expr->getSubExpr());
        }
        else if (auto expr = dyn_cast<FullExpr>(expr_))
        {
            rewrite_expression(expr->getSubExpr());
        }
        else if (auto expr = dyn_cast<ImplicitCastExpr>(expr_))
        {
            // Ignoring implicit casts
            return rewrite_expression(expr->getSubExpr());
        }
        else if (auto expr = dyn_cast<MaterializeTemporaryExpr>(expr_))
        {
            // I don't know what it is...
            return rewrite_expression(expr->getSubExpr());
        }
        else if (auto expr = dyn_cast<ParenExpr>(expr_))
        {
            emit("(");
            rewrite_expression(expr->getSubExpr());
            emit(")");
        }
        else if (auto expr = dyn_cast<CXXFunctionalCastExpr>(expr_))
        {
            rewrite_type(expr->getType());
            emit("(");
            rewrite_expression(expr->getSubExpr());
            emit(")");
        }
        else if (auto expr = dyn_cast<CXXConstructExpr>(expr_))
        {
            rewrite_type(expr->getType());
            emit("(");
            for (size_t i = 0; i < expr->getNumArgs(); i++)
            {
                rewrite_expression(expr->getArg(i));
                if (i < expr->getNumArgs() - 1)
                    emit(", ");
            }
            emit(")");
        }
        else if (auto expr = dyn_cast<BinaryOperator>(expr_))
        {
            rewrite_expression(expr->getLHS());
            emit(" ");
            emit(BinaryOperator::getOpcodeStr(expr->getOpcode()));
            emit(" ");
            rewrite_expression(expr->getRHS());
        }
        else if (auto expr = dyn_cast<UnaryOperator>(expr_))
        {
            emit(UnaryOperator::getOpcodeStr(expr->getOpcode()));
            rewrite_expression(expr->getSubExpr());
        }
        else if (auto expr = dyn_cast<CXXOperatorCallExpr>(expr_))
        {
            REWRITE_ASSERT(expr->getNumArgs() == 2);
            std::string s = getOperatorSpelling(expr->getOperator());
            if (s == "=" || s == "+" || s == "-" || s == "*" || s == "/" || s == "+=" || s == "-=" || s == "*=" || s == "/=")
            {
                rewrite_expression(expr->getArg(0));
                emit(" ");
                emit(s);
                emit(" ");
                rewrite_expression(expr->getArg(1));
            }
            else if (s == "[]")
            {
                rewrite_expression(expr->getArg(0));
                emit("[");
                rewrite_expression(expr->getArg(1));
                emit("]");
            }
            else
            {
                error_unknown_overloaded_operator(expr);
            }
        }
        else if (auto expr = dyn_cast<ConditionalOperator>(expr_))
        {
            rewrite_expression(expr->getCond());
            emit(" ? ");
            rewrite_expression(expr->getTrueExpr());
            emit(" : ");
            rewrite_expression(expr->getFalseExpr());
        }
        else if (auto expr = dyn_cast<IntegerLiteral>(expr_))
        {
            emit(string_from_source_range(expr->getSourceRange()));
        }
        else if (auto expr = dyn_cast<FloatingLiteral>(expr_))
        {
            emit(string_from_source_range(expr->getSourceRange()));
        }
        else if (auto expr = dyn_cast<DeclRefExpr>(expr_))
        {
            auto decl = expr->getDecl();
            emit(decl->getNameAsString());
        }
        else if (auto expr = dyn_cast<MemberExpr>(expr_))
        {
            if (dyn_cast<CXXThisExpr>(expr->getBase()))
                emit(expr->getMemberDecl()->getNameAsString());
            else
            {
                rewrite_expression(expr->getBase());
                // Accessing anonimous unions inside structs
                if (expr->getMemberDecl()->getNameAsString().length() > 0)
                {
                    emit(".");
                    emit(expr->getMemberDecl()->getNameAsString());
                }
            }
        }
        else if (auto expr = dyn_cast<CXXMemberCallExpr>(expr_))
        {

            if (!dyn_cast<CXXThisExpr>(expr->getImplicitObjectArgument()))
                error_can_only_call_methods_of_this(expr);

            add_method(expr->getMethodDecl());
            rewrite_expression(expr->getCallee());
            emit("(");
            for (int i = 0; i < expr->getNumArgs(); i++)
            {
                rewrite_expression(expr->getArg(i));
                if (i < expr->getNumArgs() - 1)
                    emit(", ");
            }
            emit(")");
        }
        else if (auto expr = dyn_cast<CallExpr>(expr_))
        {
            auto func = dyn_cast<FunctionDecl>(expr->getCalleeDecl());
            REWRITE_ASSERT(func);
            add_function(func);
            emit(func->getNameAsString());
            emit("(");
            for (int i = 0; i < expr->getNumArgs(); i++)
            {
                rewrite_expression(expr->getArg(i));
                if (i < expr->getNumArgs() - 1)
                    emit(", ");
            }
            emit(")");
        }
        else
        {
            error_unknown_stmt_class(expr_);
        }
    }

    void rewrite_statement(const Stmt *stmt_, size_t indent)
    {
        if (auto stmt = dyn_cast<CompoundStmt>(stmt_))
        {
            emit_indent(indent);
            emit("{\n");
            for (auto i : stmt->children())
            {
                rewrite_statement(i, indent + 1);
            }
            emit_indent(indent);
            emit("}\n");
        }
        else if (auto stmt = dyn_cast<Expr>(stmt_))
        {
            emit_indent(indent);
            rewrite_expression(stmt);
            emit(";\n");
        }
        else if (auto stmt = dyn_cast<DeclStmt>(stmt_))
        {

            for (auto i : stmt->getDeclGroup())
            {
                auto var = dyn_cast<VarDecl>(i);
                REWRITE_ASSERT(var);
                emit_indent(indent);
                rewrite_type(var->getType());
                emit(" ");
                emit(var->getNameAsString());
                emit(" = ");
                rewrite_expression(var->getInit());
                emit(";\n");
            }
        }
        else if (auto stmt = dyn_cast<IfStmt>(stmt_))
        {
            emit_indent(indent);
            emit("if (");
            rewrite_expression(stmt->getCond());
            emit(")\n");
            rewrite_statement(stmt->getThen(), indent);
            if (stmt->getElse())
            {
                emit_indent(indent);
                emit("else\n");
                rewrite_statement(stmt->getElse(), indent);
            }
        }
        else if (auto stmt = dyn_cast<WhileStmt>(stmt_))
        {
            emit_indent(indent);
            emit("while (");
            rewrite_expression(stmt->getCond());
            emit(")\n");
            rewrite_statement(stmt->getBody(), indent);
        }
        else if (auto stmt = dyn_cast<ForStmt>(stmt_))
        {
            emit_indent(indent);
            emit("for (");
            rewrite_statement(stmt->getInit(), 0);
            rewrite_expression(stmt->getCond());
            emit("; ");
            rewrite_expression(stmt->getInc());
            emit(")\n");
            rewrite_statement(stmt->getBody(), indent + 1);
        }
        else if (auto stmt = dyn_cast<ReturnStmt>(stmt_))
        {
            emit_indent(indent);
            emit("return ");
            if (stmt->getRetValue())
                rewrite_expression(stmt->getRetValue());
            emit(";\n");
        }
        else
        {
            error_unknown_stmt_class(stmt_);
        }
    }

    void rewrite_method(const CXXMethodDecl *method)
    {
        emit("void ");
        emit(method->getNameAsString());
        emit("()\n");
        rewrite_statement(method->getBody(), 0);
    }

    void rewrite_function(const FunctionDecl *function)
    {
        emit("void ");
        emit(function->getNameAsString());
        emit("()\n");
        if (function->getBody())
            rewrite_statement(function->getBody(), 0);
        else
            emit(";\n");
    }

    void rewrite_kernel(std::string class_name, std::string kernel_name)
    {
        do_emit_ = true;
        if (auto class_decl = dyn_cast<CXXRecordDecl>(find_decl_by_name(class_name)))
        {
            note_found(class_decl->getLocation(), "class");
            for (auto method_decl : class_decl->methods())
            {
                if (method_decl->getNameAsString() == kernel_name)
                {
                    note_found(method_decl->getLocation(), "kernel");
                    for (auto i : method_decl->getBody()->children())
                    {
                        if (auto for_stmt = dyn_cast<ForStmt>(i))
                        {
                            note_found(for_stmt->getForLoc(), "root for loop");
                            rewrite_statement(for_stmt->getBody(), 0);
                        }
                    }
                }
            }
        }

        for (size_t i = 0; i < called_methods.size(); i++)
        {
            outs() << "Method: " << called_methods[i]->getNameAsString() << "\n";
            rewrite_method(called_methods[i]);
        }

        for (size_t i = 0; i < called_functions.size(); i++)
        {
            outs() << "Function: " << called_functions[i]->getNameAsString() << "\n";
            rewrite_function(called_functions[i]);
        }

        outs() << buffer_ << "\n";
    }
}
