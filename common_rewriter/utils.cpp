#include "utils.h"
#include <clang/AST/ASTContext.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Lex/Lexer.h>

namespace common_rewriter {
    using namespace clang;
    using llvm::outs;

    const clang::CompilerInstance* compiler_instance;

    std::string string_from_source_range(SourceRange range) {
        // https://stackoverflow.com/questions/11083066/getting-the-source-behind-clangs-ast
        const SourceManager& source_manager = compiler_instance->getSourceManager();
        clang::SourceLocation true_end(clang::Lexer::getLocForEndOfToken(range.getEnd(), 0, source_manager, {}));
        return std::string(source_manager.getCharacterData(range.getBegin()),
                           source_manager.getCharacterData(true_end) -
                               source_manager.getCharacterData(range.getBegin()));
    }

    const NamedDecl* find_global_declaration(std::string name) {
        TranslationUnitDecl* translation_unit_declaration_context =
            compiler_instance->getASTContext().getTranslationUnitDecl();
        IdentifierInfo& identifier_info = compiler_instance->getASTContext().Idents.get(name);
        DeclContext::lookup_result res = translation_unit_declaration_context->lookup(&identifier_info);

        if (res.empty()) {
            outs() << name << " is not found\n";
            exit(1);
        }
        if (!res.isSingleResult()) {
            outs() << name << " is not unique\n";
            exit(1);
        }
        return res.front();
    }

    // https://stackoverflow.com/questions/874134/find-out-if-string-ends-with-another-string-in-c
    bool ends_with(std::string_view str, std::string_view suffix) {
        return str.size() >= suffix.size() && str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
    }

    bool starts_with(std::string_view str, std::string_view prefix) {
        return str.size() >= prefix.size() && str.compare(0, prefix.size(), prefix) == 0;
    }

    void traverse_statement(const Stmt* stmt, std::function<void(const Stmt* stmt)> callback) {
        if (!stmt)
            return;
        callback(stmt);
        for (auto i : stmt->children())
            traverse_statement(i, callback);
    }

    void traverse_function_calls(const Stmt* stmt, std::function<void(const CallExpr*)> callback) {
        traverse_statement(stmt, [callback](const Stmt* stmt_) {
            if (auto expr = dyn_cast<CallExpr>(stmt_))
                callback(expr);
        });
    }

    void traverse_this_calls(const Stmt* stmt, std::function<void(const CXXMemberCallExpr*)> callback) {
        traverse_statement(stmt, [callback](const Stmt* stmt) {
            if (auto expr = dyn_cast<CXXMemberCallExpr>(stmt))
                if (dyn_cast<CXXThisExpr>(expr->getImplicitObjectArgument()))
                    callback(expr);
        });
    }

    void traverse_this_fields(const Stmt* stmt, std::function<void(const MemberExpr*)> callback) {
        traverse_statement(stmt, [callback](const Stmt* stmt) {
            if (auto expr = dyn_cast<MemberExpr>(stmt))
                if (dyn_cast<CXXThisExpr>(expr->getBase()) && dyn_cast<FieldDecl>(expr->getMemberDecl()))
                    callback(expr);
        });
    }

    bool is_vector_specialization(QualType type) {
        if (auto record = type->getAsRecordDecl()) {
            if (auto template_ = dyn_cast<ClassTemplateSpecializationDecl>(record)) {
                return true;
            }
        }
        return false;
    }

    QualType get_vector_specialization_type(QualType type) {
        auto record = type->getAsRecordDecl();
        REWRITE_ASSERT(record);
        auto template_ = dyn_cast<ClassTemplatePartialSpecializationDecl>(record);
        REWRITE_ASSERT(template_);
        QualType element_type = template_->getTemplateArgs().get(0).getAsType();
        return element_type;
    }

} // namespace common_rewriter
