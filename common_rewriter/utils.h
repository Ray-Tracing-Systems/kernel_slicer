#pragma once
#include <clang/AST/AST.h>
#include <functional>
#include <string>
#include <vector>

#define REWRITE_ASSERT(cond) (!!(cond) ? (void)0 : (fprintf(stderr, "Assertion failed: %s\n", #cond), abort()))

namespace clang {
    class CompilerInstance;
} // namespace clang

namespace common_rewriter {
    using namespace clang;

    extern const clang::CompilerInstance* compiler_instance;

    std::string string_from_source_range(SourceRange range);
    const NamedDecl* find_global_declaration(std::string name);

    // template <typename T, typename U>
    // bool contains(const std::vector<T>& v, const U& value) {
    //     for (const auto& i : v)
    //         if (i == value)
    //             return true;
    //     return false;
    // }

    // template <typename T, typename U>
    // void push_back_unique(std::vector<T>& v, const U& value) {
    //     if (!contains(v, value))
    //         v.push_back(value);
    // }

    // https://stackoverflow.com/questions/874134/find-out-if-string-ends-with-another-string-in-c
    bool starts_with(std::string_view str, std::string_view prefix);
    bool ends_with(std::string_view str, std::string_view suffix);

    void traverse_statement(const Stmt* stmt, std::function<void(const Stmt* stmt)> callback);
    void traverse_global_function_calls(const Stmt* stmt, std::function<void(const CallExpr*)> callback);
    void traverse_this_method_calls(const Stmt* stmt, std::function<void(const CXXMemberCallExpr*)> callback);
    void traverse_this_field_accesses(const Stmt* stmt, std::function<void(const MemberExpr*)> callback);

    bool is_vector_specialization(QualType type);
    QualType get_vector_specialization_type(QualType type);

    std::string render_type_name(QualType type);
    size_t get_type_array_size(QualType type);
    bool is_builtin_type(QualType type);
    size_t get_type_alignment(QualType type);

} // namespace common_rewriter
