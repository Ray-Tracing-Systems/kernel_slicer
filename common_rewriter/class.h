#pragma once
#include <string>
#include <vector>

namespace clang {
    class FieldDecl;
    class CXXMethodDecl;
    class QualType;

} // namespace clang

namespace common_rewriter {
    extern std::vector<const clang::CXXMethodDecl*> kernels;
    extern std::vector<const clang::CXXMethodDecl*> controls;
    extern std::vector<const clang::CXXMethodDecl*> methods;
    extern std::vector<const clang::FieldDecl*> buffer_fields;
    extern std::vector<clang::QualType> buffer_fields_elements_types;
    extern std::vector<const clang::FieldDecl*> uniform_fields;

    void discover_class(std::string class_name);

} // namespace common_rewriter
