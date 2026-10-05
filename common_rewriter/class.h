#pragma once
#include <clang/AST/AST.h>
#include <map>
#include <string>
#include <vector>

namespace common_rewriter {
    using namespace clang;

    extern std::vector<const CXXMethodDecl*> kernels;
    extern std::vector<const CXXMethodDecl*> controls;
    extern std::vector<const CXXMethodDecl*> methods;
    extern std::vector<const FieldDecl*> buffer_fields;
    extern std::vector<QualType> buffer_fields_elements_types;
    extern std::vector<const FieldDecl*> uniform_fields;
    extern std::map<const CXXMethodDecl*, std::vector<const ParmVarDecl*>> kernels_dimentions;
    extern std::map<const CXXMethodDecl*, std::vector<const CXXMethodDecl*>> controls_kernel_calls;
    extern std::map<const CXXMethodDecl*, std::vector<const ValueDecl*>> kernels_bindings;

    void discover_class(std::string class_name);

} // namespace common_rewriter
