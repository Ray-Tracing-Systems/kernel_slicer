//===- Attribute.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Example clang plugin which adds an an annotation to file-scope declarations
// with the 'example' attribute.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/Sema/ParsedAttr.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "llvm/IR/Attributes.h"
#include "clang/Frontend/CompilerInstance.h"
using namespace clang;

namespace kslicer
{
  std::string GetRangeSourceCode(const clang::SourceRange a_range, const clang::CompilerInstance& compiler); 
}

namespace {

struct SetterAttrInfo : public ParsedAttrInfo {
  SetterAttrInfo() {
    OptArgs = 1;
    static constexpr Spelling S[] = {{ParsedAttr::AS_CXX11, "kslicer::setter"}}; // {ParsedAttr::AS_CXX11, "setter"},
    Spellings = S;
  }

  bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override 
  {
    // This attribute appertains to functions only.
    if (!isa<FunctionDecl>(D)) {
      S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
          << Attr << "functions";
      return false;
    }
    return true;
  }

  AttrHandling handleDeclAttribute(Sema &S, Decl *D, const ParsedAttr &Attr) const override 
  {
    // Attach an annotate attribute to the Decl.
    D->addAttr(AnnotateAttr::Create(S.Context, "setter", nullptr, 0, Attr.getRange()));
    return AttributeApplied;
  }
};


struct Kernel1DAttrInfo : public ParsedAttrInfo {
  Kernel1DAttrInfo() {
    OptArgs = 1;
    static constexpr Spelling S[] = {{ParsedAttr::AS_CXX11, "kslicer::kernel1D"}}; // {ParsedAttr::AS_CXX11, "setter"},
    Spellings = S;
  }

  bool diagAppertainsToDecl(Sema &S, const ParsedAttr &Attr, const Decl *D) const override 
  {
    // This attribute appertains to functions only.
    if (!isa<FunctionDecl>(D)) {
      S.Diag(Attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
          << Attr << "functions";
      return false;
    }
    return true;
  }

  AttrHandling handleDeclAttribute(Sema &S, Decl *D, const ParsedAttr &Attr) const override 
  {
    // Attach an annotate attribute to the Decl.
    D->addAttr(AnnotateAttr::Create(S.Context, "kernel1D", nullptr, 0, Attr.getRange()));
    return AttributeApplied;
  }
};

} // namespace

static ParsedAttrInfoRegistry::Add<SetterAttrInfo> X("setter", "");
static ParsedAttrInfoRegistry::Add<Kernel1DAttrInfo> Y("kernel1D", "");
