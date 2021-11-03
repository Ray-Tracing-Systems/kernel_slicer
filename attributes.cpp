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

struct ExampleAttrInfo : public ParsedAttrInfo {
  ExampleAttrInfo() {
    // Can take up to 15 optional arguments, to emulate accepting a variadic
    // number of arguments. This just illustrates how many arguments a
    // `ParsedAttrInfo` can hold, we will not use that much in this example.
    OptArgs = 15;
    // [[example]] and [[plugin::example]] supported.
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
    // // Check if the decl is at file scope.
    // if (!D->getDeclContext()->isFileContext()) 
    // {
    //   //unsigned ID = S.getDiagnostics().getCustomDiagID(DiagnosticsEngine::Error,"'setter' attribute only allowed at file scope");
    //   //S.Diag(Attr.getLoc(), ID);
    //   return AttributeNotApplied;
    // }

    // Attach an annotate attribute to the Decl.
    D->addAttr(AnnotateAttr::Create(S.Context, "setter", nullptr, 0, Attr.getRange()));
    
    return AttributeApplied;
  }
};

} // namespace

static ParsedAttrInfoRegistry::Add<ExampleAttrInfo> X("setter", "");
