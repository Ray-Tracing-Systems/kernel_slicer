#ifndef KSLICER_CLANG_COMPAT_H
#define KSLICER_CLANG_COMPAT_H

// Compatibility layer between LLVM/Clang 17 (default) and LLVM/Clang 22.1 (-DUSE_LLVM22=ON).

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Type.h"
#include "llvm/Support/Casting.h"

namespace kslicer
{
  /// Replacement for clang::TypeDecl::getTypeForDecl().
  /// In Clang 22 getTypeForDecl() is no longer available for TagDecl (struct/class/union/enum);
  /// the type must be obtained from ASTContext instead.
  inline const clang::Type* GetTypeForDecl(const clang::TypeDecl* a_decl)
  {
    if(a_decl == nullptr)
      return nullptr;
#ifdef KSLICER_USE_LLVM22
    if(const auto* tagDecl = llvm::dyn_cast<clang::TagDecl>(a_decl))
      return tagDecl->getASTContext().getCanonicalTagType(tagDecl).getTypePtr();
#endif
    return a_decl->getTypeForDecl();
  }
}

#endif
