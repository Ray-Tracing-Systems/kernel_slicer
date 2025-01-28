#include "kslicer.h"

#include <sstream>
#include <algorithm>

void kslicer::FunctionRewriter2::InitKernelData(kslicer::KernelInfo& a_kernelRef)
{
  m_kernelMode  = true;
  m_pCurrKernel = &a_kernelRef;
  for(auto arg : a_kernelRef.args)
  {
    if(arg.isLoopSize || arg.IsUser())
      m_kernelUserArgs.insert(arg.name);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool kslicer::FunctionRewriter2::VisitFunctionDecl_Impl(clang::FunctionDecl* fDecl)        
{
  auto hash = kslicer::GetHashOfSourceRange(fDecl->getBody()->getSourceRange());
  if(m_codeInfo->m_functionsDone.find(hash) == m_codeInfo->m_functionsDone.end()) // it is important to put functions in 'm_functionsDone'
  {
    kslicer::RewrittenFunction done;
    done.funDecl = kslicer::GetRangeSourceCode(fDecl->getSourceRange(),            m_compiler); 
    auto posBrace = done.funDecl.find("{");
    if(posBrace != std::string::npos)
      done.funDecl = done.funDecl.substr(0,posBrace); // discard func body source code
    done.funBody = kslicer::GetRangeSourceCode(fDecl->getBody()->getSourceRange(), m_compiler);
    m_codeInfo->m_functionsDone[hash] = done;
  }  
  return true; 
}

bool kslicer::FunctionRewriter2::VisitCXXMethodDecl_Impl(clang::CXXMethodDecl* fDecl)      { return true; }
bool kslicer::FunctionRewriter2::VisitMemberExpr_Impl(clang::MemberExpr* expr)             { return true; }
bool kslicer::FunctionRewriter2::VisitCXXMemberCallExpr_Impl(clang::CXXMemberCallExpr* f)  { return true; } 
bool kslicer::FunctionRewriter2::VisitFieldDecl_Impl(clang::FieldDecl* decl)               { return true; }
bool kslicer::FunctionRewriter2::VisitCXXConstructExpr_Impl(clang::CXXConstructExpr* call) { return true; } 
bool kslicer::FunctionRewriter2::VisitCallExpr_Impl(clang::CallExpr* f)                    { return true; }

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// unually both for kernel and functions
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// unually both for kernel and functions

bool kslicer::FunctionRewriter2::VisitUnaryOperator_Impl(clang::UnaryOperator* op)
{ 
  if(m_kernelMode)
  {
    // ...
  }

  return true; 
}

bool kslicer::FunctionRewriter2::VisitCXXOperatorCallExpr_Impl(clang::CXXOperatorCallExpr* expr) 
{ 
  if(m_kernelMode)
  {
    // ...
  }

  return true; 
}

bool kslicer::FunctionRewriter2::VisitVarDecl_Impl(clang::VarDecl* decl) 
{ 
  if(m_kernelMode)
  {
    // ...
  }

  return true; 
}

bool kslicer::FunctionRewriter2::VisitCStyleCastExpr_Impl(clang::CStyleCastExpr* cast)           
{ 
  if(m_kernelMode)
  {
    // ...
  }

  return true;   
}

bool kslicer::FunctionRewriter2::VisitImplicitCastExpr_Impl(clang::ImplicitCastExpr* cast) 
{ 
  if(m_kernelMode)
  {
    // ...
  }

  return true; 
}

bool kslicer::FunctionRewriter2::VisitDeclStmt_Impl(clang::DeclStmt* decl)             
{ 
  if(m_kernelMode)
  {
    // ...
  }

  return true; 
}

bool kslicer::FunctionRewriter2::VisitArraySubscriptExpr_Impl(clang::ArraySubscriptExpr* arrayExpr) 
{ 
  if(m_kernelMode)
  {
    // ...
  }

  return true; 
}

bool kslicer::FunctionRewriter2::VisitUnaryExprOrTypeTraitExpr_Impl(clang::UnaryExprOrTypeTraitExpr* szOfExpr) 
{ 
  if(m_kernelMode)
  {
    // ...
  }

  return true; 
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// usually only for kernel
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// usually only for kernel

bool kslicer::FunctionRewriter2::VisitCompoundAssignOperator_Impl(clang::CompoundAssignOperator* expr) 
{ 
  if(m_kernelMode)
  {
    // ...
  }

  return true; 
}

bool kslicer::FunctionRewriter2::VisitBinaryOperator_Impl(clang::BinaryOperator* expr)
{
  if(m_kernelMode)
  {
    // ...
  }

  return true;
}

bool  kslicer::FunctionRewriter2::VisitDeclRefExpr_Impl(clang::DeclRefExpr* expr) 
{
  if(m_kernelMode)
  {
    // ...
  }

  return true;
}

std::string kslicer::FunctionRewriter2::RecursiveRewrite(const clang::Stmt* expr)
{
  if(expr == nullptr)
    return "";
    
  FunctionRewriter2 rvCopy = *this;
  rvCopy.TraverseStmt(const_cast<clang::Stmt*>(expr));

  auto range = expr->getSourceRange();
  auto p = rvCopy.m_workAround.find(GetHashOfSourceRange(range));
  if(p != rvCopy.m_workAround.end())
    return p->second;
  else
    return m_rewriter.getRewrittenText(range);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool kslicer::KernelRewriter2::VisitUnaryOperator_Impl(clang::UnaryOperator* expr)                   { return m_pFunRW2->VisitUnaryOperator_Impl(expr); }
bool kslicer::KernelRewriter2::VisitCompoundAssignOperator_Impl(clang::CompoundAssignOperator* expr) { return m_pFunRW2->VisitCompoundAssignOperator_Impl(expr); }
bool kslicer::KernelRewriter2::VisitCXXOperatorCallExpr_Impl(clang::CXXOperatorCallExpr* expr)       { return m_pFunRW2->VisitCXXOperatorCallExpr_Impl(expr); }

bool kslicer::KernelRewriter2::VisitBinaryOperator_Impl(clang::BinaryOperator* expr)     { return m_pFunRW2->VisitBinaryOperator_Impl(expr); }
bool kslicer::KernelRewriter2::VisitVarDecl_Impl(clang::VarDecl* decl)                   { return m_pFunRW2->VisitVarDecl_Impl(decl); }
bool kslicer::KernelRewriter2::VisitCStyleCastExpr_Impl(clang::CStyleCastExpr* cast)     { return m_pFunRW2->VisitCStyleCastExpr_Impl(cast); }
bool kslicer::KernelRewriter2::VisitImplicitCastExpr_Impl(clang::ImplicitCastExpr* cast) { return m_pFunRW2->VisitImplicitCastExpr_Impl(cast); }

bool kslicer::KernelRewriter2::VisitDeclRefExpr_Impl(clang::DeclRefExpr* expr)                    { return m_pFunRW2->VisitDeclRefExpr_Impl(expr); }
bool kslicer::KernelRewriter2::VisitDeclStmt_Impl(clang::DeclStmt* decl)                          { return m_pFunRW2->VisitDeclStmt_Impl(decl); }
bool kslicer::KernelRewriter2::VisitArraySubscriptExpr_Impl(clang::ArraySubscriptExpr* arrayExpr) { return m_pFunRW2->VisitArraySubscriptExpr_Impl(arrayExpr); }
bool kslicer::KernelRewriter2::VisitUnaryExprOrTypeTraitExpr_Impl(clang::UnaryExprOrTypeTraitExpr* szOfExpr) { return m_pFunRW2->VisitUnaryExprOrTypeTraitExpr_Impl(szOfExpr); }
