#include "kslicer.h"

#include <sstream>
#include <algorithm>

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool kslicer::FunctionRewriter2::VisitFunctionDecl_Impl(clang::FunctionDecl* fDecl)    { return true; }
bool kslicer::FunctionRewriter2::VisitCXXMethodDecl_Impl(clang::CXXMethodDecl* fDecl)  { return true; }
bool kslicer::FunctionRewriter2::VisitVarDecl_Impl(clang::VarDecl* decl)               { return true; }
bool kslicer::FunctionRewriter2::VisitDeclStmt_Impl(clang::DeclStmt* decl)             { return true; }
bool kslicer::FunctionRewriter2::VisitMemberExpr_Impl(clang::MemberExpr* expr)                   { return true; }
bool kslicer::FunctionRewriter2::VisitCXXMemberCallExpr_Impl(clang::CXXMemberCallExpr* f)        { return true; } 
bool kslicer::FunctionRewriter2::VisitFieldDecl_Impl(clang::FieldDecl* decl)                     { return true; }
bool kslicer::FunctionRewriter2::VisitUnaryOperator_Impl(clang::UnaryOperator* op)               { return true; }
bool kslicer::FunctionRewriter2::VisitCStyleCastExpr_Impl(clang::CStyleCastExpr* cast)           { return true; }
bool kslicer::FunctionRewriter2::VisitImplicitCastExpr_Impl(clang::ImplicitCastExpr* cast)       { return true; }
bool kslicer::FunctionRewriter2::VisitCXXConstructExpr_Impl(clang::CXXConstructExpr* call)       { return true; } 
bool kslicer::FunctionRewriter2::VisitCXXOperatorCallExpr_Impl(clang::CXXOperatorCallExpr* expr) { return true; }
bool kslicer::FunctionRewriter2::VisitArraySubscriptExpr_Impl(clang::ArraySubscriptExpr* arrayExpr)            { return true; }
bool kslicer::FunctionRewriter2::VisitUnaryExprOrTypeTraitExpr_Impl(clang::UnaryExprOrTypeTraitExpr* szOfExpr) { return true; }
bool kslicer::FunctionRewriter2::VisitCallExpr_Impl(clang::CallExpr* f)                                        { return true; }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool kslicer::KernelRewriter2::VisitUnaryOperator_Impl(clang::UnaryOperator* expr) 
{
  m_pFunRW2->m_kernelMode = true;
  return m_pFunRW2->VisitUnaryOperator_Impl(expr);
}

bool kslicer::KernelRewriter2::VisitCompoundAssignOperator_Impl(clang::CompoundAssignOperator* expr) 
{
  m_pFunRW2->m_kernelMode = true;
  return true; //m_pFunRW2->VisitCompoundAssignOperator_Impl(expr);
}

bool kslicer::KernelRewriter2::VisitCXXOperatorCallExpr_Impl(clang::CXXOperatorCallExpr* expr)
{
  m_pFunRW2->m_kernelMode = true;
  return m_pFunRW2->VisitCXXOperatorCallExpr_Impl(expr);
}

bool kslicer::KernelRewriter2::VisitBinaryOperator_Impl(clang::BinaryOperator* expr)                
{
  m_pFunRW2->m_kernelMode = true;
  return true; // m_pFunRW2->VisitBinaryOperator_Impl(expr);
}

bool kslicer::KernelRewriter2::VisitVarDecl_Impl(clang::VarDecl* decl) 
{
  m_pFunRW2->m_kernelMode = true;
  return m_pFunRW2->VisitVarDecl_Impl(decl);
}

bool kslicer::KernelRewriter2::VisitCStyleCastExpr_Impl(clang::CStyleCastExpr* cast)
{
  m_pFunRW2->m_kernelMode = true;
  return m_pFunRW2->VisitCStyleCastExpr_Impl(cast);
}

bool kslicer::KernelRewriter2::VisitImplicitCastExpr_Impl(clang::ImplicitCastExpr* cast)            
{
  m_pFunRW2->m_kernelMode = true;
  return m_pFunRW2->VisitImplicitCastExpr_Impl(cast);
}

bool kslicer::KernelRewriter2::VisitDeclRefExpr_Impl(clang::DeclRefExpr* expr) 
{
  m_pFunRW2->m_kernelMode = true;
  return true; // m_pFunRW2->VisitDeclRefExpr_Impl(expr);
}

bool kslicer::KernelRewriter2::VisitDeclStmt_Impl(clang::DeclStmt* decl) 
{
  m_pFunRW2->m_kernelMode = true;
  return m_pFunRW2->VisitDeclStmt_Impl(decl);
}

bool kslicer::KernelRewriter2::VisitArraySubscriptExpr_Impl(clang::ArraySubscriptExpr* arrayExpr)
{
  m_pFunRW2->m_kernelMode = true;
  return m_pFunRW2->VisitArraySubscriptExpr_Impl(arrayExpr);
}

bool kslicer::KernelRewriter2::VisitUnaryExprOrTypeTraitExpr_Impl(clang::UnaryExprOrTypeTraitExpr* szOfExpr)
{
  m_pFunRW2->m_kernelMode = true;
  return m_pFunRW2->VisitUnaryExprOrTypeTraitExpr_Impl(szOfExpr);
}
