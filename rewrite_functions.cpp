#include "kslicer.h"
#include "class_gen.h"
#include "ast_matchers.h"

#include <sstream>
#include <algorithm>

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::string kslicer::FunctionRewriter::FunctionCallRewrite(const CallExpr* call)
{
  const FunctionDecl* fDecl = call->getDirectCallee();  
  if(fDecl == nullptr)             // definitely can't process nullpointer 
    return "[FunctionRewriter::FunctionCallRewrite_ERROR]";
  
  std::string argsType = "";
  if(call->getNumArgs() > 0)
  {
    const Expr* firstArgExpr = call->getArgs()[0];
    const QualType qt        = firstArgExpr->getType();
    argsType                 = qt.getAsString();
  }
  
  std::string fname   = fDecl->getNameInfo().getName().getAsString();
  std::string textRes = m_codeInfo->pShaderCC->ReplaceCallFromStdNamespace(fname, argsType);

  //std::string textRes = fDecl->getNameInfo().getName().getAsString();

  textRes += "(";
  for(unsigned i=0;i<call->getNumArgs();i++)
  {
    textRes += RecursiveRewrite(call->getArg(i));
    if(i < call->getNumArgs()-1)
      textRes += ",";
  }
  textRes += ")";

  return textRes;
}

std::string kslicer::FunctionRewriter::FunctionCallRewriteNoName(const CXXConstructExpr* call)
{
  std::string textRes = "(";
  for(unsigned i=0;i<call->getNumArgs();i++)
  {
    textRes += RecursiveRewrite(call->getArg(i));
    if(i < call->getNumArgs()-1)
      textRes += ",";
  }
  return textRes + ")";
}

std::string kslicer::FunctionRewriter::FunctionCallRewrite(const CXXConstructExpr* call)
{
  std::string textRes = call->getConstructor()->getNameInfo().getName().getAsString();
  return textRes + FunctionCallRewriteNoName(call);
}

bool kslicer::FunctionRewriter::VisitCallExpr_Impl(CallExpr* call)
{
  if(isa<CXXMemberCallExpr>(call)) // process in VisitCXXMemberCallExpr
    return true;

  const FunctionDecl* fDecl = call->getDirectCallee();  
  if(fDecl == nullptr)             // definitely can't process nullpointer 
    return true;

  // Get name of function
  //
  std::string fname = fDecl->getNameInfo().getName().getAsString();

  if(fDecl->isInStdNamespace())
  {
    std::string argsType = "";
    if(call->getNumArgs() > 0)
    {
      const Expr* firstArgExpr = call->getArgs()[0];
      const QualType qt        = firstArgExpr->getType();
      argsType                 = qt.getAsString();
    }
    
    if(WasNotRewrittenYet(call))
    { 
      auto debugMeIn = GetRangeSourceCode(call->getSourceRange(), m_compiler);     
      auto textRes   = FunctionCallRewrite(call);
      m_rewriter.ReplaceText(call->getSourceRange(), textRes);
      MarkRewritten(call);
      //std::cout << "  " << text.c_str() << " of type " << argsType.c_str() << "; --> " <<  textRes.c_str() << std::endl;
    }
  }
 
  return true;
}

std::string kslicer::FunctionRewriter::VectorTypeContructorReplace(const std::string& fname, const std::string& callText)
{
  return std::string("make_") + fname + callText;
}

bool kslicer::FunctionRewriter::VisitCXXConstructExpr_Impl(CXXConstructExpr* call)
{
  CXXConstructorDecl* ctorDecl = call->getConstructor();
  assert(ctorDecl != nullptr);
  
  //const std::string debugText = GetRangeSourceCode(call->getSourceRange(), m_compiler);   
  const std::string fname = ctorDecl->getNameInfo().getName().getAsString();
  
  //call->dump();
  if(kslicer::IsVectorContructorNeedsReplacement(fname) && WasNotRewrittenYet(call) && !ctorDecl->isCopyOrMoveConstructor() && call->getNumArgs() > 0 ) //
  {
    const std::string text    = FunctionCallRewriteNoName(call);
    const std::string textRes = VectorTypeContructorReplace(fname, text); 
    m_rewriter.ReplaceText(call->getSourceRange(), textRes);
    MarkRewritten(call);
  }

  return true;
}

std::string kslicer::FunctionRewriter::RecursiveRewrite(const Stmt* expr)
{
  if(expr == nullptr)
    return "";
  FunctionRewriter rvCopy = *this;
  rvCopy.TraverseStmt(const_cast<clang::Stmt*>(expr));
  return m_rewriter.getRewrittenText(expr->getSourceRange());
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool kslicer::NodesMarker::VisitStmt(Stmt* expr)
{
  auto hash = kslicer::GetHashOfSourceRange(expr->getSourceRange());
  m_rewrittenNodes.insert(hash);
  return true;
}

void kslicer::MarkRewrittenRecursive(const clang::Stmt* currNode, std::unordered_set<uint64_t>& a_rewrittenNodes)
{
  kslicer::NodesMarker rv(a_rewrittenNodes); 
  rv.TraverseStmt(const_cast<clang::Stmt*>(currNode));
}

void kslicer::MarkRewrittenRecursive(const clang::Decl* currNode, std::unordered_set<uint64_t>& a_rewrittenNodes)
{
  kslicer::NodesMarker rv(a_rewrittenNodes); 
  rv.TraverseDecl(const_cast<clang::Decl*>(currNode));
}
