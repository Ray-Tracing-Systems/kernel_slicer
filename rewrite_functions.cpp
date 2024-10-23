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
    textRes += RecursiveRewrite(kslicer::RemoveImplicitCast(call->getArg(i)));
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
  //std::cout << "[VisitCallExpr]: fname = " << fname.c_str() << std::endl;

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
      ReplaceTextOrWorkAround(call->getSourceRange(), textRes);
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
  
  const std::string debugText = GetRangeSourceCode(call->getSourceRange(), m_compiler);   
  const std::string fname = ctorDecl->getNameInfo().getName().getAsString();

  if(kslicer::IsVectorContructorNeedsReplacement(fname) && WasNotRewrittenYet(call) && !ctorDecl->isCopyOrMoveConstructor() && call->getNumArgs() > 0 ) //
  {
    const std::string text = FunctionCallRewriteNoName(call);
    std::string textRes    = VectorTypeContructorReplace(fname, text); 
    if(fname == "complex" && call->getNumArgs() == 1)
      textRes = "to_complex" + text;
    else if(fname == "complex" && call->getNumArgs() == 2)
      textRes = "make_complex" + text;
    ReplaceTextOrWorkAround(call->getSourceRange(), textRes);
    MarkRewritten(call);
  }

  return true;
}

bool kslicer::FunctionRewriter::VisitCXXOperatorCallExpr_Impl(clang::CXXOperatorCallExpr* expr)
{
  std::string op        = kslicer::GetRangeSourceCode(clang::SourceRange(expr->getOperatorLoc()), m_compiler); 
  std::string debugText = kslicer::GetRangeSourceCode(expr->getSourceRange(), m_compiler);   

  static std::unordered_map<std::string, std::string> remapOp = {{"+","add"}, {"-","sub"}, {"*","mul"}, {"/","div"}};

  if((op == "+" || op == "-" || op == "*" || op == "/") && WasNotRewrittenYet(expr))
  {
    clang::Expr* left  = kslicer::RemoveImplicitCast(expr->getArg(0));
    clang::Expr* right = kslicer::RemoveImplicitCast(expr->getArg(1));

    const std::string leftType  = left->getType().getAsString();
    const std::string rightType = right->getType().getAsString();
    const std::string keyType   = "complex"; 

    if(debugText == "eta * cosThetaI")
    {
      int a = 2;
      //expr->dump();
    }

    if(leftType == keyType || rightType == keyType)
    {
      const std::string leftText  = RecursiveRewrite(left);
      const std::string rightText = RecursiveRewrite(right);
      
      std::string rewrittenOp;
      {
        if(leftType == keyType && rightType == keyType)
          rewrittenOp = keyType + "_" + remapOp[op] + "(" + leftText + "," + rightText + ")";
        else if (leftType == keyType)
          rewrittenOp = keyType + "_" + remapOp[op] + "_real(" + leftText + "," + rightText + ")";
        else if(rightType == keyType)
          rewrittenOp =  "real_" + remapOp[op] + "_" + keyType + "(" + leftText + "," + rightText + ")";
      }
      ReplaceTextOrWorkAround(expr->getSourceRange(), rewrittenOp);
      MarkRewritten(expr);
    }
  }
  return true;
}

std::string kslicer::FunctionRewriter::RecursiveRewrite(const clang::Stmt* expr)
{
  if(expr == nullptr)
    return "";
  FunctionRewriter rvCopy = *this;
  rvCopy.TraverseStmt(const_cast<clang::Stmt*>(expr));

  auto range = expr->getSourceRange();
  auto p = rvCopy.m_workAround.find(GetHashOfSourceRange(range));
  if(p != rvCopy.m_workAround.end())
    return p->second;
  else
    return m_rewriter.getRewrittenText(range);
}


bool kslicer::FunctionRewriter::BadASTPattern(const clang::Stmt* expr)
{
  // bad pattern #1:
  // CXXConstructExpr 
  // `-ImplicitCastExpr 
  //  `-DeclRefExpr 
  if(clang::isa<clang::CXXConstructExpr>(expr)) 
  {
    const auto construct = clang::dyn_cast<clang::CXXConstructExpr>(expr);
    if(construct->getNumArgs() == 1)
    {
      auto argNode = construct->getArg(0);
      if(clang::isa<clang::ImplicitCastExpr>(argNode))
      {
        const auto argNodeUnderlying = kslicer::RemoveImplicitCast(argNode);
        if(clang::isa<clang::DeclRefExpr>(argNodeUnderlying))
          return true;
      }
    }
  }

  return false;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool kslicer::NodesMarker::VisitStmt(clang::Stmt* expr)
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
