#include "kslicer.h"

#include <sstream>
#include <algorithm>

void kslicer::FunctionRewriter2::InitKernelData(kslicer::KernelInfo& a_kernelRef, const std::string& a_fakeOffsetExp)
{
  m_kernelMode  = true;
  m_pCurrKernel = &a_kernelRef;
  for(auto arg : a_kernelRef.args)
  {
    if(arg.isLoopSize || arg.IsUser())
      m_kernelUserArgs.insert(arg.name);
  }

  m_shit = a_kernelRef.currentShit;

  // fill other auxilary structures
  //
  m_fakeOffsetExp = a_fakeOffsetExp;
  m_variables.reserve(m_codeInfo->dataMembers.size());
  for(const auto& var : m_codeInfo->dataMembers)
    m_variables[var.name] = var;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


bool kslicer::FunctionRewriter2::NeedToRewriteMemberExpr(const clang::MemberExpr* expr, std::string& out_text)
{
  if(!m_kernelMode)
    return false;

  clang::ValueDecl* pValueDecl = expr->getMemberDecl();
  if(!clang::isa<clang::FieldDecl>(pValueDecl))
    return false;

  clang::FieldDecl*  pFieldDecl   = clang::dyn_cast<clang::FieldDecl>(pValueDecl);
  std::string        fieldName    = pFieldDecl->getNameAsString();
  clang::RecordDecl* pRecodDecl   = pFieldDecl->getParent();
  const std::string  thisTypeName = kslicer::CleanTypeName(pRecodDecl->getNameAsString());
  
  if(!WasNotRewrittenYet(expr))
    return false;
  
  //std::string debugText = kslicer::GetRangeSourceCode(expr->getSourceRange(), m_compiler);

  // (1) setter access
  //
  std::string setter, containerName;
  if(CheckSettersAccess(expr, m_codeInfo, m_compiler, &setter, &containerName)) // process setter access
  {
    out_text = setter + "_" + containerName;
    return true; 
  }
  
  bool usedWithVBR = false;
  auto pFoundInAllData = m_codeInfo->allDataMembers.find(fieldName);
  if(pFoundInAllData != m_codeInfo->allDataMembers.end())
    usedWithVBR = pFoundInAllData->second.bindWithRef;

  bool inCompositiClass = false;
  auto pPrefix = m_codeInfo->composPrefix.find(thisTypeName);
  std::string classPrefix = "";
  if(m_pCurrFuncInfo != nullptr)
    classPrefix = m_pCurrFuncInfo->prefixName;
  
  if(m_pCurrFuncInfo != nullptr && pPrefix != m_codeInfo->composPrefix.end())
  {
    fieldName = pPrefix->second + "_" + fieldName;
    inCompositiClass = true;
  }
  else if(thisTypeName != m_codeInfo->mainClassName) // (2) *payload ==>  payload[fakeOffset],  RTV, process access to arguments payload->xxx
  {
    clang::Expr* baseExpr = expr->getBase(); 
    assert(baseExpr != nullptr);

    const std::string baseName = GetRangeSourceCode(baseExpr->getSourceRange(), m_compiler);

    size_t foundId  = size_t(-1);
    bool needOffset = false;
    for(size_t i=0;i<m_pCurrKernel->args.size();i++)
    {
      if(m_pCurrKernel->args[i].name == baseName)
      {
        foundId    = i;
        needOffset = m_pCurrKernel->args[i].needFakeOffset;
        break;
      }
    }
    
    bool isKernel = m_codeInfo->IsKernel(m_pCurrKernel->name) && !processFuncMember;

    if(foundId != size_t(-1)) // else we didn't found 'payload' in kernel arguments, so just ignore it
    {
      // now split 'payload->xxx' to 'payload' (baseName) and 'xxx' (memberName); 
      // 
      const std::string exprContent = GetRangeSourceCode(expr->getSourceRange(), m_compiler);
      auto pos = exprContent.find("->");

      if(pos != std::string::npos && !processFuncMember)
      {    
        const std::string memberName = exprContent.substr(pos+2);
        if(m_codeInfo->megakernelRTV && SLANG_ELIMINATE_LOCAL_POINTERS) 
        {
          out_text = baseName + "." + memberName;
          return true;
        }
        else if(needOffset)
        {
          out_text = baseName + "[" + m_fakeOffsetExp + "]." + memberName;
          return true;
        }
      }
    }
    else if(!isKernel) // for common member functions
    {
      const std::string exprContent = GetRangeSourceCode(expr->getSourceRange(), m_compiler);
      auto pos = exprContent.find("->");
      if(pos != std::string::npos) 
      {
        const std::string memberName = exprContent.substr(pos+2);
        out_text = baseName + "." + memberName;
        return true;
      }
    }
  }
  else if(m_codeInfo->dataClassNames.find(thisTypeName) != m_codeInfo->dataClassNames.end() && usedWithVBR) 
  {
    out_text = "all_references." + fieldName + "." + fieldName;
    return true;
  }

  // (3) member ==> ubo.member
  // 
  const auto pMember = m_variables.find(fieldName);
  if(pMember == m_variables.end())
    return false;

  if(inCompositiClass && WasNotRewrittenYet(expr))
  {
    if(pMember->second.isContainer)
      out_text = pMember->second.name;
    else
      out_text = m_codeInfo->pShaderCC->UBOAccess(pMember->second.name);
    return true;
  }
  
  // (2) put ubo->var instead of var, leave containers as they are
  // process arrays and large data structures because small can be read once in the beggining of kernel
  // // m_currKernel.hasInitPass &&
  const bool isInLoopInitPart   = !m_codeInfo->IsRTV() && (expr->getSourceRange().getEnd()   < m_pCurrKernel->loopInsides.getBegin());
  const bool isInLoopFinishPart = !m_codeInfo->IsRTV() && (expr->getSourceRange().getBegin() > m_pCurrKernel->loopInsides.getEnd());
  const bool hasLargeSize     = true; // (pMember->second.sizeInBytes > kslicer::READ_BEFORE_USE_THRESHOLD);
  const bool inMegaKernel     = m_codeInfo->megakernelRTV;
  const bool subjectedToRed   = m_pCurrKernel->subjectedToReduction.find(fieldName) != m_pCurrKernel->subjectedToReduction.end();
  
  if(m_codeInfo->pShaderCC->IsISPC() && subjectedToRed)
    return false;
  
  if(!pMember->second.isContainer && WasNotRewrittenYet(expr) && (isInLoopInitPart || isInLoopFinishPart || !subjectedToRed) && 
                                                                 (isInLoopInitPart || isInLoopFinishPart || pMember->second.isArray || hasLargeSize || inMegaKernel)) 
  {
    out_text = m_codeInfo->pShaderCC->UBOAccess(pMember->second.name);
    clang::SourceRange thisRng = expr->getSourceRange();
    clang::SourceRange endkRng = m_pCurrKernel->loopOutsidesFinish;
    if(thisRng.getEnd() == endkRng.getEnd()) // fixing stnrange bug
      kslicer::PrintError("possible end-of-loop bug", thisRng, m_compiler.getSourceManager());
    return true;
  }

  return false;
}

bool kslicer::FunctionRewriter2::CheckIfExprHasArgumentThatNeedFakeOffset(const std::string& exprStr)
{
  if(m_pCurrKernel == nullptr)
    return false;

  bool needOffset = false;
  for(const auto arg: m_pCurrKernel->args)
  {
    if(exprStr.find(arg.name) != std::string::npos)
    {
      if(arg.needFakeOffset)
      {
        needOffset = true;
        break;
      }
    }
  }

  return needOffset;
}

bool kslicer::FunctionRewriter2::NameNeedsFakeOffset(const std::string& a_name) const
{
  if(m_pCurrKernel == nullptr)
    return false;

   bool exclude = false;
   for(auto arg : m_pCurrKernel->args)
   {
     if(arg.needFakeOffset && arg.name == a_name)
       exclude = true;
   }
   return exclude;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

kslicer::RewrittenFunction kslicer::FunctionRewriter2::RewriteFunction(clang::FunctionDecl* fDecl)
{
  return FunctionRewriter::RewriteFunction(fDecl);
}

std::string kslicer::FunctionRewriter2::RewriteFuncDecl(clang::FunctionDecl* fDecl)
{
  return FunctionRewriter::RewriteFuncDecl(fDecl);
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
    m_codeInfo->m_functionsDone[hash] = RewriteFunction(fDecl);
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
  
  std::string shallow;
  if(DetectAndRewriteShallowPattern(expr, shallow)) 
    return shallow;

  FunctionRewriter2 rvCopy = *this;
  rvCopy.TraverseStmt(const_cast<clang::Stmt*>(expr));

  auto range = expr->getSourceRange();
  auto p = rvCopy.m_workAround.find(GetHashOfSourceRange(range));
  if(p != rvCopy.m_workAround.end())
    return p->second;
  else
    return m_rewriter.getRewrittenText(range);
}

bool kslicer::FunctionRewriter2::DetectAndRewriteShallowPattern(const clang::Stmt* expr, std::string& a_out)
{
  if(clang::isa<clang::MemberExpr>(expr))
  {
    const clang::MemberExpr* memberExpr = clang::dyn_cast<clang::MemberExpr>(expr);
    if(m_kernelMode && NeedToRewriteMemberExpr(memberExpr, a_out))
      return true;
  }

  return false;
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

bool kslicer::KernelRewriter2::VisitDeclRefExpr_Impl(clang::DeclRefExpr* expr)                    
{ 
  return m_pFunRW2->VisitDeclRefExpr_Impl(expr); 
}

bool kslicer::KernelRewriter2::VisitDeclStmt_Impl(clang::DeclStmt* decl)                          { return m_pFunRW2->VisitDeclStmt_Impl(decl); }
bool kslicer::KernelRewriter2::VisitArraySubscriptExpr_Impl(clang::ArraySubscriptExpr* arrayExpr) { return m_pFunRW2->VisitArraySubscriptExpr_Impl(arrayExpr); }
bool kslicer::KernelRewriter2::VisitUnaryExprOrTypeTraitExpr_Impl(clang::UnaryExprOrTypeTraitExpr* szOfExpr) { return m_pFunRW2->VisitUnaryExprOrTypeTraitExpr_Impl(szOfExpr); }
bool kslicer::KernelRewriter2::VisitMemberExpr_Impl(clang::MemberExpr* expr) 
{ 
  return m_pFunRW2->VisitMemberExpr_Impl(expr); 
}

bool kslicer::KernelRewriter2::VisitCXXConstructExpr_Impl(clang::CXXConstructExpr* call)
{
  return m_pFunRW2->VisitCXXConstructExpr_Impl(call); 
}

bool kslicer::KernelRewriter2::VisitCallExpr_Impl(clang::CallExpr* call)
{
  return m_pFunRW2->VisitCallExpr_Impl(call); 
}
