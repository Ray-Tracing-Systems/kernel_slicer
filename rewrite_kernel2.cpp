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

std::string kslicer::FunctionRewriter2::GlobalPtrOffsetText(const kslicer::GlobalPtrMatch& a_match)
{
  std::string res;
  if(a_match.localVar != nullptr)
    res = kslicer::LocalGlobalPointerOffsetName(a_match.localVar);
  for(const clang::Expr* offsetExpr : a_match.offsetExprs)
  {
    const std::string offset = "(" + RecursiveRewrite(offsetExpr) + ")";
    res = (res == "") ? offset : res + " + " + offset;
  }
  return (res == "") ? std::string("0") : res;
}

std::string kslicer::FunctionRewriter2::GlobalBufferAccessText(const std::string& a_buffer, const std::string& a_index)
{
  return a_buffer + "[" + a_index + "]";
}

std::string kslicer::FunctionRewriter2::ShitPointerBuffer(const kslicer::ArgMatch& a_pointer)
{
  return a_pointer.actual;
}

const kslicer::ArgMatch* kslicer::FunctionRewriter2::FindShitPointer(const clang::Expr* a_expr, std::vector<const clang::Expr*>* a_pAdds) const
{
  if(m_shit.pointers.empty() || a_expr == nullptr)
    return nullptr;
  std::vector<const clang::Expr*> adds;
  const clang::Expr* expr = kslicer::StripPointerAdditions(a_expr, &adds);
  if(!clang::isa<clang::DeclRefExpr>(expr) || !clang::isa<clang::ParmVarDecl>(clang::dyn_cast<clang::DeclRefExpr>(expr)->getDecl()))
    return nullptr;
  const std::string name = clang::dyn_cast<clang::DeclRefExpr>(expr)->getDecl()->getNameAsString();
  for(const auto& p : m_shit.pointers)
  {
    if(p.formal == name)
    {
      if(a_pAdds != nullptr)
        (*a_pAdds) = adds;
      return &p;
    }
  }
  return nullptr;
}

std::string kslicer::FunctionRewriter2::ShitPointerOffsetText(const kslicer::ArgMatch& a_pointer, const std::vector<const clang::Expr*>& a_adds)
{
  std::string res = a_pointer.formal + "Offset";
  for(const clang::Expr* add : a_adds)
    res += " + (" + RecursiveRewrite(add) + ")";
  return res;
}

bool kslicer::FunctionRewriter2::RewriteLocalGlobalPtrDecl(clang::VarDecl* a_decl)
{
  if(!m_kernelMode || m_pCurrKernel == nullptr || a_decl == nullptr)
    return false;
  auto match = kslicer::MatchLocalGlobalPointerVar(a_decl, *m_pCurrKernel, m_compiler);
  if(!match.valid)
    return false;
  if(a_decl->getInit() == nullptr || !WasNotRewrittenYet(a_decl->getInit()))
    return true;
  const std::string text = "uint " + kslicer::LocalGlobalPointerOffsetName(a_decl) + " = uint(" + GlobalPtrOffsetText(match) + ")";
  ReplaceText(a_decl->getSourceRange(), text);
  MarkRewritten(a_decl->getInit());
  return true;
}

bool kslicer::FunctionRewriter2::RewriteGlobalPtrAccess(clang::Expr* a_expr)
{
  if(a_expr == nullptr || !WasNotRewrittenYet(a_expr))
    return false;

  const clang::Expr* base   = nullptr;
  const clang::Expr* idx    = nullptr;
  std::string        member = "";
  if(clang::isa<clang::ArraySubscriptExpr>(a_expr))                                                                  // 'p[i]'
  {
    base = clang::dyn_cast<clang::ArraySubscriptExpr>(a_expr)->getBase();
    idx  = clang::dyn_cast<clang::ArraySubscriptExpr>(a_expr)->getIdx();
  }
  else if(clang::isa<clang::UnaryOperator>(a_expr) && clang::dyn_cast<clang::UnaryOperator>(a_expr)->getOpcode() == clang::UO_Deref) // '*p'
    base = clang::dyn_cast<clang::UnaryOperator>(a_expr)->getSubExpr();
  else if(clang::isa<clang::MemberExpr>(a_expr) && clang::dyn_cast<clang::MemberExpr>(a_expr)->isArrow())                // 'p->x'
  {
    const auto memberExpr = clang::dyn_cast<clang::MemberExpr>(a_expr);
    base   = memberExpr->getBase();
    member = memberExpr->getMemberDecl()->getNameAsString();
  }
  if(base == nullptr)
    return false;
  
  std::string buffer, offset;
  if(m_kernelMode && m_pCurrKernel != nullptr)                // local pointer variable in kernel or member function
  {
    auto match = kslicer::MatchGlobalPointer(base, *m_pCurrKernel, m_compiler);
    if(!match.valid || match.localVar == nullptr)
      return false;
    buffer = match.container;
    offset = GlobalPtrOffsetText(match);
  }
  else                                                        // pointer argument of function copy for global memory
  {
    std::vector<const clang::Expr*> adds;
    const kslicer::ArgMatch* pPointer = FindShitPointer(base, &adds);
    if(pPointer == nullptr)
      return false;
    buffer = ShitPointerBuffer(*pPointer);
    offset = ShitPointerOffsetText(*pPointer, adds);
  }

  std::string index = offset;
  if(idx != nullptr)
    index = offset + " + (" + RecursiveRewrite(idx) + ")";
  
  std::string text = GlobalBufferAccessText(buffer, index);
  if(member != "")
    text += "." + member;
  ReplaceText(a_expr->getSourceRange(), text);
  MarkRewritten(a_expr);
  return true;
}

bool kslicer::FunctionRewriter2::IsUserFunction(const clang::FunctionDecl* a_func)
{
  if(a_func == nullptr)
    return false;
  const std::string fileName = std::string(m_compiler.getSourceManager().getFilename(a_func->getSourceRange().getBegin()));
  return !m_codeInfo->IsInExcludedFolder(fileName);
}

bool kslicer::FunctionRewriter2::CheckLocalGlobalPtrUse(clang::DeclRefExpr* a_expr)
{
  if(!m_kernelMode || m_pCurrKernel == nullptr || !WasNotRewrittenYet(a_expr))
    return false;
  const auto pVar = clang::dyn_cast<clang::VarDecl>(a_expr->getDecl());
  if(pVar == nullptr || !kslicer::MatchLocalGlobalPointerVar(pVar, *m_pCurrKernel, m_compiler).valid)
    return false;
  kslicer::PrintError("local pointer to buffer '" + pVar->getNameAsString() + "' is used in unsupported way; supported are 'p[i]', '*p', 'p->x', 'p + k' and passing 'p' or 'p + k' to functions", 
                      a_expr->getSourceRange(), m_compiler.getSourceManager());
  MarkRewritten(a_expr);
  return true;
}

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
  
  std::string debugText = kslicer::GetRangeSourceCode(expr->getSourceRange(), m_compiler);


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
  
  // check 'surf_data.aperture' case
  clang::Expr* baseExpr = expr->getBase();
  if(baseExpr != nullptr)
  {
    const std::string originalText = kslicer::GetRangeSourceCode(expr->getSourceRange(), m_compiler); 
    const std::string baseName     = GetRangeSourceCode(baseExpr->getSourceRange(), m_compiler);
    if(originalText == baseName + "." + fieldName)
      return false;
  }

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

  const bool isInLoopInitPart   = (m_pCurrKernel->pattern != kslicer::PATTERN_TP::PATTERN_RTV) && (expr->getSourceRange().getEnd()   < m_pCurrKernel->loopInsides.getBegin());
  const bool isInLoopFinishPart = (m_pCurrKernel->pattern != kslicer::PATTERN_TP::PATTERN_RTV) && (expr->getSourceRange().getBegin() > m_pCurrKernel->loopInsides.getEnd());
  const bool hasLargeSize   = true; // (pMember->second.sizeInBytes > kslicer::READ_BEFORE_USE_THRESHOLD);
  const bool inMegaKernel   = m_codeInfo->megakernelRTV;
  const bool subjectedToRed = m_pCurrKernel->subjectedToReduction.find(fieldName) != m_pCurrKernel->subjectedToReduction.end();
  const bool checkContainer = !pMember->second.isContainer || m_codeInfo->placeVectorsInUBO; // || CUDA_ENABLE_UBO 

  if(m_codeInfo->pShaderCC->IsISPC() && subjectedToRed && !isInLoopInitPart && !isInLoopFinishPart) // inside loop ISPC uses local varying variable
    return false;
  
  if(checkContainer && WasNotRewrittenYet(expr) && (isInLoopInitPart || isInLoopFinishPart || !subjectedToRed) && 
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

bool kslicer::FunctionRewriter2::NeedToRewriteDeclRefExpr(const clang::DeclRefExpr* expr, std::string& out_text)
{
  if(!m_kernelMode)
    return false;

  const clang::ValueDecl* pDecl = expr->getDecl();
  if(m_pCurrKernel != nullptr)                     // local variable which is declared before the loop and placed to UBO
  {
    for(const auto& var : m_pCurrKernel->localVarsInUBO)
    {
      if(var.declLoc == pDecl->getLocation())
      {
        out_text = m_codeInfo->pShaderCC->UBOAccess(var.uboName);
        return true;
      }
    }
  }

  if(!clang::isa<clang::ParmVarDecl>(pDecl))
    return false;

  clang::QualType qt = pDecl->getType();
  if(qt->isPointerType()) // we can't put pointers to push constants, but can copy references as full structure data
    return false;

  const std::string textOri = kslicer::GetRangeSourceCode(expr->getSourceRange(), m_compiler); //

  bool insideKernel = true;
  if(m_pCurrFuncInfo != nullptr)
    insideKernel = m_pCurrFuncInfo->isKernel;

  if(insideKernel && m_kernelUserArgs.find(textOri) != m_kernelUserArgs.end() && ((m_pCurrKernel->pattern != PATTERN_TP::PATTERN_RTV) || !m_codeInfo->megakernelRTV || m_pCurrKernel->isMega))
  {
    out_text = this->KGenArgsName() + textOri;
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

std::string kslicer::FunctionRewriter2::CompleteFunctionCallRewrite(clang::CallExpr* call)
{
  std::string rewrittenRes = "";
  for(unsigned i=0;i<call->getNumArgs(); i++)
  {
    rewrittenRes += RecursiveRewrite(call->getArg(i));
    if(i!=call->getNumArgs()-1)
      rewrittenRes += ", ";
  }
  rewrittenRes += ")";
  return rewrittenRes;
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
bool kslicer::FunctionRewriter2::VisitCXXMemberCallExpr_Impl(clang::CXXMemberCallExpr* f)  
{ 
  return true; 
} 
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
    DARExpr_TextureAccess(expr);
    auto opRange = expr->getSourceRange();
    if(opRange.getEnd()   <= m_pCurrKernel->loopInsides.getBegin() || 
       opRange.getBegin() >= m_pCurrKernel->loopInsides.getEnd() ) // not inside loop
      return true;   
  
    const auto numArgs = expr->getNumArgs();
    if(numArgs != 2)
      return true; 
    
    std::string op = GetRangeSourceCode(clang::SourceRange(expr->getOperatorLoc()), m_compiler);  
    if(op == "+=" || op == "-=" || op == "*=")
    {
      const clang::Expr* lhs = expr->getArg(0);
      const clang::Expr* rhs = expr->getArg(1);
      
      std::string resText;
      if(WasNotRewrittenYet(expr) && NeedToRewriteReductionOp(op, lhs, rhs, expr, resText))
      {
        ReplaceText(expr->getSourceRange(), resText); 
        MarkRewritten(expr);
      }
    }
    else if (op == "=")
    {
      const clang::Expr* lhs = expr->getArg(0);
      const clang::Expr* rhs = expr->getArg(1);
  
      DARExpr_ReductionFunc(lhs, rhs, expr);
    }

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
    auto opRange = expr->getSourceRange();
    if(opRange.getEnd()   <= m_pCurrKernel->loopInsides.getBegin() || 
       opRange.getBegin() >= m_pCurrKernel->loopInsides.getEnd() ) // not inside loop
      return true;   
  
    const clang::Expr* lhs = expr->getLHS();
    const clang::Expr* rhs = expr->getRHS();
    const std::string  op  = std::string(expr->getOpcodeStr()); // std::string op = GetRangeSourceCode(clang::SourceRange(expr->getOperatorLoc()), m_compiler);
    
    std::string resText;
    if(WasNotRewrittenYet(expr) && NeedToRewriteReductionOp(op, lhs, rhs, expr, resText))
    {
      ReplaceText(expr->getSourceRange(), resText); 
      MarkRewritten(expr);
    }
  }

  return true; 
}

bool kslicer::FunctionRewriter2::VisitBinaryOperator_Impl(clang::BinaryOperator* expr)
{
  if(m_kernelMode)
  {
    auto opRange = expr->getSourceRange();

    if((m_pCurrKernel->pattern != kslicer::PATTERN_TP::PATTERN_RTV) && (opRange.getEnd()<= m_pCurrKernel->loopInsides.getBegin() || opRange.getBegin() >= m_pCurrKernel->loopInsides.getEnd())) // not inside loop
      return true;  
  
    const auto op = expr->getOpcodeStr();
    if(op != "=")
      return true;
    
    DARExpr_TextureAccess(expr);
  
    const clang::Expr* lhs = expr->getLHS();
    const clang::Expr* rhs = expr->getRHS();
    
    DARExpr_ReductionFunc(lhs, rhs, expr);
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

bool kslicer::FunctionRewriter2::VisitFloatingLiteral_Impl(clang::FloatingLiteral* expr) 
{
  return true;
}

bool kslicer::FunctionRewriter2::VisitReturnStmt_Impl(clang::ReturnStmt* ret)
{
  return true;
}


std::string kslicer::FunctionRewriter2::RecursiveRewrite(const clang::Stmt* expr)
{
  if(expr == nullptr)
    return "";
  if(clang::isa<clang::CXXDefaultArgExpr>(expr)) // default argument of function call: its value
    return RecursiveRewrite(clang::dyn_cast<clang::CXXDefaultArgExpr>(expr)->getExpr());
  
  std::string shallow;
  if(DetectAndRewriteShallowPattern(expr, shallow)) 
    return shallow;

  FunctionRewriter2 rvCopy = *this;
  rvCopy.TraverseStmt(const_cast<clang::Stmt*>(expr));

  return kslicer::GetRewrittenText(m_rewriter, expr->getSourceRange());
}

bool kslicer::FunctionRewriter2::DetectAndRewriteShallowPattern(const clang::Stmt* expr, std::string& a_out)
{
  if(!m_kernelMode)
    return false;
  
  if(clang::isa<clang::MemberExpr>(expr))
  {
    const clang::MemberExpr* memberExpr = clang::dyn_cast<clang::MemberExpr>(expr);
    if(NeedToRewriteMemberExpr(memberExpr, a_out))
      return true;
  }
  else if(clang::isa<clang::DeclRefExpr>(expr))
  {
    const clang::DeclRefExpr* drExpr = clang::dyn_cast<clang::DeclRefExpr>(expr);
    if(NeedToRewriteDeclRefExpr(drExpr, a_out))
      return true;
  }
  else if(clang::isa<clang::BinaryOperator>(expr))
  {
    const clang::BinaryOperator* binOp = clang::dyn_cast<clang::BinaryOperator>(expr);
    const clang::Expr* lhs = kslicer::RemoveImplicitCast(binOp->getLHS());  // Указатель на 'a'
    const clang::Expr* rhs = kslicer::RemoveImplicitCast(binOp->getRHS());  // Указатель на 'b'
    std::string opCode = std::string(binOp->getOpcodeStr());
    std::string leftRes  = kslicer::GetRangeSourceCode(lhs->getSourceRange(), m_compiler);
    std::string rightRes = kslicer::GetRangeSourceCode(rhs->getSourceRange(), m_compiler);
    if(DetectAndRewriteShallowPattern(lhs, leftRes) || DetectAndRewriteShallowPattern(rhs, rightRes))
    {
      a_out = leftRes + opCode + rightRes;
      return true;
    }
  }
  else if(clang::isa<clang::CXXOperatorCallExpr>(expr)) 
  {
    const clang::CXXOperatorCallExpr* opCall = clang::dyn_cast<clang::CXXOperatorCallExpr>(expr);
    if(opCall->getNumArgs() != 2)                 // unary operators, 'it->second' for example
      return false;
    if(kslicer::GetHashMapSubscript(opCall) != nullptr) // 'm_items[key]' needs full rewrite
      return false;
    const clang::Expr* lhs = kslicer::RemoveImplicitCast(opCall->getArg(0));
    const clang::Expr* rhs = kslicer::RemoveImplicitCast(opCall->getArg(1));
    std::string opCode = clang::getOperatorSpelling(opCall->getOperator());
    std::string leftRes  = kslicer::GetRangeSourceCode(lhs->getSourceRange(), m_compiler);
    std::string rightRes = kslicer::GetRangeSourceCode(rhs->getSourceRange(), m_compiler);
    if(DetectAndRewriteShallowPattern(lhs, leftRes) || DetectAndRewriteShallowPattern(rhs, rightRes))
    {
      if(opCode == "[]")
        a_out = leftRes + "[" + rightRes + "]";
      else
        a_out = leftRes + opCode + rightRes;
      return true;
    }
  }

  return false;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool kslicer::FunctionRewriter2::NeedToRewriteReductionOp(const std::string& op, const clang::Expr* lhs, const clang::Expr* rhs, const clang::Expr* expr, std::string& outStr)
{
  if(!m_kernelMode || m_pCurrKernel == nullptr)
    return false;

  // detect cases like "m_bodies[i].vel_charge.x += acceleration.x", extract "m_bodies[i]"; "vel_charge.x" is not supported right now !!!
  //
  while(clang::isa<clang::MemberExpr>(lhs))
  {
    const clang::MemberExpr* lhsMember = clang::dyn_cast<const clang::MemberExpr>(lhs);
    lhs = lhsMember->getBase(); 
  }

  if(clang::isa<clang::CXXOperatorCallExpr>(lhs)) // do not process here code like "vector[i] += ..." or "texture[int2(x,y)] += ..."
    return false;

  const std::string leftVar = GetRangeSourceCode(lhs->getSourceRange().getBegin(), m_compiler);
  const std::string leftStr = GetRangeSourceCode(lhs->getSourceRange(), m_compiler);
  
  auto pShaderRewriter = m_codeInfo->pShaderFuncRewriter;  
  auto p = m_pCurrKernel->usedMembers.find(leftVar);
  if(p != m_pCurrKernel->usedMembers.end())
  {
    KernelInfo::ReductionAccess access;
    access.type      = KernelInfo::REDUCTION_TYPE::UNKNOWN;
    access.rightExpr = GetRangeSourceCode(rhs->getSourceRange(), m_compiler);
    access.leftExpr  = leftStr;
    access.dataType  = rhs->getType().getAsString();
    ReplaceFirst(access.dataType, "const ", "");
    access.dataType = pShaderRewriter->RewriteStdVectorTypeStr(access.dataType); 
   
    if(leftVar != leftStr && clang::isa<clang::ArraySubscriptExpr>(lhs))
    {
      auto lhsArray    = clang::dyn_cast<const clang::ArraySubscriptExpr>(lhs);
      const clang::Expr* idx  = lhsArray->getIdx();  // array index
      const clang::Expr* name = lhsArray->getBase(); // array name

      access.leftIsArray = true;
      access.arraySize   = 0;
      access.arrayIndex  = GetRangeSourceCode(idx->getSourceRange(), m_compiler);
      access.arrayName   = GetRangeSourceCode(name->getSourceRange(), m_compiler); 
      
      // extract array size
      //
      const clang::Expr* nextNode = kslicer::RemoveImplicitCast(lhsArray->getLHS());  
      if(clang::isa<clang::MemberExpr>(nextNode))
      {
        const clang::MemberExpr* pMemberExpr = clang::dyn_cast<const clang::MemberExpr>(nextNode); 
        const clang::ValueDecl*  valDecl     = pMemberExpr->getMemberDecl();
        const clang::QualType    qt          = valDecl->getType();
        const auto               typePtr     = qt.getTypePtr(); 
        if(typePtr->isConstantArrayType()) 
        {    
          auto arrayType   = clang::dyn_cast<const clang::ConstantArrayType>(typePtr);     
          access.arraySize = arrayType->getSize().getLimitedValue(); 
        } 
      }
   
      if(access.arraySize == 0)
        kslicer::PrintError("[KernelRewriter::ProcessReductionOp]: can't determine array size ", lhs->getSourceRange(), m_compiler.getSourceManager());
    }

    if(op == "+=")
      access.type    = KernelInfo::REDUCTION_TYPE::ADD;
    else if(op == "*=")
      access.type    = KernelInfo::REDUCTION_TYPE::MUL;
    else if(op == "-=")
      access.type    = KernelInfo::REDUCTION_TYPE::SUB;
    
    //auto exprHash = kslicer::GetHashOfSourceRange(expr->getSourceRange());

    if(WasNotRewrittenYet(expr) && !IsISPC())
    {
      std::string rightStr2  = RecursiveRewrite(rhs);
      std::string localIdStr = m_codeInfo->pShaderCC->LocalIdExpr(m_pCurrKernel->GetDim(), m_pCurrKernel->wgSize);
      if(access.leftIsArray)
        outStr = access.arrayName + "Shared[" + access.arrayIndex + "][" + localIdStr + "] " + access.GetOp(m_codeInfo->pShaderCC) + " " + rightStr2;
      else
        outStr = leftVar + "Shared[" + localIdStr + "] " + access.GetOp(m_codeInfo->pShaderCC) + " " + rightStr2;
      return true;
    }
  }

  return false;
}

void kslicer::FunctionRewriter2::DARExpr_ReductionFunc(const clang::Expr* lhs, const clang::Expr* rhs, const clang::Expr* expr)
{
  if(!clang::isa<clang::MemberExpr>(lhs) || !m_kernelMode || m_pCurrKernel == nullptr)
    return;

  std::string leftStr  = GetRangeSourceCode(lhs->getSourceRange(), m_compiler);
  std::string rightStr = GetRangeSourceCode(rhs->getSourceRange(), m_compiler);
  auto p = m_pCurrKernel->usedMembers.find(leftStr);
  if(p == m_pCurrKernel->usedMembers.end())
    return;

  if(clang::isa<clang::MaterializeTemporaryExpr>(rhs))
  {
    auto tmp = clang::dyn_cast<clang::MaterializeTemporaryExpr>(rhs);
    rhs = tmp->getSubExpr();
  }

  if(!clang::isa<clang::CallExpr>(rhs))
  {
    PrintWarning("unsupported expression for reduction via assigment inside loop; must be 'a = f(a,b)'", rhs->getSourceRange(), m_compiler.getSourceManager());
    PrintWarning("reduction for variable '" +  leftStr + "' will not be generated, sure this is ok for you?", rhs->getSourceRange(), m_compiler.getSourceManager());
    return;
  }  
  
  auto call    = clang::dyn_cast<clang::CallExpr>(rhs);
  auto numArgs = call->getNumArgs();
  if(numArgs != 2)
  {
    PrintError("function which is used in reduction must have 2 args; a = f(a,b)'", expr->getSourceRange(), m_compiler.getSourceManager());
    return;
  }

  const clang::Expr* arg0 = call->getArg(0);
  const clang::Expr* arg1 = call->getArg(1);

  std::string arg0Str = GetRangeSourceCode(arg0->getSourceRange(), m_compiler);
  std::string arg1Str = GetRangeSourceCode(arg1->getSourceRange(), m_compiler);
  
  std::string secondArg;
  if(arg0Str == leftStr)
  {
    secondArg = arg1Str;
  }
  else if(arg1Str == leftStr)
  {
    secondArg = arg0Str;
  }
  else
  {
    PrintError("incorrect arguments of reduction function, one of them must be same as assigment result; a = f(a,b)'", expr->getSourceRange(), m_compiler.getSourceManager());
    return;
  }

  std::string callExpr = GetRangeSourceCode(call->getSourceRange(), m_compiler);
  std::string fname    = callExpr.substr(0, callExpr.find_first_of('('));
  
  KernelInfo::ReductionAccess access;
 
  access.type      = KernelInfo::REDUCTION_TYPE::FUNC;
  access.funcName  = fname;
  access.rightExpr = secondArg;
  access.leftExpr  = leftStr;
  access.dataType  = lhs->getType().getAsString();

  //auto exprHash = kslicer::GetHashOfSourceRange(expr->getSourceRange());
  if (WasNotRewrittenYet(expr) && !IsISPC())
  {
    std::string argsType = "";
    if(numArgs > 0)
    {
      const clang::Expr* firstArgExpr = arg0;
      const clang::QualType qt        = firstArgExpr->getType();
      argsType                        = qt.getAsString();
    }
    
    const std::string leftStr2   = RecursiveRewrite(arg0);
    const std::string rightStr2  = RecursiveRewrite(arg1);
    const std::string localIdStr = m_codeInfo->pShaderCC->LocalIdExpr(m_pCurrKernel->GetDim(), m_pCurrKernel->wgSize);
    const std::string left       = leftStr2 + "Shared[" + localIdStr + "]";
    fname = m_codeInfo->pShaderCC->ReplaceCallFromStdNamespace(fname, argsType);
    ReplaceText(expr->getSourceRange(), left + " = " + fname + "(" + left + ", " + rightStr2 + ")" ); 
    MarkRewritten(expr);
  }
}

void kslicer::FunctionRewriter2::DARExpr_RWTexture(clang::CXXOperatorCallExpr* expr, bool write)
{
  if(!m_kernelMode || m_pCurrKernel == nullptr)
    return;

  const auto currAccess = write ? kslicer::TEX_ACCESS::TEX_ACCESS_WRITE : kslicer::TEX_ACCESS::TEX_ACCESS_READ;
  const auto hash = kslicer::GetHashOfSourceRange(expr->getSourceRange());
  if(m_visitedTexAccessNodes.find(hash) != m_visitedTexAccessNodes.end())
    return;

  std::string objName = GetRangeSourceCode(clang::SourceRange(expr->getExprLoc()), m_compiler);
  
  // (1) process if member access
  //
  auto pMember = m_codeInfo->allDataMembers.find(objName);
  if(pMember != m_codeInfo->allDataMembers.end())
  {
    pMember->second.tmask = kslicer::TEX_ACCESS(int(pMember->second.tmask) | int(currAccess));
    auto p = m_pCurrKernel->texAccessInMemb.find(objName);
    if(p != m_pCurrKernel->texAccessInMemb.end())
      p->second = kslicer::TEX_ACCESS( int(p->second) | int(currAccess));
    else
    m_pCurrKernel->texAccessInMemb[objName] = currAccess;
  }
  
  // (2) process if kernel argument access
  //
  for(const auto& arg : m_pCurrKernel->args)
  {
    if(arg.name == objName)
    {
      auto p = m_pCurrKernel->texAccessInArgs.find(objName);
      if(p != m_pCurrKernel->texAccessInArgs.end())
        p->second = kslicer::TEX_ACCESS( int(p->second) | int(currAccess));
      else
      m_pCurrKernel->texAccessInArgs[objName] = currAccess;
    }
  }

  m_visitedTexAccessNodes.insert(hash);
}

void kslicer::FunctionRewriter2::DARExpr_TextureAccess(clang::CXXOperatorCallExpr* expr)
{
  if(!m_kernelMode || m_pCurrKernel == nullptr)
    return;

  std::string op = GetRangeSourceCode(clang::SourceRange(expr->getOperatorLoc()), m_compiler); 
  std::string debugText = GetRangeSourceCode(expr->getSourceRange(), m_compiler);     
  if(expr->isAssignmentOp()) // detect a_brightPixels[coord] = color;
  {
    clang::Expr* left = expr->getArg(0); 
    if(clang::isa<clang::CXXOperatorCallExpr>(left))
    {
      clang::CXXOperatorCallExpr* leftOp = clang::dyn_cast<clang::CXXOperatorCallExpr>(left);
      std::string op2 = GetRangeSourceCode(clang::SourceRange(leftOp->getOperatorLoc()), m_compiler);  
      if(op2 == "]" || op2 == "[" || op2 == "[]")
        DARExpr_RWTexture(leftOp, true);
    }
  }
  else if(op == "]" || op == "[" || op == "[]")
    DARExpr_RWTexture(expr, false);
}

void kslicer::FunctionRewriter2::DARExpr_TextureAccess(clang::BinaryOperator* expr)
{
  if(!m_kernelMode || m_pCurrKernel == nullptr)
    return;

  std::string op = GetRangeSourceCode(clang::SourceRange(expr->getOperatorLoc()), m_compiler); 
  std::string debugText = GetRangeSourceCode(expr->getSourceRange(), m_compiler);     
  if(expr->isAssignmentOp()) // detect a_brightPixels[coord] = color;
  {
    clang::Expr* left = expr->getLHS(); 
    if(clang::isa<clang::CXXOperatorCallExpr>(left))
    {
      clang::CXXOperatorCallExpr* leftOp = clang::dyn_cast<clang::CXXOperatorCallExpr>(left);
      std::string op2 = GetRangeSourceCode(clang::SourceRange(leftOp->getOperatorLoc()), m_compiler);  
      if(op2 == "]" || op2 == "[" || op2 == "[]")
        DARExpr_RWTexture(leftOp, true);
    }
  }
  else if((op == "]" || op == "[" || op == "[]") && clang::isa<clang::CXXOperatorCallExpr>(expr))
    DARExpr_RWTexture(clang::dyn_cast<clang::CXXOperatorCallExpr>(expr), false);
}

void kslicer::FunctionRewriter2::DARExpr_TextureAccess(clang::CXXMemberCallExpr* call)
{
  if(!m_kernelMode || m_pCurrKernel == nullptr)
    return;

  clang::CXXMethodDecl* fDecl = call->getMethodDecl();  
  if(fDecl == nullptr)  
    return;

  //std::string debugText = GetRangeSourceCode(call->getSourceRange(), m_compiler); 
  std::string fname     = fDecl->getNameInfo().getName().getAsString();
  clang::Expr* pTexName =	call->getImplicitObjectArgument(); 
  std::string objName   = GetRangeSourceCode(pTexName->getSourceRange(), m_compiler);     

  if(fname == "sample" || fname == "Sample")
  {
    auto samplerArg         = call->getArg(0);
    std::string samplerName = GetRangeSourceCode(samplerArg->getSourceRange(), m_compiler); 

    // (1) process if member access
    //
    auto pMember = m_codeInfo->allDataMembers.find(objName);
    if(pMember != m_codeInfo->allDataMembers.end())
    {
      pMember->second.tmask   = kslicer::TEX_ACCESS( int(pMember->second.tmask) | int(kslicer::TEX_ACCESS::TEX_ACCESS_SAMPLE));
      auto p = m_pCurrKernel->texAccessInMemb.find(objName);
      if(p != m_pCurrKernel->texAccessInMemb.end())
      {
        p->second = kslicer::TEX_ACCESS( int(p->second) | int(kslicer::TEX_ACCESS::TEX_ACCESS_SAMPLE));
        m_pCurrKernel->texAccessSampler[objName] = samplerName;
      }
      else
      {
        m_pCurrKernel->texAccessInMemb [objName] = kslicer::TEX_ACCESS::TEX_ACCESS_SAMPLE;
        m_pCurrKernel->texAccessSampler[objName] = samplerName;
      }
    }
    
    // (2) process if kernel argument access
    //
    for(const auto& arg : m_pCurrKernel->args)
    {
      if(arg.name == objName)
      {
        auto p = m_pCurrKernel->texAccessInArgs.find(objName);
        if(p != m_pCurrKernel->texAccessInArgs.end())
        {
          p->second = kslicer::TEX_ACCESS( int(p->second) | int(kslicer::TEX_ACCESS::TEX_ACCESS_SAMPLE));
          m_pCurrKernel->texAccessSampler[objName] = samplerName;
        }
        else
        {
          m_pCurrKernel->texAccessInArgs[objName]  = kslicer::TEX_ACCESS::TEX_ACCESS_SAMPLE;
          m_pCurrKernel->texAccessSampler[objName] = samplerName;
        }
      }
    }
  }

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::string kslicer::KernelRewriter2::RecursiveRewrite(const clang::Stmt* expr)
{
  if(expr == nullptr)
    return "";

  KernelRewriter2 rvCopy = *this;
  rvCopy.TraverseStmt(const_cast<clang::Stmt*>(expr));
  
  return kslicer::GetRewrittenText(m_rewriter, expr->getSourceRange());
}

bool kslicer::KernelRewriter2::VisitUnaryOperator_Impl(clang::UnaryOperator* expr)                   { return m_pFunRW2->VisitUnaryOperator_Impl(expr); }
bool kslicer::KernelRewriter2::VisitCompoundAssignOperator_Impl(clang::CompoundAssignOperator* expr) { return m_pFunRW2->VisitCompoundAssignOperator_Impl(expr); }
bool kslicer::KernelRewriter2::VisitCXXOperatorCallExpr_Impl(clang::CXXOperatorCallExpr* expr)       { return m_pFunRW2->VisitCXXOperatorCallExpr_Impl(expr); }

bool kslicer::KernelRewriter2::VisitBinaryOperator_Impl(clang::BinaryOperator* expr)     { return m_pFunRW2->VisitBinaryOperator_Impl(expr); }
bool kslicer::KernelRewriter2::VisitVarDecl_Impl(clang::VarDecl* decl)                   { return m_pFunRW2->VisitVarDecl_Impl(decl); }
bool kslicer::KernelRewriter2::VisitCStyleCastExpr_Impl(clang::CStyleCastExpr* cast)     { return m_pFunRW2->VisitCStyleCastExpr_Impl(cast); }
bool kslicer::KernelRewriter2::VisitImplicitCastExpr_Impl(clang::ImplicitCastExpr* cast) { return m_pFunRW2->VisitImplicitCastExpr_Impl(cast); }

bool kslicer::KernelRewriter2::VisitDeclRefExpr_Impl(clang::DeclRefExpr* expr)           {  return m_pFunRW2->VisitDeclRefExpr_Impl(expr); }
bool kslicer::KernelRewriter2::VisitFloatingLiteral_Impl(clang::FloatingLiteral* expr)   {  return m_pFunRW2->VisitFloatingLiteral_Impl(expr); }

bool kslicer::KernelRewriter2::VisitDeclStmt_Impl(clang::DeclStmt* decl)                          { return m_pFunRW2->VisitDeclStmt_Impl(decl); }
bool kslicer::KernelRewriter2::VisitArraySubscriptExpr_Impl(clang::ArraySubscriptExpr* arrayExpr) { return m_pFunRW2->VisitArraySubscriptExpr_Impl(arrayExpr); }
bool kslicer::KernelRewriter2::VisitUnaryExprOrTypeTraitExpr_Impl(clang::UnaryExprOrTypeTraitExpr* szOfExpr) { return m_pFunRW2->VisitUnaryExprOrTypeTraitExpr_Impl(szOfExpr); }
bool kslicer::KernelRewriter2::VisitMemberExpr_Impl(clang::MemberExpr* expr) 
{ 
  return m_pFunRW2->VisitMemberExpr_Impl(expr); 
}

bool kslicer::KernelRewriter2::VisitCXXMemberCallExpr_Impl(clang::CXXMemberCallExpr* call)
{
  return m_pFunRW2->VisitCXXMemberCallExpr_Impl(call); 
}

bool kslicer::KernelRewriter2::VisitCXXConstructExpr_Impl(clang::CXXConstructExpr* call)
{
  return m_pFunRW2->VisitCXXConstructExpr_Impl(call); 
}

bool kslicer::KernelRewriter2::VisitCallExpr_Impl(clang::CallExpr* call)
{
  const bool res = m_pFunRW2->VisitCallExpr_Impl(call);
  m_pFunRW2->AppendDefaultArgs(call); // 'VisitCallExpr' of FunctionRewriter is not called for kernels and member functions
  return res;
}

bool kslicer::KernelRewriter2::VisitReturnStmt_Impl(clang::ReturnStmt* ret)
{
  return m_pFunRW2->VisitReturnStmt_Impl(ret); 
}

bool kslicer::FunctionRewriter2::GetUintAliasWord(const clang::Expr* a_lvalue, std::string& a_bufferName, std::string& a_wordIndex)
{
  const clang::ASTContext& astContext = m_compiler.getASTContext();

  uint64_t offset = 0;
  bool     pathOk = true;
  const clang::Expr* base = kslicer::GetMemberPathOffset(a_lvalue, astContext, offset, pathOk);
  if(!pathOk)
    return false;

  const clang::Expr* bufferExpr = nullptr;
  const clang::Expr* indexExpr  = nullptr;
  if(auto arrayExpr = clang::dyn_cast<clang::ArraySubscriptExpr>(base))        // 'a_out[i]', 'a_out' is a pointer argument
  {
    bufferExpr = arrayExpr->getBase()->IgnoreParenImpCasts();
    indexExpr  = arrayExpr->getIdx();
    if(!clang::isa<clang::DeclRefExpr>(bufferExpr) && !clang::isa<clang::MemberExpr>(bufferExpr))
      return false;
  }
  else if(auto opCall = clang::dyn_cast<clang::CXXOperatorCallExpr>(base))     // 'm_vec[i]', 'm_vec' is std::vector member
  {
    if(opCall->getOperator() != clang::OO_Subscript || opCall->getNumArgs() != 2 || kslicer::GetHashMapSubscript(opCall) != nullptr)
      return false;
    bufferExpr = opCall->getArg(0)->IgnoreParenImpCasts();
    indexExpr  = opCall->getArg(1);
    if(!clang::isa<clang::MemberExpr>(bufferExpr))
      return false;
  }
  else
    return false;

  const clang::QualType elemType = base->getType().getNonReferenceType();
  const uint64_t elemSize = astContext.getTypeSizeInChars(elemType).getQuantity();
  if(elemSize == 0 || elemSize % 4 != 0 || offset % 4 != 0)
    return false;

  a_bufferName = RecursiveRewrite(bufferExpr);
  a_wordIndex  = "uint(" + RecursiveRewrite(indexExpr) + ")";
  if(elemSize != 4)
    a_wordIndex += "*" + std::to_string(elemSize/4);
  if(offset != 0)
    a_wordIndex += " + " + std::to_string(offset/4);
  return true;
}
