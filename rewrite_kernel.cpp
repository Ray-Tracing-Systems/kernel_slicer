#include "kslicer.h"
#include "class_gen.h"
#include "ast_matchers.h"

#include <sstream>
#include <algorithm>

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool kslicer::KernelRewriter::VisitForStmt(clang::ForStmt* forLoop)
{
  if(!m_infoPass) // we find nodes only during info pass
    return true;
  
  const clang::Stmt* loopStart  = forLoop->getInit();
  const clang::Expr* loopStride =	forLoop->getInc();
  const clang::Expr* loopSize   = forLoop->getCond(); 
  const clang::Stmt* loopBody   = forLoop->getBody();
  
  if(!clang::isa<clang::DeclStmt>(loopStart))
    return true;
  const clang::DeclStmt* initVarDS = clang::dyn_cast<clang::DeclStmt>(loopStart);
  const clang::Decl*     initVarD  = initVarDS->getSingleDecl();
  if(!clang::isa<clang::VarDecl>(initVarD))
    return true;  
  const clang::VarDecl* initVar        = clang::dyn_cast<clang::VarDecl>(initVarD);
  const clang::SourceRange startRange  = initVar->getAnyInitializer()->getSourceRange();
  const clang::SourceRange sizeRange   = loopSize->getSourceRange();
  const clang::SourceRange strideRange = loopStride->getSourceRange();

  const std::string startText  = kslicer::GetRangeSourceCode(startRange, m_compiler);
  const std::string sizeText   = kslicer::GetRangeSourceCode(sizeRange, m_compiler);
  const std::string strideText = kslicer::GetRangeSourceCode(strideRange, m_compiler);

  std::string opCodeStr = "<";
  if(clang::isa<clang::BinaryOperator>(loopSize))
  {
    const clang::BinaryOperator* opCompare = clang::dyn_cast<const clang::BinaryOperator>(loopSize);
    opCodeStr = opCompare->getOpcodeStr();
  }

  for(auto& loop : m_currKernel.loopIters)
  {
    if(loop.startText != startText || loop.condTextOriginal != sizeText || loop.iterTextOriginal != strideText)
      continue;
    loop.startNode  = initVar->getAnyInitializer();
    loop.sizeNode   = loopSize;
    loop.strideNode = loopStride;
    loop.bodyNode   = loopBody;
    if(opCodeStr == "<=")
      loop.condKind = kslicer::KernelInfo::IPV_LOOP_KIND::LOOP_KIND_LESS_EQUAL;
    else
      loop.condKind = kslicer::KernelInfo::IPV_LOOP_KIND::LOOP_KIND_LESS;
  }

  return true;
}


bool kslicer::CheckSettersAccess(const clang::MemberExpr* expr, const MainClassInfo* a_codeInfo, const clang::CompilerInstance& a_compiler,
                                 std::string* setterS, std::string* setterM)
{
  const clang::Expr* baseExpr1 = expr->getBase(); 

  if(!clang::isa<clang::MemberExpr>(baseExpr1))
     return false;

  const clang::MemberExpr* baseExpr = clang::dyn_cast<const clang::MemberExpr>(baseExpr1);

  clang::ValueDecl* pValueDecl1 = expr->getMemberDecl();
  clang::ValueDecl* pValueDecl2 = baseExpr->getMemberDecl();
  
  if(!clang::isa<clang::FieldDecl>(pValueDecl1) || !clang::isa<clang::FieldDecl>(pValueDecl2))
    return false;

  clang::FieldDecl* pFieldDecl1  = clang::dyn_cast<clang::FieldDecl>(pValueDecl1);
  clang::FieldDecl* pFieldDecl2  = clang::dyn_cast<clang::FieldDecl>(pValueDecl2);

  clang::RecordDecl* pRecodDecl1 = pFieldDecl1->getParent();
  clang::RecordDecl* pRecodDecl2 = pFieldDecl2->getParent();

  const std::string typeName1 = pRecodDecl1->getNameAsString(); // MyInOut
  const std::string typeName2 = pRecodDecl2->getNameAsString(); // Main class name

  const std::string debugText1 = kslicer::GetRangeSourceCode(expr->getSourceRange(), a_compiler);     // m_out.summ
  const std::string debugText2 = kslicer::GetRangeSourceCode(baseExpr->getSourceRange(), a_compiler); // m_out
  
  if(typeName2 != a_codeInfo->mainClassName)
    return false;
  auto p = a_codeInfo->m_setterVars.find(debugText2);
  if(p == a_codeInfo->m_setterVars.end())
    return false;

  if(setterS != nullptr)
    (*setterS) = debugText2;
  if(setterM != nullptr)
    (*setterM) = debugText1.substr(debugText1.find(".")+1, debugText1.size());
  return true;
}

bool kslicer::KernelRewriter::NeedToRewriteMemberExpr(const clang::MemberExpr* expr, std::string& out_text)
{
  if(m_infoPass)
    return false;
  clang::ValueDecl* pValueDecl = expr->getMemberDecl();
  if(!isa<FieldDecl>(pValueDecl))
    return false;

  clang::FieldDecl* pFieldDecl   = dyn_cast<FieldDecl>(pValueDecl);
  clang::RecordDecl* pRecodDecl  = pFieldDecl->getParent();
  const std::string thisTypeName = pRecodDecl->getNameAsString();
  
  if(!WasNotRewrittenYet(expr))
    return false;

  // (1) setter access
  //
  std::string setter, containerName;
  if(CheckSettersAccess(expr, m_codeInfo, m_compiler, &setter, &containerName)) // process setter access
  {
    out_text = setter + "_" + containerName;
    return true; 
  }

  // (2) *payload ==>  payload[fakeOffset],  RTV, process access to arguments payload->xxx
  //
  if(thisTypeName != m_mainClassName)
  {
    Expr* baseExpr = expr->getBase(); 
    assert(baseExpr != nullptr);

    const std::string baseName = GetRangeSourceCode(baseExpr->getSourceRange(), m_compiler);

    size_t foundId  = size_t(-1);
    bool needOffset = false;
    for(size_t i=0;i<m_args.size();i++)
    {
      if(m_args[i].name == baseName)
      {
        foundId    = i;
        needOffset = m_args[i].needFakeOffset;
        break;
      }
    }
    
    bool isKernel = m_codeInfo->IsKernel(m_currKernel.name) && !processFuncMember;

    if(foundId != size_t(-1)) // else we didn't found 'payload' in kernel arguments, so just ignore it
    {
      // now split 'payload->xxx' to 'payload' (baseName) and 'xxx' (memberName); 
      // 
      const std::string exprContent = GetRangeSourceCode(expr->getSourceRange(), m_compiler);
      auto pos = exprContent.find("->");
      assert(pos != std::string::npos);
      const std::string memberName = exprContent.substr(pos+2);
  
      if(m_codeInfo->megakernelRTV && m_codeInfo->pShaderCC->IsGLSL()) 
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
    else if(!isKernel && m_codeInfo->pShaderCC->IsGLSL()) // for common member functions
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
  
  //if(thisTypeName != m_mainClassName)
  //  return false;

  // (3) member ==> ubo.member
  //
  const std::string fieldName = pFieldDecl->getNameAsString(); 
  const auto pMember = m_variables.find(fieldName);
  if(pMember == m_variables.end())
    return false;

  //auto exprHash = kslicer::GetHashOfSourceRange(expr->getSourceRange());

  // (2) put ubo->var instead of var, leave containers as they are
  // process arrays and large data structures because small can be read once in the beggining of kernel
  // // m_currKernel.hasInitPass &&
  const bool isInLoopInitPart   = !m_codeInfo->IsRTV() && (expr->getSourceRange().getEnd()   < m_currKernel.loopInsides.getBegin());
  const bool isInLoopFinishPart = !m_codeInfo->IsRTV() && (expr->getSourceRange().getBegin() > m_currKernel.loopInsides.getEnd());
  const bool hasLargeSize     = true; // (pMember->second.sizeInBytes > kslicer::READ_BEFORE_USE_THRESHOLD);
  const bool inMegaKernel     = m_codeInfo->megakernelRTV;
  const bool subjectedToRed   = m_currKernel.subjectedToReduction.find(fieldName) != m_currKernel.subjectedToReduction.end();
  if(!pMember->second.isContainer && WasNotRewrittenYet(expr) && !m_infoPass && (isInLoopInitPart || isInLoopFinishPart || !subjectedToRed) && 
                                                                 (isInLoopInitPart || isInLoopFinishPart || pMember->second.isArray || hasLargeSize || inMegaKernel)) 
  {
    out_text = m_codeInfo->pShaderCC->UBOAccess(pMember->second.name);
    clang::SourceRange thisRng = expr->getSourceRange();
    clang::SourceRange endkRng = m_currKernel.loopOutsidesFinish;
    if(thisRng.getEnd() == endkRng.getEnd()) // fixing stnrange bug
      kslicer::PrintError("possible end-of-loop bug", thisRng, m_compiler.getSourceManager());
    return true;
  }

  return false;
}


bool kslicer::KernelRewriter::VisitMemberExpr_Impl(clang::MemberExpr* expr)
{
  if(m_infoPass) // don't have to rewrite during infoPass
  {
    std::string setter, containerName;
    if(CheckSettersAccess(expr, m_codeInfo, m_compiler, &setter, &containerName))
    {
      clang::QualType qt = expr->getType(); // 
      kslicer::UsedContainerInfo container;
      container.type     = qt.getAsString();
      container.name     = setter + "_" + containerName;            
      container.kind     = kslicer::GetKindOfType(qt);
      container.isConst  = qt.isConstQualified();
      container.isSetter = true;
      container.setterPrefix = setter;
      container.setterSuffix = containerName;
      m_currKernel.usedContainers[container.name] = container;
    }
    return true; 
  }

  std::string rewrittenText;
  if(NeedToRewriteMemberExpr(expr, rewrittenText))
  {
    m_rewriter.ReplaceText(expr->getSourceRange(), rewrittenText);
    MarkRewritten(expr);
  }

  return true;
}

std::string kslicer::KernelRewriter::FunctionCallRewrite(const CallExpr* call)
{
  const FunctionDecl* fDecl = call->getDirectCallee();  
  if(fDecl == nullptr)             // definitely can't process nullpointer 
    return "[KernelRewriter::FunctionCallRewrite_ERROR]";
 
  std::string textRes = fDecl->getNameInfo().getName().getAsString(); //m_codeInfo->pShaderCC->ReplaceCallFromStdNamespace(fname, argsType);
      
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

std::string kslicer::KernelRewriter::FunctionCallRewriteNoName(const clang::CXXConstructExpr* call)
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

std::string kslicer::KernelRewriter::FunctionCallRewrite(const CXXConstructExpr* call)
{
  std::string textRes = call->getConstructor()->getNameInfo().getName().getAsString();
  return textRes + FunctionCallRewriteNoName(call);
}

bool kslicer::KernelRewriter::VisitCallExpr_Impl(CallExpr* call)
{
  if(m_infoPass) // don't have to rewrite during infoPass
    return true; 

  if(isa<CXXMemberCallExpr>(call) || isa<CXXConstructExpr>(call)) // process else-where
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

std::string kslicer::KernelRewriter::VectorTypeContructorReplace(const std::string& fname, const std::string& callText)
{
  return std::string("make_") + fname + callText;
}

bool kslicer::KernelRewriter::VisitCXXConstructExpr_Impl(CXXConstructExpr* call) // #TODO: seems it forks only for clspv, refactor this function please
{
  if(m_infoPass) // don't have to rewrite during infoPass
    return true; 

  CXXConstructorDecl* ctorDecl = call->getConstructor();
  assert(ctorDecl != nullptr);
  
  // Get name of function
  //
  const DeclarationNameInfo dni = ctorDecl->getNameInfo();
  const DeclarationName dn      = dni.getName();
  const std::string fname       = dn.getAsString();

  bool needReplacement = kslicer::IsVectorContructorNeedsReplacement(fname);
  bool wasNotDone      = WasNotRewrittenYet(call);

  if(needReplacement && wasNotDone && !ctorDecl->isCopyOrMoveConstructor()) // 
  {
    const std::string textOrig = GetRangeSourceCode(call->getSourceRange(), m_compiler);
    const std::string text     = FunctionCallRewriteNoName(call);
    const std::string textRes  = VectorTypeContructorReplace(fname, text);

    if(isa<CXXTemporaryObjectExpr>(call) || IsGLSL())
    {
      m_rewriter.ReplaceText(call->getSourceRange(), textRes);
    }
    else
    {
      auto pos1 = textOrig.find_first_of("{");
      auto pos2 = textOrig.find_first_of("(");
      auto pos  = std::min(pos1, pos2);
      const std::string varName = textOrig.substr(0, pos);
      m_rewriter.ReplaceText(call->getSourceRange(), varName + " = " + textRes);
    }
    
    MarkRewritten(call);
  }

  return true;
}


bool kslicer::KernelRewriter::VisitCXXMemberCallExpr_Impl(CXXMemberCallExpr* f)
{
  if(m_infoPass) // don't have to rewrite during infoPass
  {
    DetectTextureAccess(f);
    return true; 
  }

  // Get name of function
  //
  const DeclarationNameInfo dni = f->getMethodDecl()->getNameInfo();
  const DeclarationName dn      = dni.getName();
  const std::string fname       = dn.getAsString();
  
  // Get name of "this" type; we should check wherther this member is std::vector<T>  
  //
  const clang::QualType qt = f->getObjectType();
  const auto& thisTypeName = qt.getAsString();
  CXXRecordDecl* typeDecl  = f->getRecordDecl(); 

  const bool isVector = (typeDecl != nullptr && isa<ClassTemplateSpecializationDecl>(typeDecl)) && thisTypeName.find("vector<") != std::string::npos; 
  //const auto exprHash = kslicer::GetHashOfSourceRange(f->getSourceRange());

  if(isVector && WasNotRewrittenYet(f))
  {
    const std::string exprContent = GetRangeSourceCode(f->getSourceRange(), m_compiler);
    const auto posOfPoint         = exprContent.find(".");
    const std::string memberNameA = exprContent.substr(0, posOfPoint);

    if(fname == "size" || fname == "capacity")
    {
      const std::string memberNameB = memberNameA + "_" + fname;
      m_rewriter.ReplaceText(f->getSourceRange(), m_codeInfo->pShaderCC->UBOAccess(memberNameB) );
      MarkRewritten(f);
    }
    else if(fname == "resize")
    {
      if(f->getSourceRange().getBegin() <= m_currKernel.loopOutsidesInit.getEnd()) // TODO: SEEMS INCORECT LOGIC
      {
        assert(f->getNumArgs() == 1);
        const Expr* currArgExpr  = f->getArgs()[0];
        std::string newSizeValue = RecursiveRewrite(currArgExpr); 
        std::string memberNameB  = memberNameA + "_size = " + newSizeValue;
        m_rewriter.ReplaceText(f->getSourceRange(), m_codeInfo->pShaderCC->UBOAccess(memberNameB) );
        MarkRewritten(f);
      }
    }
    else if(fname == "push_back")
    {
      assert(f->getNumArgs() == 1);
      const Expr* currArgExpr  = f->getArgs()[0];
      std::string newElemValue = RecursiveRewrite(currArgExpr);

      std::string memberNameB  = memberNameA + "_size";
      std::string resulingText = m_codeInfo->pShaderCC->RewritePushBack(memberNameA, memberNameB, newElemValue);
      m_rewriter.ReplaceText(f->getSourceRange(), resulingText);
      MarkRewritten(f);
    }
    else if(fname == "data")
    {
      m_rewriter.ReplaceText(f->getSourceRange(), memberNameA);
      MarkRewritten(f);
    }
    else 
    {
      kslicer::PrintError(std::string("Unsuppoted std::vector method") + fname, f->getSourceRange(), m_compiler.getSourceManager());
    }
  }
  else if(thisTypeName == "struct ISceneObject" && fname.find("RayQuery_") != std::string::npos && WasNotRewrittenYet(f))
  {
    const std::string exprContent = GetRangeSourceCode(f->getSourceRange(), m_compiler);
    const auto posOfPoint         = exprContent.find("->");
    const std::string memberNameA = exprContent.substr(0, posOfPoint);
    //const std::string tail        = exprContent.substr(posOfPoint+2);
    std::string resCallText = memberNameA + "_" + fname + "(";
    for(unsigned i=0;i<f->getNumArgs(); i++)
    {
      resCallText += RecursiveRewrite(f->getArg(i));
      if(i!=f->getNumArgs()-1)
        resCallText += ", ";
    }
    resCallText += ")";
    //m_rewriter.ReplaceText(f->getSourceRange(), memberNameA + "_" + tail);
    m_rewriter.ReplaceText(f->getSourceRange(), resCallText);
    MarkRewritten(f);
  }

  return true;
}

bool kslicer::KernelRewriter::VisitReturnStmt_Impl(ReturnStmt* ret)
{
  Expr* retExpr = ret->getRetValue();
  if (!retExpr)
    return true;
  
  if(processFuncMember) // don't rewrite return statements for function members
    return true;
  
  if(!m_infoPass && WasNotRewrittenYet(ret) && m_kernelIsBoolTyped && !m_codeInfo->megakernelRTV)
  {
    std::string retExprText = RecursiveRewrite(retExpr);
    m_rewriter.ReplaceText(ret->getSourceRange(), std::string("kgenExitCond = ") + retExprText + ";"); // "; goto KGEN_EPILOG"); !!! GLSL DOE NOT SUPPPRT GOTOs!!!
    MarkRewritten(ret);
    return true;
  }

  clang::Expr* pRetExpr = ret->getRetValue();
  if(!isa<clang::CallExpr>(pRetExpr))
    return true;
  
  clang::CallExpr* callExpr = dyn_cast<CallExpr>(pRetExpr);

  // assotiate this buffer with target hierarchy 
  //
  auto retQt = pRetExpr->getType();
  if(retQt->isPointerType())
    retQt = retQt->getPointeeType();

  std::string fname = callExpr->getDirectCallee()->getNameInfo().getName().getAsString();
  if(fname != "MakeObjPtr")
    return true;

  assert(callExpr->getNumArgs() == 2);
  const Expr* firstArgExpr  = callExpr->getArgs()[0];
  const Expr* secondArgExpr = callExpr->getArgs()[1];
 
  if(m_infoPass) // don't have to rewrite during infoPass
  {
    std::string retTypeName = kslicer::CutOffStructClass(retQt.getAsString());
    
    // get ObjData buffer name
    //
    std::string makerObjBufferName = kslicer::GetRangeSourceCode(secondArgExpr->getSourceRange(), m_compiler);
    ReplaceFirst(makerObjBufferName, ".data()", "");

    for(auto& h : m_codeInfo->GetDispatchingHierarchies())
    {
      if(h.second.interfaceName == retTypeName)
      {
        h.second.objBufferName = makerObjBufferName;       
        break;
      }
    }
  }
  else if(WasNotRewrittenYet(ret) && m_kernelIsMaker && !m_codeInfo->megakernelRTV)
  { 
    // change 'return MakeObjPtr(objPtr, ObjData) to 'kgen_objPtr = objPtr'
    //
    std::string retExprText = RecursiveRewrite(firstArgExpr);
    m_rewriter.ReplaceText(ret->getSourceRange(), std::string("{ kgen_objPtr = ") + retExprText + "; }"); // goto KGEN_EPILOG;  
    MarkRewritten(ret);
  }

  return true;
}

bool kslicer::KernelRewriter::NameNeedsFakeOffset(const std::string& a_name) const
{
   bool exclude = false;
   for(auto arg : m_currKernel.args)
   {
     if(arg.needFakeOffset && arg.name == a_name)
       exclude = true;
   }
   return exclude;
}


bool kslicer::KernelRewriter::CheckIfExprHasArgumentThatNeedFakeOffset(const std::string& exprStr)
{
  bool needOffset = false;
  for(const auto arg: m_args)
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

bool kslicer::KernelRewriter::VisitUnaryOperator_Impl(UnaryOperator* expr)
{
  Expr* subExpr =	expr->getSubExpr();
  if(subExpr == nullptr)
    return true;

  const auto op = expr->getOpcodeStr(expr->getOpcode());
  if(op == "++" || op == "--") // detect ++ and -- for reduction
  {
    auto opRange = expr->getSourceRange();
    if(opRange.getEnd()   <= m_currKernel.loopInsides.getBegin() || 
       opRange.getBegin() >= m_currKernel.loopInsides.getEnd() ) // not inside loop
      return true;     
    
    const auto op = expr->getOpcodeStr(expr->getOpcode());
    std::string leftStr = GetRangeSourceCode(subExpr->getSourceRange(), m_compiler);

    auto p = m_currKernel.usedMembers.find(leftStr);
    if(p != m_currKernel.usedMembers.end() && WasNotRewrittenYet(expr))
    {
      KernelInfo::ReductionAccess access;
      access.type      = KernelInfo::REDUCTION_TYPE::UNKNOWN;
      access.rightExpr = "";
      access.leftExpr  = leftStr;
      access.dataType  = subExpr->getType().getAsString();

      if(op == "++")
        access.type    = KernelInfo::REDUCTION_TYPE::ADD_ONE;
      else if(op == "--")
        access.type    = KernelInfo::REDUCTION_TYPE::SUB_ONE;

      if(m_infoPass)
      {
        m_currKernel.hasFinishPass = m_currKernel.hasFinishPass || !access.SupportAtomicLastStep(); // if atomics can not be used, we must insert additional finish pass
        m_currKernel.subjectedToReduction[leftStr] = access;
      }
      else
      {
        std::string leftStr2   = RecursiveRewrite(expr->getSubExpr()); 
        std::string localIdStr = m_codeInfo->pShaderCC->LocalIdExpr(m_currKernel.GetDim(), m_currKernel.wgSize);
        m_rewriter.ReplaceText(expr->getSourceRange(), leftStr2 + "Shared[" + localIdStr + "]++");
        MarkRewritten(expr);
      }
    }
  }

  // detect " *(something)"
  //
  if(expr->canOverflow() || op != "*") // -UnaryOperator ...'LiteMath::uint':'unsigned int' lvalue prefix '*' cannot overflow
    return true;

  std::string exprInside = RecursiveRewrite(subExpr);

  // check if this argument actually need fake Offset
  //
  const bool needOffset = CheckIfExprHasArgumentThatNeedFakeOffset(exprInside);
  if(needOffset && WasNotRewrittenYet(expr))
  {
    if(m_codeInfo->megakernelRTV)
      m_rewriter.ReplaceText(expr->getSourceRange(), exprInside);
    else
      m_rewriter.ReplaceText(expr->getSourceRange(), exprInside + "[" + m_fakeOffsetExp + "]");
    MarkRewritten(expr);
  }

  return true;
}

void kslicer::KernelRewriter::ProcessReductionOp(const std::string& op, const Expr* lhs, const Expr* rhs, const Expr* expr)
{
  auto pShaderRewriter = m_codeInfo->pShaderFuncRewriter;
  std::string leftVar = GetRangeSourceCode(lhs->getSourceRange().getBegin(), m_compiler);
  std::string leftStr = GetRangeSourceCode(lhs->getSourceRange(), m_compiler);
  auto p = m_currKernel.usedMembers.find(leftVar);
  if(p != m_currKernel.usedMembers.end())
  {
    KernelInfo::ReductionAccess access;
    access.type      = KernelInfo::REDUCTION_TYPE::UNKNOWN;
    access.rightExpr = GetRangeSourceCode(rhs->getSourceRange(), m_compiler);
    access.leftExpr  = leftStr;
    access.dataType  = rhs->getType().getAsString();
    ReplaceFirst(access.dataType, "const ", "");
    access.dataType = pShaderRewriter->RewriteStdVectorTypeStr(access.dataType); 
   
    if(leftVar != leftStr && isa<ArraySubscriptExpr>(lhs))
    {
      auto lhsArray    = dyn_cast<const ArraySubscriptExpr>(lhs);
      const Expr* idx  = lhsArray->getIdx();  // array index
      const Expr* name = lhsArray->getBase(); // array name

      access.leftIsArray = true;
      access.arraySize   = 0;
      access.arrayIndex  = GetRangeSourceCode(idx->getSourceRange(), m_compiler);
      access.arrayName   = GetRangeSourceCode(name->getSourceRange(), m_compiler); 
      
      // extract array size
      //
      const Expr* nextNode = lhsArray->getLHS();
      
      if(isa<ImplicitCastExpr>(nextNode))
      {
        auto cast = dyn_cast<const ImplicitCastExpr>(nextNode);
        nextNode  = cast->getSubExpr();
      }
      
      if(isa<MemberExpr>(nextNode))
      {
        const MemberExpr* pMemberExpr = dyn_cast<const MemberExpr>(nextNode); 
        const ValueDecl*  valDecl     = pMemberExpr->getMemberDecl();
        const QualType    qt          = valDecl->getType();
        const auto        typePtr     = qt.getTypePtr(); 
        if(typePtr->isConstantArrayType()) 
        {    
          auto arrayType   = dyn_cast<const ConstantArrayType>(typePtr);     
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

    if(m_infoPass)
    {
      m_currKernel.hasFinishPass = m_currKernel.hasFinishPass || !access.SupportAtomicLastStep(); // if atomics can not be used, we must insert additional finish pass
      m_currKernel.subjectedToReduction[leftStr] = access;
    }
    else if(WasNotRewrittenYet(expr))
    {
      std::string rightStr2  = RecursiveRewrite(rhs);
      std::string localIdStr = m_codeInfo->pShaderCC->LocalIdExpr(m_currKernel.GetDim(), m_currKernel.wgSize);
      if(access.leftIsArray)
        m_rewriter.ReplaceText(expr->getSourceRange(), access.arrayName + "Shared[" + access.arrayIndex + "][" + localIdStr + "] " + access.GetOp(m_codeInfo->pShaderCC) + " " + rightStr2);
      else
        m_rewriter.ReplaceText(expr->getSourceRange(), leftVar + "Shared[" + localIdStr + "] " + access.GetOp(m_codeInfo->pShaderCC) + " " + rightStr2);
      MarkRewritten(expr);
    }
  }
}


bool kslicer::KernelRewriter::VisitCompoundAssignOperator_Impl(CompoundAssignOperator* expr)
{
  auto opRange = expr->getSourceRange();
  if(opRange.getEnd()   <= m_currKernel.loopInsides.getBegin() || 
     opRange.getBegin() >= m_currKernel.loopInsides.getEnd() ) // not inside loop
    return true;   

  const Expr* lhs = expr->getLHS();
  const Expr* rhs = expr->getRHS();
  const auto  op  = expr->getOpcodeStr();

  ProcessReductionOp(op.str(), lhs, rhs, expr);

  return true;
}

void kslicer::KernelRewriter::ProcessReadWriteTexture(clang::CXXOperatorCallExpr* expr, bool write)
{
  const auto currAccess = write ? kslicer::TEX_ACCESS::TEX_ACCESS_WRITE : kslicer::TEX_ACCESS::TEX_ACCESS_READ;
  const auto hash = kslicer::GetHashOfSourceRange(expr->getSourceRange());
  if(m_visitedTexAccessNodes.find(hash) != m_visitedTexAccessNodes.end())
    return;

  std::string objName = GetRangeSourceCode(SourceRange(expr->getExprLoc()), m_compiler);
  
  // (1) process if member access
  //
  auto pMember = m_codeInfo->allDataMembers.find(objName);
  if(pMember != m_codeInfo->allDataMembers.end())
  {
    pMember->second.tmask = kslicer::TEX_ACCESS(int(pMember->second.tmask) | int(currAccess));
    auto p = m_currKernel.texAccessInMemb.find(objName);
    if(p != m_currKernel.texAccessInMemb.end())
      p->second = kslicer::TEX_ACCESS( int(p->second) | int(currAccess));
    else
      m_currKernel.texAccessInMemb[objName] = currAccess;
  }
  
  // (2) process if kernel argument access
  //
  for(const auto& arg : m_currKernel.args)
  {
    if(arg.name == objName)
    {
      auto p = m_currKernel.texAccessInArgs.find(objName);
      if(p != m_currKernel.texAccessInArgs.end())
        p->second = kslicer::TEX_ACCESS( int(p->second) | int(currAccess));
      else
        m_currKernel.texAccessInArgs[objName] = currAccess;
    }
  }

  m_visitedTexAccessNodes.insert(hash);
}

void kslicer::KernelRewriter::DetectTextureAccess(clang::CXXMemberCallExpr* call)
{
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
      auto p = m_currKernel.texAccessInMemb.find(objName);
      if(p != m_currKernel.texAccessInMemb.end())
      {
        p->second = kslicer::TEX_ACCESS( int(p->second) | int(kslicer::TEX_ACCESS::TEX_ACCESS_SAMPLE));
        m_currKernel.texAccessSampler[objName] = samplerName;
      }
      else
      {
        m_currKernel.texAccessInMemb [objName] = kslicer::TEX_ACCESS::TEX_ACCESS_SAMPLE;
        m_currKernel.texAccessSampler[objName] = samplerName;
      }
    }
    
    // (2) process if kernel argument access
    //
    for(const auto& arg : m_currKernel.args)
    {
      if(arg.name == objName)
      {
        auto p = m_currKernel.texAccessInArgs.find(objName);
        if(p != m_currKernel.texAccessInArgs.end())
        {
          p->second = kslicer::TEX_ACCESS( int(p->second) | int(kslicer::TEX_ACCESS::TEX_ACCESS_SAMPLE));
          m_currKernel.texAccessSampler[objName] = samplerName;
        }
        else
        {
          m_currKernel.texAccessInArgs[objName]  = kslicer::TEX_ACCESS::TEX_ACCESS_SAMPLE;
          m_currKernel.texAccessSampler[objName] = samplerName;
        }
      }
    }
  }

}

void kslicer::KernelRewriter::DetectTextureAccess(CXXOperatorCallExpr* expr)
{
  std::string op = GetRangeSourceCode(SourceRange(expr->getOperatorLoc()), m_compiler); 
  //std::string debugText = GetRangeSourceCode(expr->getSourceRange(), m_compiler);     
  if(expr->isAssignmentOp()) // detect a_brightPixels[coord] = color;
  {
    clang::Expr* left = expr->getArg(0); 
    if(clang::isa<clang::CXXOperatorCallExpr>(left))
    {
      clang::CXXOperatorCallExpr* leftOp = clang::dyn_cast<clang::CXXOperatorCallExpr>(left);
      std::string op2 = GetRangeSourceCode(SourceRange(leftOp->getOperatorLoc()), m_compiler);  
      if(op2 == "]" || op2 == "[" || op2 == "[]")
        ProcessReadWriteTexture(leftOp, true);
    }
  }
  else if(op == "]" || op == "[" || op == "[]")
    ProcessReadWriteTexture(expr, false);
}

bool kslicer::KernelRewriter::VisitCXXOperatorCallExpr_Impl(CXXOperatorCallExpr* expr)
{
  DetectTextureAccess(expr);
  auto opRange = expr->getSourceRange();
  if(opRange.getEnd()   <= m_currKernel.loopInsides.getBegin() || 
     opRange.getBegin() >= m_currKernel.loopInsides.getEnd() ) // not inside loop
    return true;   

  const auto numArgs = expr->getNumArgs();
  if(numArgs != 2)
    return true; 
  
  std::string op = GetRangeSourceCode(SourceRange(expr->getOperatorLoc()), m_compiler);  
  if(op == "+=" || op == "-=" || op == "*=")
  {
    const Expr* lhs = expr->getArg(0);
    const Expr* rhs = expr->getArg(1);

    ProcessReductionOp(op, lhs, rhs, expr);
  }

  return true;
}

void kslicer::KernelRewriter::DetectTextureAccess(clang::BinaryOperator* expr)
{
  std::string op = GetRangeSourceCode(SourceRange(expr->getOperatorLoc()), m_compiler); 
  //std::string debugText = GetRangeSourceCode(expr->getSourceRange(), m_compiler);     
  if(expr->isAssignmentOp()) // detect a_brightPixels[coord] = color;
  {
    clang::Expr* left = expr->getLHS(); 
    if(clang::isa<clang::CXXOperatorCallExpr>(left))
    {
      clang::CXXOperatorCallExpr* leftOp = clang::dyn_cast<clang::CXXOperatorCallExpr>(left);
      std::string op2 = GetRangeSourceCode(SourceRange(leftOp->getOperatorLoc()), m_compiler);  
      if(op2 == "]" || op2 == "[" || op2 == "[]")
        ProcessReadWriteTexture(leftOp, true);
    }
  }
  //else if(op == "]" || op == "[" || op == "[]")
  //  ProcessReadWriteTexture(expr, false);
}

bool kslicer::KernelRewriter::VisitBinaryOperator_Impl(BinaryOperator* expr) // detect reduction like m_var = F(m_var,expr)
{
  auto opRange = expr->getSourceRange();
  if(!m_codeInfo->IsRTV() && (opRange.getEnd() <= m_currKernel.loopInsides.getBegin() || opRange.getBegin() >= m_currKernel.loopInsides.getEnd())) // not inside loop
    return true;  

  const auto op = expr->getOpcodeStr();
  if(op != "=")
    return true;
  
  DetectTextureAccess(expr);
  const Expr* lhs = expr->getLHS();
  const Expr* rhs = expr->getRHS();

  if(!isa<MemberExpr>(lhs))
    return true;

  std::string leftStr  = GetRangeSourceCode(lhs->getSourceRange(), m_compiler);
  auto p = m_currKernel.usedMembers.find(leftStr);
  if(p == m_currKernel.usedMembers.end())
    return true;
  
  if(!isa<CallExpr>(rhs))
  {
    PrintWarning("unsupported expression for reduction via assigment inside loop; must be 'a = f(a,b)'", rhs->getSourceRange(), m_compiler.getSourceManager());
    PrintWarning("reduction for variable '" +  leftStr + "' will not be generated, sure this is ok for you?", rhs->getSourceRange(), m_compiler.getSourceManager());
    return true;
  }
  
  auto call    = dyn_cast<CallExpr>(rhs);
  auto numArgs = call->getNumArgs();
  if(numArgs != 2)
  {
    PrintError("function which is used in reduction must have 2 args; a = f(a,b)'", expr->getSourceRange(), m_compiler.getSourceManager());
    return true;
  }
  
  const Expr* arg0 = call->getArg(0);
  const Expr* arg1 = call->getArg(1);

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
    PrintError("incorrect arguments of reduction function, one of them must be same as assigment result; a = f(a,b)'", call->getSourceRange(), m_compiler.getSourceManager());
    return true;
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

  if(m_infoPass)
  {
    m_currKernel.hasFinishPass = m_currKernel.hasFinishPass || !access.SupportAtomicLastStep(); // if atomics can not be used, we must insert additional finish pass
    m_currKernel.subjectedToReduction[leftStr] = access;
  }
  else if (WasNotRewrittenYet(expr))
  {
    std::string argsType = "";
    if(call->getNumArgs() > 0)
    {
      const Expr* firstArgExpr = call->getArgs()[0];
      const QualType qt        = firstArgExpr->getType();
      argsType                 = qt.getAsString();
    }
    
    const std::string leftStr2   = RecursiveRewrite(arg0);
    const std::string rightStr2  = RecursiveRewrite(arg1);
    const std::string localIdStr = m_codeInfo->pShaderCC->LocalIdExpr(m_currKernel.GetDim(), m_currKernel.wgSize);
    const std::string left       = leftStr2 + "Shared[" + localIdStr + "]";
    fname = m_codeInfo->pShaderCC->ReplaceCallFromStdNamespace(fname, argsType);
    m_rewriter.ReplaceText(expr->getSourceRange(), left + " = " + fname + "(" + left + ", " + rightStr2 + ")" ); 
    MarkRewritten(expr);
  }
  return true;
}


std::string kslicer::KernelRewriter::RecursiveRewrite(const Stmt* expr)
{
  if(expr == nullptr)
    return "";

  KernelRewriter rvCopy = *this;
  rvCopy.TraverseStmt(const_cast<clang::Stmt*>(expr));
  return m_rewriter.getRewrittenText(expr->getSourceRange());
}
