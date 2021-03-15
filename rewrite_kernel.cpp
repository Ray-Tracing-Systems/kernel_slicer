#include "kslicer.h"
#include "class_gen.h"
#include "ast_matchers.h"

#include <sstream>
#include <algorithm>

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool kslicer::KernelRewriter::VisitMemberExpr(MemberExpr* expr)
{
  if(m_infoPass) // don't have to rewrite during infoPass
    return true; 

  ValueDecl* pValueDecl = expr->getMemberDecl();
  if(!isa<FieldDecl>(pValueDecl))
    return true;

  FieldDecl* pFieldDecl  = dyn_cast<FieldDecl>(pValueDecl);
  assert(pFieldDecl != nullptr);
  RecordDecl* pRecodDecl = pFieldDecl->getParent();
  assert(pRecodDecl != nullptr);

  const std::string thisTypeName = pRecodDecl->getNameAsString();
  if(thisTypeName != m_mainClassName)
  {
    // process access to arguments payload->xxx
    //
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

    if(foundId == size_t(-1)) // we didn't found 'payload' in kernela arguments, so just ignore it
      return true;
    
    // now split 'payload->xxx' to 'payload' (baseName) and 'xxx' (memberName); 
    // 
    const std::string exprContent = GetRangeSourceCode(expr->getSourceRange(), m_compiler);
    auto pos = exprContent.find("->");
    assert(pos != std::string::npos);

    const std::string memberName = exprContent.substr(pos+2);

    if(needOffset && WasNotRewrittenYet(expr))
    {
      m_rewriter.ReplaceText(expr->getSourceRange(), baseName + "[" + m_fakeOffsetExp + "]." + memberName);
      MarkRewritten(expr);
    }

    return true;
  }

  // process access to class member data
  // 

  // (1) get variable offset in buffer by its name 
  //
  const std::string fieldName = pFieldDecl->getNameAsString(); 
  const auto pMember = m_variables.find(fieldName);
  if(pMember == m_variables.end())
    return true;

  auto exprHash = kslicer::GetHashOfSourceRange(expr->getSourceRange());

  // (2) put ubo->var instead of var, leave containers as they are
  // process arrays and large data structures because small can be read once in the neggining of kernel
  //
  const bool isInLoopInitPart = expr->getSourceRange().getBegin() <= m_currKernel.loopOutsidesInit.getEnd();
  const bool hasLargeSize     = (pMember->second.sizeInBytes > kslicer::READ_BEFORE_USE_THRESHOLD);
  if(!pMember->second.isContainer && (isInLoopInitPart || pMember->second.isArray || hasLargeSize) && WasNotRewrittenYet(expr) && !m_infoPass) 
  {
    //const std::string debugMe = GetRangeSourceCode(expr->getSourceRange(), m_compiler);
    std::string rewrittenName = m_codeInfo->pShaderCC->UBOAccess(pMember->second.name);
    m_rewriter.ReplaceText(expr->getSourceRange(), rewrittenName);
    MarkRewritten(expr);
  }
  
  return true;
}

std::string kslicer::KernelRewriter::FunctionCallRewrite(const CallExpr* call)
{
  const FunctionDecl* fDecl = call->getDirectCallee();  
  if(fDecl == nullptr)             // definitely can't process nullpointer 
    return "[FunctionCallRewrite_ERROR]";
 
  std::string textRes = fDecl->getNameInfo().getName().getAsString(); //m_codeInfo->pShaderCC->ReplaceCallFromStdNamespace(fname, argsType);
      
  textRes += "(";
  for(int i=0;i<call->getNumArgs();i++)
  {
    textRes += RecursiveRewrite(call->getArg(i));
    if(i < call->getNumArgs()-1)
      textRes += ",";
  }
  textRes += ")";

  return textRes;
}

std::string kslicer::KernelRewriter::FunctionCallRewrite(const CXXConstructExpr* call)
{
  std::string textRes = call->getConstructor()->getNameInfo().getName().getAsString();
      
  textRes += "(";
  for(int i=0;i<call->getNumArgs();i++)
  {
    textRes += RecursiveRewrite(call->getArg(i));
    if(i < call->getNumArgs()-1)
      textRes += ",";
  }
  textRes += ")";

  return textRes;
}

bool kslicer::KernelRewriter::VisitCallExpr(CallExpr* call)
{
  if(m_infoPass) // don't have to rewrite during infoPass
    return true; 

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

bool kslicer::KernelRewriter::VisitCXXConstructExpr(CXXConstructExpr* call)
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

  if(m_codeInfo->pShaderCC->IsVectorTypeNeedsContructorReplacement(fname) && WasNotRewrittenYet(call) && call->getNumArgs() > 1)
  {
    const std::string text    = FunctionCallRewrite(call);
    const std::string textRes = m_codeInfo->pShaderCC->VectorTypeContructorReplace(fname, text);
    m_rewriter.ReplaceText(call->getSourceRange(), textRes);
    MarkRewritten(call);
  }

  return true;
}


bool kslicer::KernelRewriter::VisitCXXMemberCallExpr(CXXMemberCallExpr* f)
{
  if(m_infoPass) // don't have to rewrite during infoPass
    return true; 

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
  const auto exprHash = kslicer::GetHashOfSourceRange(f->getSourceRange());

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
      if(f->getSourceRange().getBegin() <= m_currKernel.loopOutsidesInit.getEnd())
      {
        assert(f->getNumArgs() == 1);
        const Expr* currArgExpr  = f->getArgs()[0];
        std::string newSizeValue = kslicer::GetRangeSourceCode(currArgExpr->getSourceRange(), m_compiler); 
        std::string memberNameB  = memberNameA + "_size = " + newSizeValue;
        m_rewriter.ReplaceText(f->getSourceRange(), m_codeInfo->pShaderCC->UBOAccess(memberNameB) );
        MarkRewritten(f);
      }
    }
    else if(fname == "push_back")
    {
      assert(f->getNumArgs() == 1);
      const Expr* currArgExpr  = f->getArgs()[0];
      std::string newElemValue = kslicer::GetRangeSourceCode(currArgExpr->getSourceRange(), m_compiler);

      std::string memberNameB  = memberNameA + "_size";
      m_rewriter.ReplaceText(f->getSourceRange(), std::string("{ uint offset = atomic_inc(&") + m_codeInfo->pShaderCC->UBOAccess(memberNameB) + "); " + 
                                                                 memberNameA + "[offset] = " + newElemValue + ";}");
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
 
  return true;
}

bool kslicer::KernelRewriter::VisitReturnStmt(ReturnStmt* ret)
{
  if(m_infoPass) // don't have to rewrite during infoPass
    return true; 
      
  Expr* retExpr = ret->getRetValue();
  if (!retExpr || !m_kernelIsBoolTyped)
    return true;
  
  if(WasNotRewrittenYet(ret))
  {
    std::string retExprText = RecursiveRewrite(retExpr);
    m_rewriter.ReplaceText(ret->getSourceRange(), std::string("kgenExitCond = ") + retExprText + "; goto KGEN_EPILOG");
    MarkRewritten(ret);
  }

  return true;
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

bool kslicer::KernelRewriter::VisitUnaryOperator(UnaryOperator* expr)
{
  const auto op = expr->getOpcodeStr(expr->getOpcode());
  //const auto opCheck = std::string(op);
  //std::string opCheck2 = GetRangeSourceCode(expr->getSourceRange(), m_compiler);

  Expr* subExpr =	expr->getSubExpr();
  if(subExpr == nullptr)
    return true;

  if(op == "++" || op == "--") // detect ++ and -- for reduction
  {
    auto opRange = expr->getSourceRange();
    if(opRange.getEnd() <= m_currKernel.loopInsides.getBegin() || opRange.getBegin() >= m_currKernel.loopInsides.getEnd() ) // not inside loop
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
        std::string localIdStr = m_codeInfo->pShaderCC->LocalIdExpr(m_currKernel.GetDim());
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
    m_rewriter.ReplaceText(expr->getSourceRange(), exprInside + "[" + m_fakeOffsetExp + "]");
    MarkRewritten(expr);
  }

  return true;
}

void kslicer::KernelRewriter::ProcessReductionOp(const std::string& op, const Expr* lhs, const Expr* rhs, const Expr* expr)
{
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
    access.dataType = m_codeInfo->RemoveTypeNamespaces(access.dataType);
   
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
    
    auto exprHash = kslicer::GetHashOfSourceRange(expr->getSourceRange());

    if(m_infoPass)
    {
      m_currKernel.hasFinishPass = m_currKernel.hasFinishPass || !access.SupportAtomicLastStep(); // if atomics can not be used, we must insert additional finish pass
      m_currKernel.subjectedToReduction[leftStr] = access;
    }
    else if(WasNotRewrittenYet(expr))
    {
      std::string leftStr2   = RecursiveRewrite(lhs);
      std::string rightStr2  = RecursiveRewrite(rhs);
      std::string localIdStr = m_codeInfo->pShaderCC->LocalIdExpr(m_currKernel.GetDim());
      m_rewriter.ReplaceText(expr->getSourceRange(), leftStr2 + "Shared[" + localIdStr + "] " + access.GetOp(m_codeInfo->pShaderCC) + " " + rightStr2);
      MarkRewritten(expr);
    }
  }
}


bool kslicer::KernelRewriter::VisitCompoundAssignOperator(CompoundAssignOperator* expr)
{
  auto opRange = expr->getSourceRange();
  if(opRange.getEnd() <= m_currKernel.loopInsides.getBegin() || opRange.getBegin() >= m_currKernel.loopInsides.getEnd() ) // not inside loop
    return true;   

  const Expr* lhs = expr->getLHS();
  const Expr* rhs = expr->getRHS();
  const auto  op  = expr->getOpcodeStr();

  ProcessReductionOp(op.str(), lhs, rhs, expr);

  return true;
}

bool kslicer::KernelRewriter::VisitCXXOperatorCallExpr(CXXOperatorCallExpr* expr)
{
  auto opRange = expr->getSourceRange();
  if(opRange.getEnd() <= m_currKernel.loopInsides.getBegin() || opRange.getBegin() >= m_currKernel.loopInsides.getEnd() ) // not inside loop
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

bool kslicer::KernelRewriter::VisitBinaryOperator(BinaryOperator* expr) // detect reduction like m_var = F(m_var,expr)
{
  auto opRange = expr->getSourceRange();
  if(opRange.getEnd() <= m_currKernel.loopInsides.getBegin() || opRange.getBegin() >= m_currKernel.loopInsides.getEnd() ) // not inside loop
    return true;  

  const auto op = expr->getOpcodeStr();
  if(op != "=")
    return true;
  
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
    PrintError("unsupported expression for reduction via assigment inside loop; must be 'a = f(a,b)'", rhs->getSourceRange(), m_compiler.getSourceManager());
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

  auto exprHash = kslicer::GetHashOfSourceRange(expr->getSourceRange());

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
    const std::string localIdStr = m_codeInfo->pShaderCC->LocalIdExpr(m_currKernel.GetDim());
    const std::string left       = leftStr2 + "Shared[" + localIdStr + "]";
    fname = m_codeInfo->pShaderCC->ReplaceCallFromStdNamespace(fname, argsType);
    m_rewriter.ReplaceText(expr->getSourceRange(), left + " = " + fname + "(" + left + ", " + rightStr2 + ")" ); 
    MarkRewritten(expr);
  }
  return true;
}


std::string kslicer::KernelRewriter::RecursiveRewrite(const Stmt* expr)
{
  std::string debugMeIn = GetRangeSourceCode(expr->getSourceRange(), m_compiler);
  KernelRewriter rvCopy = *this;

  rvCopy.TraverseStmt(const_cast<clang::Stmt*>(expr));
  std::string outRes = m_rewriter.getRewrittenText(expr->getSourceRange());
  return outRes;
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
