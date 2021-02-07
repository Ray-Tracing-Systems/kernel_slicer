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
  ValueDecl* pValueDecl =	expr->getMemberDecl();
  
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

    if(needOffset)
       m_rewriter.ReplaceText(expr->getSourceRange(), baseName + "[" + m_fakeOffsetExp + "]." + memberName);

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

  // (2) put ubo->var instead of var, leave containers as they are
  // process arrays and large data structures because small can be read once in the neggining of kernel
  //
  const bool isInLoopInitPart = expr->getSourceRange().getBegin() <= m_currKernel.loopOutsidesInit.getEnd();
  const bool hasLargeSize     = (pMember->second.sizeInBytes > kslicer::READ_BEFORE_USE_THRESHOLD);
  if(!pMember->second.isContainer && (isInLoopInitPart || pMember->second.isArray || hasLargeSize)) 
  {
    //const std::string debugMe = GetRangeSourceCode(expr->getSourceRange(), m_compiler);
    std::string rewrittenName = m_codeInfo->pShaderCC->UBOAccess(pMember->second.name);
    m_rewriter.ReplaceText(expr->getSourceRange(), rewrittenName);
  }
  
  return true;
}

bool kslicer::KernelRewriter::VisitCXXMemberCallExpr(CXXMemberCallExpr* f)
{
  // Get name of function
  //
  const DeclarationNameInfo dni = f->getMethodDecl()->getNameInfo();
  const DeclarationName dn      = dni.getName();
  const std::string fname       = dn.getAsString();
  
  // Get name of "this" type; we should check wherther this member is std::vector<T>  
  //
  const clang::QualType qt = f->getObjectType();
  const std::string& thisTypeName = qt.getAsString();
  //const auto* fieldTypePtr = qt.getTypePtr(); 
  //assert(fieldTypePtr != nullptr);
  //auto typeDecl = fieldTypePtr->getAsRecordDecl();  
  CXXRecordDecl* typeDecl = f->getRecordDecl(); 

  const bool isVector = (typeDecl != nullptr && isa<ClassTemplateSpecializationDecl>(typeDecl)) && thisTypeName.find("vector<") != std::string::npos; 
  
  if(isVector)
  {
    const std::string exprContent = GetRangeSourceCode(f->getSourceRange(), m_compiler);
    const auto posOfPoint         = exprContent.find(".");
    const std::string memberNameA = exprContent.substr(0, posOfPoint);

    if(fname == "size" || fname == "capacity")
    {
      const std::string memberNameB = memberNameA + "_" + fname;
      m_rewriter.ReplaceText(f->getSourceRange(), m_codeInfo->pShaderCC->UBOAccess(memberNameB) );
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
      }
    }
    else if(fname == "push_back")
    {
      m_rewriter.ReplaceText(f->getSourceRange(), "//replace push_back()");
    }
    else if(fname == "data")
    {
      m_rewriter.ReplaceText(f->getSourceRange(), memberNameA);
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
  Expr* retExpr = ret->getRetValue();
  if (!retExpr || !m_kernelIsBoolTyped)
    return true;

  std::string retExprText = GetRangeSourceCode(retExpr->getSourceRange(), m_compiler);
  m_rewriter.ReplaceText(ret->getSourceRange(), std::string("kgenExitCond = ") + retExprText + "; goto KGEN_EPILOG");
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
    if(p != m_currKernel.usedMembers.end())
    {
      KernelInfo::ReductionAccess access;
      access.type      = KernelInfo::REDUCTION_TYPE::UNKNOWN;
      access.rightExpr = "";
      access.dataType  = subExpr->getType().getAsString();

      if(op == "++")
        access.type    = KernelInfo::REDUCTION_TYPE::ADD_ONE;
      else if(op == "--")
        access.type    = KernelInfo::REDUCTION_TYPE::SUB_ONE;

      m_currKernel.subjectedToReduction[leftStr] = access;
      m_rewriter.ReplaceText(expr->getSourceRange(), leftStr + "Shared[get_local_id(0)]++");
    }
  }

  // detect " *(something)"
  //
  if(expr->canOverflow() || op != "*") // -UnaryOperator ...'LiteMath::uint':'unsigned int' lvalue prefix '*' cannot overflow
    return true;

  std::string exprInside = GetRangeSourceCode(subExpr->getSourceRange(), m_compiler);

  // check if this argument actually need fake Offset
  //
  const bool needOffset = CheckIfExprHasArgumentThatNeedFakeOffset(exprInside);

  if(needOffset)
    m_rewriter.ReplaceText(expr->getSourceRange(), exprInside + "[" + m_fakeOffsetExp + "]");

  return true;
}

bool kslicer::KernelRewriter::VisitCompoundAssignOperator(CompoundAssignOperator* expr)
{
  auto opRange = expr->getSourceRange();
  if(opRange.getEnd() <= m_currKernel.loopInsides.getBegin() || opRange.getBegin() >= m_currKernel.loopInsides.getEnd() ) // not inside loop
    return true;   

  const Expr* lhs     = expr->getLHS();
  const Expr* rhs     = expr->getRHS();
  const auto  op      = expr->getOpcodeStr();
  std::string leftStr = GetRangeSourceCode(lhs->getSourceRange(), m_compiler);

  auto p = m_currKernel.usedMembers.find(leftStr);
  if(p != m_currKernel.usedMembers.end())
  {
    KernelInfo::ReductionAccess access;
    access.type      = KernelInfo::REDUCTION_TYPE::UNKNOWN;
    access.rightExpr = GetRangeSourceCode(rhs->getSourceRange(), m_compiler);
    access.dataType  = rhs->getType().getAsString();

    if(op == "+=")
      access.type    = KernelInfo::REDUCTION_TYPE::ADD;
    else if(op == "*=")
      access.type    = KernelInfo::REDUCTION_TYPE::MUL;
    else if(op == "-=")
      access.type    = KernelInfo::REDUCTION_TYPE::SUB;
      
    m_currKernel.subjectedToReduction[leftStr] = access;
    m_rewriter.ReplaceText(expr->getSourceRange(), leftStr + "Shared[get_local_id(0)] " + access.GetOp() + " " + access.rightExpr);
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

  const std::string callExpr = GetRangeSourceCode(call->getSourceRange(), m_compiler);
  const std::string fname    = callExpr.substr(0, callExpr.find_first_of('('));
  
  KernelInfo::ReductionAccess access;
 
  access.type      = KernelInfo::REDUCTION_TYPE::FUNC;
  access.funcName  = fname;
  access.rightExpr = secondArg;
  access.dataType  = lhs->getType().getAsString();

  m_currKernel.subjectedToReduction[leftStr] = access;
  m_rewriter.ReplaceText(expr->getSourceRange(), "// func. reduce " + leftStr + " with " + fname + " and " + access.rightExpr);

  return true;
}
