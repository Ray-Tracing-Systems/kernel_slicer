#include "kslicer.h"
#include "class_gen.h"
#include "ast_matchers.h"
#include "initial_pass.h"

#include <sstream>
#include <algorithm>

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

kslicer::KernelInfoVisitor::KernelInfoVisitor(clang::Rewriter &R, const clang::CompilerInstance& a_compiler, 
                                              kslicer::MainClassInfo* a_codeInfo, kslicer::KernelInfo& a_kernel) : 
                                              m_rewriter(R), m_compiler(a_compiler), 
                                              m_codeInfo(a_codeInfo), m_currKernel(a_kernel)
{
  
}

bool kslicer::KernelInfoVisitor::VisitForStmt(clang::ForStmt* forLoop)
{  
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

bool kslicer::KernelInfoVisitor::VisitMemberExpr(clang::MemberExpr* expr)
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

void kslicer::KernelInfoVisitor::DetectTextureAccess(clang::CXXMemberCallExpr* call)
{
  clang::CXXMethodDecl* fDecl = call->getMethodDecl();  
  if(fDecl == nullptr)  
    return;

  //std::string debugText = GetRangeSourceCode(call->getSourceRange(), m_compiler); 
  std::string fname     = fDecl->getNameInfo().getName().getAsString();
  clang::Expr* pTexName =	call->getImplicitObjectArgument(); 
  std::string objName   = kslicer::GetRangeSourceCode(pTexName->getSourceRange(), m_compiler);     

  if(fname == "sample" || fname == "Sample")
  {
    auto samplerArg         = call->getArg(0);
    std::string samplerName = kslicer::GetRangeSourceCode(samplerArg->getSourceRange(), m_compiler); 

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

void kslicer::KernelInfoVisitor::DetectTextureAccess(clang::BinaryOperator* expr)
{
  std::string op        = kslicer::GetRangeSourceCode(clang::SourceRange(expr->getOperatorLoc()), m_compiler); 
  std::string debugText = kslicer::GetRangeSourceCode(expr->getSourceRange(), m_compiler);     
  if(expr->isAssignmentOp()) // detect a_brightPixels[coord] = color;
  {
    clang::Expr* left = expr->getLHS(); 
    if(clang::isa<clang::CXXOperatorCallExpr>(left))
    {
      clang::CXXOperatorCallExpr* leftOp = clang::dyn_cast<clang::CXXOperatorCallExpr>(left);
      std::string op2 = kslicer::GetRangeSourceCode(clang::SourceRange(leftOp->getOperatorLoc()), m_compiler);  
      if(op2 == "]" || op2 == "[" || op2 == "[]")
        ProcessReadWriteTexture(leftOp, true);
    }
  }
  else if((op == "]" || op == "[" || op == "[]") && clang::isa<clang::CXXOperatorCallExpr>(expr))
    ProcessReadWriteTexture(clang::dyn_cast<clang::CXXOperatorCallExpr>(expr), false);
}


void kslicer::KernelInfoVisitor::DetectTextureAccess(clang::CXXOperatorCallExpr* expr)
{
  std::string op        = kslicer::GetRangeSourceCode(clang::SourceRange(expr->getOperatorLoc()), m_compiler); 
  std::string debugText = kslicer::GetRangeSourceCode(expr->getSourceRange(), m_compiler);     
  if(expr->isAssignmentOp()) // detect a_brightPixels[coord] = color;
  {
    clang::Expr* left = expr->getArg(0); 
    if(clang::isa<clang::CXXOperatorCallExpr>(left))
    {
      clang::CXXOperatorCallExpr* leftOp = clang::dyn_cast<clang::CXXOperatorCallExpr>(left);
      std::string op2 = kslicer::GetRangeSourceCode(clang::SourceRange(leftOp->getOperatorLoc()), m_compiler);  
      if(op2 == "]" || op2 == "[" || op2 == "[]")
        ProcessReadWriteTexture(leftOp, true);
    }
  }
  else if(op == "]" || op == "[" || op == "[]")
    ProcessReadWriteTexture(expr, false);
}

void kslicer::KernelInfoVisitor::ProcessReadWriteTexture(clang::CXXOperatorCallExpr* expr, bool write)
{
  const auto currAccess = write ? kslicer::TEX_ACCESS::TEX_ACCESS_WRITE : kslicer::TEX_ACCESS::TEX_ACCESS_READ;
  const auto hash = kslicer::GetHashOfSourceRange(expr->getSourceRange());
  if(m_visitedTexAccessNodes.find(hash) != m_visitedTexAccessNodes.end())
    return;

  std::string objName = kslicer::GetRangeSourceCode(clang::SourceRange(expr->getExprLoc()), m_compiler);
  
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



bool kslicer::KernelInfoVisitor::VisitCXXMemberCallExpr(clang::CXXMemberCallExpr* f)
{
  DetectTextureAccess(f);
  return true; 
}

bool kslicer::KernelInfoVisitor::VisitReturnStmt(clang::ReturnStmt* ret)
{
  clang::Expr* retExpr = ret->getRetValue();
  if (!retExpr)
    return true;

  clang::Expr* pRetExpr = ret->getRetValue();
  if(!clang::isa<clang::CallExpr>(pRetExpr))
    return true;
  
  clang::CallExpr* callExpr = clang::dyn_cast<clang::CallExpr>(pRetExpr);

  // assotiate this buffer with target hierarchy 
  //
  auto retQt = pRetExpr->getType();
  if(retQt->isPointerType())
    retQt = retQt->getPointeeType();

  std::string fname = callExpr->getDirectCallee()->getNameInfo().getName().getAsString();
  if(fname != "MakeObjPtr")
    return true;

  assert(callExpr->getNumArgs() == 2);
  const clang::Expr* firstArgExpr  = callExpr->getArgs()[0];
  const clang::Expr* secondArgExpr = callExpr->getArgs()[1];
  
  std::string retTypeName = kslicer::CutOffStructClass(retQt.getAsString());
  
  // get ObjData buffer name
  //
  std::string makerObjBufferName = kslicer::GetRangeSourceCode(secondArgExpr->getSourceRange(), m_compiler);
  ReplaceFirst(makerObjBufferName, ".data()", "");
  for(auto& h : m_codeInfo->m_vhierarchy)
  {
    if(h.second.interfaceName == retTypeName)
    {
      h.second.objBufferName = makerObjBufferName;       
      break;
    }
  }

  return true;
}

bool kslicer::KernelInfoVisitor::VisitUnaryOperator(clang::UnaryOperator* expr)
{
  clang::Expr* subExpr =	expr->getSubExpr();
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
    std::string leftStr = kslicer::GetRangeSourceCode(subExpr->getSourceRange(), m_compiler);

    auto p = m_currKernel.usedMembers.find(leftStr);
    if(p != m_currKernel.usedMembers.end())
    {
      kslicer::KernelInfo::ReductionAccess access;
      access.type      = kslicer::KernelInfo::REDUCTION_TYPE::UNKNOWN;
      access.rightExpr = "";
      access.leftExpr  = leftStr;
      access.dataType  = subExpr->getType().getAsString();

      if(op == "++")
        access.type    = kslicer::KernelInfo::REDUCTION_TYPE::ADD_ONE;
      else if(op == "--")
        access.type    = kslicer::KernelInfo::REDUCTION_TYPE::SUB_ONE;

      
      m_currKernel.hasFinishPass = m_currKernel.hasFinishPass || !access.SupportAtomicLastStep(); // if atomics can not be used, we must insert additional finish pass
      m_currKernel.subjectedToReduction[leftStr] = access;
    }
  }

  return true;
}

void kslicer::KernelInfoVisitor::ProcessReductionOp(const std::string& op, const clang::Expr* lhs, const clang::Expr* rhs, const clang::Expr* expr)
{
  auto pShaderRewriter = m_codeInfo->pShaderFuncRewriter;
  std::string leftVar = kslicer::GetRangeSourceCode(lhs->getSourceRange().getBegin(), m_compiler);
  std::string leftStr = kslicer::GetRangeSourceCode(lhs->getSourceRange(), m_compiler);
  auto p = m_currKernel.usedMembers.find(leftVar);
  if(p != m_currKernel.usedMembers.end())
  {
    kslicer::KernelInfo::ReductionAccess access;
    access.type      = kslicer::KernelInfo::REDUCTION_TYPE::UNKNOWN;
    access.rightExpr = kslicer::GetRangeSourceCode(rhs->getSourceRange(), m_compiler);
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
      access.arrayIndex  = kslicer::GetRangeSourceCode(idx->getSourceRange(), m_compiler);
      access.arrayName   = kslicer::GetRangeSourceCode(name->getSourceRange(), m_compiler); 
      
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
      access.type    = kslicer::KernelInfo::REDUCTION_TYPE::ADD;
    else if(op == "*=")
      access.type    = kslicer::KernelInfo::REDUCTION_TYPE::MUL;
    else if(op == "-=")
      access.type    = kslicer::KernelInfo::REDUCTION_TYPE::SUB;
    
    //auto exprHash = kslicer::GetHashOfSourceRange(expr->getSourceRange());

    m_currKernel.hasFinishPass = m_currKernel.hasFinishPass || !access.SupportAtomicLastStep(); // if atomics can not be used, we must insert additional finish pass
    m_currKernel.subjectedToReduction[leftStr] = access;
  }
}

void kslicer::KernelInfoVisitor::DetectFuncReductionAccess(const clang::Expr* lhs, const clang::Expr* rhs, const clang::Expr* expr)
{
  if(!clang::isa<clang::MemberExpr>(lhs))
    return;

  std::string leftStr  = kslicer::GetRangeSourceCode(lhs->getSourceRange(), m_compiler);
  std::string rightStr = kslicer::GetRangeSourceCode(rhs->getSourceRange(), m_compiler);
  auto p = m_currKernel.usedMembers.find(leftStr);
  if(p == m_currKernel.usedMembers.end())
    return;

  if(clang::isa<clang::MaterializeTemporaryExpr>(rhs))
  {
    auto tmp = clang::dyn_cast<clang::MaterializeTemporaryExpr>(rhs);
    rhs = tmp->getSubExpr();
  }

  if(!clang::isa<clang::CallExpr>(rhs))
  {
    kslicer::PrintWarning("unsupported expression for reduction via assigment inside loop; must be 'a = f(a,b)'", rhs->getSourceRange(), m_compiler.getSourceManager());
    kslicer::PrintWarning("reduction for variable '" +  leftStr + "' will not be generated, sure this is ok for you?", rhs->getSourceRange(), m_compiler.getSourceManager());
    return;
  }  
  
  auto call    = clang::dyn_cast<clang::CallExpr>(rhs);
  auto numArgs = call->getNumArgs();
  if(numArgs != 2)
  {
    kslicer::PrintError("function which is used in reduction must have 2 args; a = f(a,b)'", expr->getSourceRange(), m_compiler.getSourceManager());
    return;
  }

  const clang::Expr* arg0 = call->getArg(0);
  const clang::Expr* arg1 = call->getArg(1);

  std::string arg0Str = kslicer::GetRangeSourceCode(arg0->getSourceRange(), m_compiler);
  std::string arg1Str = kslicer::GetRangeSourceCode(arg1->getSourceRange(), m_compiler);
  
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
    kslicer::PrintError("incorrect arguments of reduction function, one of them must be same as assigment result; a = f(a,b)'", expr->getSourceRange(), m_compiler.getSourceManager());
    return;
  }

  std::string callExpr = kslicer::GetRangeSourceCode(call->getSourceRange(), m_compiler);
  std::string fname    = callExpr.substr(0, callExpr.find_first_of('('));
  
  kslicer::KernelInfo::ReductionAccess access;
 
  access.type      = kslicer::KernelInfo::REDUCTION_TYPE::FUNC;
  access.funcName  = fname;
  access.rightExpr = secondArg;
  access.leftExpr  = leftStr;
  access.dataType  = lhs->getType().getAsString();

  //auto exprHash = kslicer::GetHashOfSourceRange(expr->getSourceRange());

  m_currKernel.hasFinishPass = m_currKernel.hasFinishPass || !access.SupportAtomicLastStep(); // if atomics can not be used, we must insert additional finish pass
  m_currKernel.subjectedToReduction[leftStr] = access;
}

bool kslicer::KernelInfoVisitor::VisitCompoundAssignOperator(clang::CompoundAssignOperator* expr)
{
  auto opRange = expr->getSourceRange();
  if(opRange.getEnd()   <= m_currKernel.loopInsides.getBegin() || 
     opRange.getBegin() >= m_currKernel.loopInsides.getEnd() ) // not inside loop
    return true;   

  const clang::Expr* lhs = expr->getLHS();
  const clang::Expr* rhs = expr->getRHS();
  const auto  op  = expr->getOpcodeStr();

  ProcessReductionOp(op.str(), lhs, rhs, expr);

  return true;
}

bool kslicer::KernelInfoVisitor::VisitCXXOperatorCallExpr(clang::CXXOperatorCallExpr* expr)
{
  std::string debugTxt = kslicer::GetRangeSourceCode(expr->getSourceRange(), m_compiler); 

  DetectTextureAccess(expr);
  auto opRange = expr->getSourceRange();
  if(opRange.getEnd()   <= m_currKernel.loopInsides.getBegin() || 
     opRange.getBegin() >= m_currKernel.loopInsides.getEnd() ) // not inside loop
    return true;   

  const auto numArgs = expr->getNumArgs();
  if(numArgs != 2)
    return true; 
  
  std::string op = kslicer::GetRangeSourceCode(clang::SourceRange(expr->getOperatorLoc()), m_compiler);  
  if(op == "+=" || op == "-=" || op == "*=")
  {
    const clang::Expr* lhs = expr->getArg(0);
    const clang::Expr* rhs = expr->getArg(1);

    ProcessReductionOp(op, lhs, rhs, expr);
  }
  else if (op == "=")
  {
    const clang::Expr* lhs = expr->getArg(0);
    const clang::Expr* rhs = expr->getArg(1);

    DetectFuncReductionAccess(lhs, rhs, expr);
  }

  return true;
}

bool kslicer::KernelInfoVisitor::VisitBinaryOperator(clang::BinaryOperator* expr) // detect reduction like m_var = F(m_var,expr)
{
  auto opRange = expr->getSourceRange();
  if(!m_codeInfo->IsRTV() && (opRange.getEnd() <= m_currKernel.loopInsides.getBegin() || opRange.getBegin() >= m_currKernel.loopInsides.getEnd())) // not inside loop
    return true;  

  const auto op = expr->getOpcodeStr();
  if(op != "=")
    return true;
  
  DetectTextureAccess(expr);

  const clang::Expr* lhs = expr->getLHS();
  const clang::Expr* rhs = expr->getRHS();
  
  DetectFuncReductionAccess(lhs, rhs, expr);
  
  return true;
}


bool kslicer::KernelInfoVisitor::NameNeedsFakeOffset(const std::string& a_name) const
{
   bool exclude = false;
   for(auto arg : m_currKernel.args)
   {
     if(arg.needFakeOffset && arg.name == a_name)
       exclude = true;
   }
   return exclude;
}

bool kslicer::KernelInfoVisitor::VisitCallExpr(clang::CallExpr* call)
{
  return true;
}
