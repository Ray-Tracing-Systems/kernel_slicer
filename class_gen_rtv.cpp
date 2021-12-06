#include "kslicer.h"
#include "class_gen.h"
#include "ast_matchers.h"
#include "extractor.h"

#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/Parse/ParseAST.h"

#include <sstream>
#include <algorithm>

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//// tid, fakeOffset(tidX,tidY,kgen_iNumElementsX) or fakeOffset2(tidX,tidY,tidX,kgen_iNumElementsX, kgen_iNumElementsY)
//
std::string kslicer::GetFakeOffsetExpression(const kslicer::KernelInfo& a_funcInfo, const std::vector<kslicer::ArgFinal>& threadIds, const std::string names[3]) 
{ 
  if(threadIds.size() == 1)
    return threadIds[0].name;
  else if(threadIds.size() == 2)
    return std::string("fakeOffset(") + threadIds[0].name + "," + threadIds[1].name + "," + names[0] + ")";
  else if(threadIds.size() == 3)
    return std::string("fakeOffset2(") + threadIds[0].name + "," + threadIds[1].name + "," + threadIds[2].name + "," + names[0] + "," + names[1] + ")";
  else
    return "tid";
}

std::vector<std::string> kslicer::GetAllPredefinedThreadIdNamesRTV()
{
  return {"tid", "tidX", "tidY", "tidZ"};
}

uint32_t kslicer::RTV_Pattern::GetKernelDim(const kslicer::KernelInfo& a_kernel) const
{
  return uint32_t(GetKernelTIDArgs(a_kernel).size());
} 

void kslicer::RTV_Pattern::VisitAndRewrite_CF(MainFuncInfo& a_mainFunc, clang::CompilerInstance& compiler)
{
  //const std::string&   a_mainClassName = this->mainClassName;
  //const CXXMethodDecl* a_node          = a_mainFunc.Node;
  //const std::string&   a_mainFuncName  = a_mainFunc.Name;
  //std::string&         a_outFuncDecl   = a_mainFunc.GeneratedDecl;
  GetCFSourceCodeCmd(a_mainFunc, compiler, this->megakernelRTV); // ==> write this->allDescriptorSetsInfo, a_mainFunc
  a_mainFunc.endDSNumber   = allDescriptorSetsInfo.size();
  a_mainFunc.InOuts        = kslicer::ListParamsOfMainFunc(a_mainFunc.Node, compiler);
}


void kslicer::RTV_Pattern::AddSpecVars_CF(std::vector<MainFuncInfo>& a_mainFuncList, std::unordered_map<std::string, KernelInfo>& a_kernelList)
{
  // (1) first scan all main functions, if no one needed just exit
  //
  std::unordered_set<std::string> kernelsToAddFlags;
  std::unordered_set<std::string> kernelsAddedFlags;

  for(auto& mainFunc : a_mainFuncList)
  {
    if(mainFunc.ExitExprIfCond.size() != 0)
    {
      for(const auto& kernelName : mainFunc.UsedKernels)
        kernelsToAddFlags.insert(kernelName);
      mainFunc.needToAddThreadFlags = true;
    }
  }

  if(kernelsToAddFlags.empty())
    return;

  // (2) add flags to kernels which we didn't process yet
  //
  while(!kernelsToAddFlags.empty())
  {
    // (2.1) process kernels
    //
    for(auto kernName : kernelsToAddFlags)
      kernelsAddedFlags.insert(kernName);

    // (2.2) find all main functions which used processed kernels 
    //
    for(auto& mainFunc : a_mainFuncList)
    {
      for(const auto& kernelName : kernelsAddedFlags)
      {
        if(mainFunc.UsedKernels.find(kernelName) != mainFunc.UsedKernels.end())
        {
          mainFunc.needToAddThreadFlags = true;
          break;
        }
      }
    }

    // (2.3) process main functions again, check we processed all kernels for each main function which 'needToAddThreadFlags' 
    //
    kernelsToAddFlags.clear();
    for(const auto& mainFunc : a_mainFuncList)
    {
      if(!mainFunc.needToAddThreadFlags)
        continue;

      for(const auto& kName : mainFunc.UsedKernels)
      {
        const auto p = kernelsAddedFlags.find(kName);
        if(p == kernelsAddedFlags.end())
          kernelsToAddFlags.insert(kName);
      }
    }
  }

  // (3) finally add actual variables to MainFunc, arguments to kernels and reference to kernel call 
  //
  DataLocalVarInfo   tFlagsLocalVar;
  KernelInfo::ArgInfo    tFlagsArg;

  tFlagsLocalVar.name = "threadFlags";
  tFlagsLocalVar.type = "uint";
  tFlagsLocalVar.sizeInBytes = sizeof(uint32_t);
  
  tFlagsArg.name           = "kgen_threadFlags";
  tFlagsArg.needFakeOffset = true;
  tFlagsArg.isThreadFlags  = true;
  tFlagsArg.size           = 1; // array size 
  tFlagsArg.type           = "uint*";
  
  // Add threadFlags to kernel arguments
  //
  for(auto kName : kernelsAddedFlags)
  {
    auto pKernel = a_kernelList.find(kName);
    if(pKernel == a_kernelList.end())
      continue;

    size_t foundId = size_t(-1);
    for(size_t i=0;i<pKernel->second.args.size();i++)
    {
      if(pKernel->second.args[i].name == tFlagsArg.name)
      {
        foundId = i;
        break;
      }
    }

    if(foundId == size_t(-1))
    {
      pKernel->second.args.push_back(tFlagsArg);
      pKernel->second.checkThreadFlags = true;
    }
  }  
 
  // add thread flags to MainFuncions and all kernel calls for each MainFunc
  //
  for(auto& mainFunc : a_mainFuncList)
  {
    auto p = mainFunc.Locals.find(tFlagsLocalVar.name);
    if(p != mainFunc.Locals.end() || !mainFunc.needToAddThreadFlags)
      continue;

    mainFunc.Locals[tFlagsLocalVar.name] = tFlagsLocalVar;
  }

}

void kslicer::RTV_Pattern::PlugSpecVarsInCalls_CF(const std::vector<MainFuncInfo>&                   a_mainFuncList, 
                                                  const std::unordered_map<std::string, KernelInfo>& a_kernelList,
                                                  std::vector<KernelCallInfo>&                       a_kernelCalls)
{
  // list kernels and main functions
  //
  std::unordered_map<std::string, size_t> mainFuncIdByName;
  for(size_t i=0;i<a_mainFuncList.size();i++)
    mainFuncIdByName[a_mainFuncList[i].Name] = i;

  ArgReferenceOnCall tFlagsArgRef;
  tFlagsArgRef.argType = KERN_CALL_ARG_TYPE::ARG_REFERENCE_LOCAL;
  tFlagsArgRef.name    = "threadFlags";
  tFlagsArgRef.kind    = DATA_KIND::KIND_POD;
  tFlagsArgRef.umpersanned = true;

  // add thread flags to MainFuncions and all kernel calls for each MainFunc
  //
  for(auto& call : a_kernelCalls)
  {
    const auto& mainFunc = a_mainFuncList[mainFuncIdByName[call.callerName]];
    if(!mainFunc.needToAddThreadFlags)
      continue;

    auto p2 = a_kernelList.find(call.originKernelName);
    if(p2 != a_kernelList.end())
      call.descriptorSetsInfo.push_back(tFlagsArgRef);
  }
  
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

kslicer::RTV_Pattern::MList kslicer::RTV_Pattern::ListMatchers_CF(const std::string& mainFuncName)
{
  std::vector<clang::ast_matchers::StatementMatcher> list;
  list.push_back(kslicer::MakeMatch_LocalVarOfMethod(mainFuncName));
  list.push_back(kslicer::MakeMatch_MemberVarOfMethod(mainFuncName));
  list.push_back(kslicer::MakeMatch_MethodCallFromMethod(mainFuncName));
  list.push_back(kslicer::MakeMatch_SingleForLoopInsideFunction(mainFuncName));
  list.push_back(kslicer::MakeMatch_IfInsideForLoopInsideFunction(mainFuncName));
  list.push_back(kslicer::MakeMatch_FunctionCallInsideForLoopInsideFunction(mainFuncName));
  list.push_back(kslicer::MakeMatch_IfReturnFromFunction(mainFuncName));
  return list;
}

kslicer::RTV_Pattern::MHandlerCFPtr kslicer::RTV_Pattern::MatcherHandler_CF(kslicer::MainFuncInfo& a_mainFuncRef, const clang::CompilerInstance& a_compiler)
{
  return std::move(std::make_unique<kslicer::MainFuncAnalyzerRT>(std::cout, *this, a_compiler.getASTContext(), a_mainFuncRef));
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

kslicer::RTV_Pattern::MList kslicer::RTV_Pattern::ListMatchers_KF(const std::string& a_kernelName)
{
  std::vector<clang::ast_matchers::StatementMatcher> list;
  list.push_back(kslicer::MakeMatch_MemberVarOfMethod(a_kernelName));
  list.push_back(kslicer::MakeMatch_FunctionCallFromFunction(a_kernelName));
  return list;
}

kslicer::RTV_Pattern::MHandlerKFPtr kslicer::RTV_Pattern::MatcherHandler_KF(KernelInfo& kernel, const clang::CompilerInstance& a_compiler)
{
  return std::move(std::make_unique<kslicer::UsedCodeFilter>(std::cout, *this, &kernel, a_compiler));
}


void kslicer::RTV_Pattern::ProcessCallArs_KF(const KernelCallInfo& a_call)
{
  // (1) call from base class
  //
  MainClassInfo::ProcessCallArs_KF(a_call); 

  // (2) add ray tracing specific
  //
  auto pFoundKernel = kernels.find(a_call.originKernelName);
  if(pFoundKernel != kernels.end()) 
  {
    auto& actualParameters = a_call.descriptorSetsInfo;
    for(size_t argId = 0; argId<actualParameters.size(); argId++)
    {
      if(actualParameters[argId].argType == kslicer::KERN_CALL_ARG_TYPE::ARG_REFERENCE_LOCAL)
        pFoundKernel->second.args[argId].needFakeOffset = true; 
    }
  }
}

void kslicer::RTV_Pattern::VisitAndPrepare_KF(KernelInfo& a_funcInfo, const clang::CompilerInstance& compiler)
{
  //a_funcInfo.astNode->dump();
  Rewriter rewrite2;
  rewrite2.setSourceMgr(compiler.getSourceManager(), compiler.getLangOpts());
  
  auto pVisitor = pShaderCC->MakeKernRewriter(rewrite2, compiler, this, a_funcInfo, "", true);
  pVisitor->TraverseDecl(const_cast<clang::CXXMethodDecl*>(a_funcInfo.astNode));
}


void kslicer::RTV_Pattern::ProcessKernelArg(KernelInfo::ArgInfo& arg, const KernelInfo& a_kernel) const 
{
  auto pdef = GetAllPredefinedThreadIdNamesRTV();
  auto id   = std::find(pdef.begin(), pdef.end(), arg.name);
  arg.isThreadID = (id != pdef.end()); 
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
\brief processing of C++ member function for virtual kernels

  1) C++ class --> C style struct; this --> self; 
  2) *payload => payload[tid]; payload->member => payload[tid].member (TBD)
  3) this->vector[...]                                                (TBD)

*/
class MemberRewriter : public kslicer::FunctionRewriter
{
public:
  
  MemberRewriter(clang::Rewriter &R, const clang::CompilerInstance& a_compiler, kslicer::MainClassInfo* a_codeInfo, kslicer::MainClassInfo::DImplClass& dImpl) : 
                FunctionRewriter(R, a_compiler, a_codeInfo), 
                m_processed(dImpl.memberFunctions), m_fields(dImpl.fields), m_className(dImpl.name), m_mainClassName(a_codeInfo->mainClassName)
  { 
    
  }

  bool VisitMemberExpr_Impl(clang::MemberExpr* expr) override
  {
    clang::ValueDecl* pValueDecl = expr->getMemberDecl();
    if(!clang::isa<clang::FieldDecl>(pValueDecl))
      return true;

    clang::FieldDecl* pFieldDecl   = clang::dyn_cast<clang::FieldDecl>(pValueDecl);
    clang::RecordDecl* pRecodDecl  = pFieldDecl->getParent();
    const std::string thisTypeName = pRecodDecl->getNameAsString();

    if(thisTypeName != m_className) // ignore other than this-> expr
      return true;
    
    if(WasNotRewrittenYet(expr))
    {
      const clang::Expr* baseExpr = expr->getBase(); 
      std::string exprContent     = RecursiveRewrite(baseExpr);
      m_rewriter.ReplaceText(expr->getSourceRange(), "self->" + exprContent);
      MarkRewritten(expr);
    }

    return true;
  }

  bool VisitUnaryOperator_Impl(clang::UnaryOperator* expr) override 
  { 
    const auto op = expr->getOpcodeStr(expr->getOpcode());
    if(expr->canOverflow() || op != "*") // -UnaryOperator ... lvalue prefix '*' cannot overflow
      return true;

    clang::Expr* subExpr = expr->getSubExpr();
    if(subExpr == nullptr)
      return true;
    
    std::string exprInside = kslicer::GetRangeSourceCode(subExpr->getSourceRange(), m_compiler);  ; //RecursiveRewrite(subExpr);
    if(m_fakeOffsArgs.find(exprInside) == m_fakeOffsArgs.end())
      return true;

    if(WasNotRewrittenYet(subExpr))
    {
      if(m_codeInfo->megakernelRTV)
        m_rewriter.ReplaceText(expr->getSourceRange(), exprInside);
      else
        m_rewriter.ReplaceText(expr->getSourceRange(), exprInside + "[" + m_fakeOffsetExp + "]");
    }

    return true; 
  }
  
  /// \return whether \p Ty points to a const type, or is a const reference.
  //
  static bool isPointerToConst(clang::QualType Ty) 
  {
    return !Ty->getPointeeType().isNull() && Ty->getPointeeType().getCanonicalType().isConstQualified();
  }

  std::string RewriteMemberDecl(clang::CXXMethodDecl* fDecl, const std::string& classTypeName)
  {
    std::string fname  = fDecl->getNameInfo().getName().getAsString();
    std::string result = m_codeInfo->pShaderFuncRewriter->RewriteStdVectorTypeStr(fDecl->getReturnType().getAsString()) + " " + classTypeName + "_" + fname + "(\n  __global const " + classTypeName + "* self";
    if(fDecl->getNumParams() != 0)
      result += ", \n  ";

    bool isKernel = m_codeInfo->IsKernel(fname); 

    for(uint32_t i=0; i < fDecl->getNumParams(); i++)
    {
      const clang::ParmVarDecl* pParam  = fDecl->getParamDecl(i);
      const clang::QualType typeOfParam =	pParam->getType();

      if(typeOfParam.getAsString().find(m_mainClassName) != std::string::npos)
      {
        if(isPointerToConst(typeOfParam))
          result += "__global const struct ";
        else
          result += "__global struct ";
        
        result += m_mainClassName + "_UBO_Data* " + pParam->getNameAsString();
      }
      else
      {
        if(typeOfParam->isPointerType() && isKernel)
          result += "__global ";
        result += kslicer::GetRangeSourceCode(pParam->getSourceRange(), m_compiler); 
      }

      if(i!=fDecl->getNumParams()-1)
        result += ", \n  ";
    }

    return result + ")\n  ";
  }

  std::unordered_set<std::string> ListFakeOffArgsForKernelNamed(const std::string& fname)
  {
    std::unordered_set<std::string> fakeOffsArgs;
    if(m_codeInfo->IsKernel(fname))     
    {
      auto p = m_codeInfo->kernels.find(fname);
      if(p != m_codeInfo->kernels.end())
      {
         for(const auto& arg : p->second.args)
         {
           if(arg.needFakeOffset)
             fakeOffsArgs.insert(arg.name);

           if(arg.isThreadID)
             m_fakeOffsetExp = arg.name; // TODO: if we have 2D thread id this is more complex a bit ... 
         }
      }
    }
    return fakeOffsArgs;
  }

  bool VisitCXXMethodDecl_Impl(clang::CXXMethodDecl* fDecl) override
  {
    if(isCopy)
      return true;

    std::string fname = fDecl->getNameInfo().getName().getAsString();
    auto thisType     = fDecl->getThisType();
    auto qtOfClass    = thisType->getPointeeType(); 
    std::string classTypeName = kslicer::CutOffStructClass(qtOfClass.getAsString());

    if(classTypeName.find(fname) != std::string::npos || classTypeName.find(fname.substr(1)) != std::string::npos || fname == "GetTag" || fname == "GetSizeOf")
      return true; // exclude constructor, destructor and special functions
    
    if(WasNotRewrittenYet(fDecl->getBody()))
    { 
      if(m_codeInfo->IsKernel(fname))                          // enable fakeOffset rewrite
        m_fakeOffsArgs = ListFakeOffArgsForKernelNamed(fname); //
      std::string declSource = RewriteMemberDecl(fDecl, classTypeName);
      std::string bodySource = RecursiveRewrite(fDecl->getBody());
      m_fakeOffsArgs.clear();                                  // disable fakeOffset rewrite

      kslicer::MainClassInfo::DImplFunc funcData;
      funcData.decl          = fDecl;
      funcData.name          = fname;
      funcData.srcRewritten  = declSource + bodySource;
      funcData.isEmpty       = false;
      funcData.isConstMember = fDecl->isConst();
      //funcData.mainClassPass = mainClassDataPass;

      if(clang::isa<clang::CompoundStmt>(fDecl->getBody()))
      {
        clang::CompoundStmt* pBody = clang::dyn_cast<clang::CompoundStmt>(fDecl->getBody());
        funcData.isEmpty = pBody->body_empty();
      }
     
      m_processed.push_back(funcData);
      MarkRewritten(fDecl->getBody());
    }

    return true;
  }

  bool VisitCXXMemberCallExpr_Impl(clang::CXXMemberCallExpr* call) override
  {
    const clang::FunctionDecl* fDecl = call->getDirectCallee();  
    const std::string fname          = fDecl->getNameInfo().getName().getAsString();
    const clang::QualType qt         = call->getObjectType();
    std::string classTypeName        = kslicer::CutOffStructClass(qt.getAsString());

    if(WasNotRewrittenYet(call))
    { 
      std::string textRes = classTypeName + "_" + fname;
      textRes += "(self";
      if(call->getNumArgs() > 0)
        textRes += ",";
      for(unsigned i=0;i<call->getNumArgs();i++)
      {
        textRes += RecursiveRewrite(call->getArg(i));
        if(i < call->getNumArgs()-1)
          textRes += ",";
      }
      textRes += ")";
      
      m_rewriter.ReplaceText(call->getSourceRange(), textRes);
      MarkRewritten(call);
    }

    return true;
  }

  bool VisitFieldDecl_Impl(clang::FieldDecl* pFieldDecl) override 
  { 
    clang::RecordDecl* pRecodDecl  = pFieldDecl->getParent();
    const std::string thisTypeName = pRecodDecl->getNameAsString();
    if(thisTypeName == m_className)
      m_fields.push_back(kslicer::GetRangeSourceCode(pFieldDecl->getSourceRange(), m_compiler));
    return true; 
  } 

private:
    
  std::vector<kslicer::MainClassInfo::DImplFunc>& m_processed;
  std::vector<std::string>&                       m_fields;
  const std::string&                              m_className;
  const std::string&                              m_mainClassName;

  bool isCopy = false;
  std::unordered_set<std::string> m_fakeOffsArgs;
  std::string                     m_fakeOffsetExp;
  ///////////////////////////////////////////////////////////////////////////////////////////////////
  
  //std::unordered_set<uint64_t>  m_rewrittenNodes;
  inline void MarkRewritten(const clang::Stmt* expr) { FunctionRewriter::MarkRewritten(expr); }

  inline bool WasNotRewrittenYet(const clang::Stmt* expr) { return FunctionRewriter::WasNotRewrittenYet(expr); }

  std::string RecursiveRewrite(const clang::Stmt* expr) override
  {
    if(expr == nullptr)
      return "";
    MemberRewriter rvCopy = *this;
    rvCopy.isCopy = true;
    rvCopy.TraverseStmt(const_cast<clang::Stmt*>(expr));
    return m_rewriter.getRewrittenText(expr->getSourceRange());
  }
  
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void kslicer::RTV_Pattern::AddDispatchingHierarchy(const std::string& a_className, const std::string& a_makerName)
{
  std::cout << "   found class hierarchy: " << a_className.c_str() << " from " << a_makerName.c_str() << std::endl;

  DHierarchy hdata;
  hdata.interfaceName = kslicer::CutOffStructClass(a_className);
  hdata.makerName     = a_makerName;
  hdata.dispatchType  = this->defaultVkernelType;
  hdata.implementations.clear();
  m_vhierarchy[hdata.interfaceName] = hdata;
  allKernels[a_makerName].interfaceName = hdata.interfaceName;
} 

void kslicer::RTV_Pattern::AddDispatchingKernel(const std::string& a_className, const std::string& a_kernelName)
{
  std::cout << "   found virtual kernel dispatch: " << a_className.c_str() << "::" << a_kernelName.c_str() << std::endl;
  m_vkernelPairs.push_back(std::pair(kslicer::CutOffStructClass(a_className), a_kernelName));
} 

void kslicer::RTV_Pattern::ProcessDispatchHierarchies(const std::vector<const clang::CXXRecordDecl*>& a_decls, const clang::CompilerInstance& a_compiler)
{
  //
  //
  for(auto& p : m_vhierarchy)
  {
    const clang::CXXRecordDecl* pBaseClass = nullptr;
    std::string className = kslicer::CutOffStructClass(p.first);
    
    // find target base class
    //
    for(const auto& decl : a_decls)
    {
      const std::string testName = decl->getNameAsString();
      if(testName == className)
      {
        pBaseClass = decl;
        break;
      }
      //std::cout << "  found class: " << testName.c_str() << std::endl;
    }

    p.second.interfaceDecl = pBaseClass;

    // find all derived classes for target base class
    //
    
    clang::Rewriter rewrite2;
    rewrite2.setSourceMgr(a_compiler.getSourceManager(), a_compiler.getLangOpts());

    for(const auto& decl : a_decls)
    {
      if(decl->isDerivedFrom(pBaseClass))
      {
        DImplClass dImpl;
        dImpl.decl = decl;
        dImpl.name = decl->getNameAsString();
        // extract all member functions from class that should be rewritten
        //
        MemberRewriter rv(rewrite2, a_compiler, this, dImpl); 
        rv.TraverseDecl(const_cast<clang::CXXRecordDecl*>(dImpl.decl));                                  
        
        dImpl.isEmpty = true;
        for(auto member : dImpl.memberFunctions)
        {
          if(!member.isEmpty)
          {
            dImpl.isEmpty = false;
            break;
          }
        }

        for(auto& k : kernels)
        {
          if(k.second.className == dImpl.name)
            k.second.interfaceName = className;
        }

        p.second.implementations.push_back(dImpl);
      }
    }
  }
  
  // debug output
  //
  for(const auto& p : m_vhierarchy)
  {
    for(const auto& impl : p.second.implementations)
      std::cout << "  found " << p.first.c_str() << " --> " << impl.name.c_str() << std::endl;
  }

}


class TagSeeker : public clang::RecursiveASTVisitor<TagSeeker>
{
public:
  
  TagSeeker(const clang::CompilerInstance& a_compiler, std::vector<kslicer::DeclInClass>& a_constants, std::unordered_map<std::string, std::string>& a_tagByName) : 
            m_compiler(a_compiler), m_sm(a_compiler.getSourceManager()), m_knownConstants(a_constants), m_tagByClassName(a_tagByName) { m_tagByClassName.clear(); }

  bool VisitCXXMethodDecl(const clang::CXXMethodDecl* f)
  { 
    if(!f->hasBody())
      return true;

    // Get name of function
    const std::string fname = f->getNameInfo().getName().getAsString();
    if(fname == "GetTag" || fname == "GetTypeId")
    {
      const clang::QualType qThisType = f->getThisType();   
      const clang::QualType classType = qThisType->getPointeeType();
      const std::string thisTypeName  = kslicer::CutOffStructClass(classType.getAsString());
      
      auto funcBody = f->getBody();
      if(clang::isa<clang::CompoundStmt>(funcBody))
      {
        clang::CompoundStmt* s2 = clang::dyn_cast<clang::CompoundStmt>(funcBody);
        for(auto iter = s2->body_begin(); iter != s2->body_end(); ++iter)
        {
          if(clang::isa<clang::ReturnStmt>(*iter))
          {
            funcBody = *iter;
            break;
          }
        }
      }
      
      if(!clang::isa<clang::ReturnStmt>(funcBody))
      {
        std::cout << "  [TagSeeker::Error]: " << "Can't find returt statement in 'GetTag/GetTypeId' fuction body for '" <<  thisTypeName.c_str() <<  "' class." << std::endl;
        return true;
      }

      clang::ReturnStmt* retStmt = clang::dyn_cast< clang::ReturnStmt>(funcBody);
      clang::Expr* retVal        = retStmt->getRetValue(); 
      const std::string tagName  = kslicer::GetRangeSourceCode(retVal->getSourceRange(), m_compiler);
      
      for(const auto& decl : m_knownConstants)
      {
        //auto pos = funcBody.find(decl.name);
        if(decl.name == tagName)
        {
          m_tagByClassName[thisTypeName] = decl.name;
          break;
        }
      }
    }
    
    return true;
  }

private:
  
  const clang::CompilerInstance& m_compiler;
  const clang::SourceManager&    m_sm;
  std::vector<kslicer::DeclInClass>& m_knownConstants;
  std::unordered_map<std::string, std::string>& m_tagByClassName; 
};


void kslicer::RTV_Pattern::ExtractHierarchiesConstants(const clang::CompilerInstance& compiler, clang::tooling::ClangTool& Tool)
{
  for(auto& p : m_vhierarchy)
  {
    //// (1) get all constants inside interface
    //
    std::cout << "  process " << p.second.interfaceName.c_str() << std::endl;
    p.second.usedDecls = kslicer::ExtractTCFromClass(p.second.interfaceName, p.second.interfaceDecl, compiler, Tool);

    //// (2) juxtapose constant TAG and class implementation by analyzing GetTag() function
    //
    TagSeeker visitor(compiler, p.second.usedDecls, p.second.tagByClassName);
    for(auto impl : p.second.implementations)
      visitor.TraverseDecl(const_cast<clang::CXXRecordDecl*>(impl.decl));
  }

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

kslicer::KernelInfo kslicer::joinToMegaKernel(const std::vector<const KernelInfo*>& a_kernels, const MainFuncInfo& cf)
{
  KernelInfo res;
  if(a_kernels.size() == 0)
    return res;
  
  // (0) basic kernel info
  //
  res.name      = cf.Name + "Mega";
  res.className = a_kernels[0]->className;
  res.astNode   = cf.Node; 
  
  // (1) Add CF arguments as megakernel arguments
  //
  res.args.resize(0);
  for(const auto& var : cf.InOuts)
  {
    KernelInfo::ArgInfo argInfo = kslicer::ProcessParameter(var.paramNode);
    argInfo.name = var.name;
    argInfo.type = var.type;
    argInfo.kind = var.kind;
    argInfo.isThreadID = var.isThreadId;
    res.args.push_back(argInfo);
  }
  
  // (2) join all used members, containers and e.t.c.
  //
  for(size_t i=0;i<a_kernels.size();i++)
  {
    for(const auto& kv : a_kernels[i]->usedMembers)
      res.usedMembers.insert(kv);
    for(const auto& kv : a_kernels[i]->usedContainers)
      res.usedContainers.insert(kv);
    for(const auto& f : a_kernels[i]->shittyFunctions)
      res.shittyFunctions.push_back(f);

    for(const auto& x : a_kernels[i]->texAccessInArgs)
      res.texAccessInArgs.insert(x);
    for(const auto& x : a_kernels[i]->texAccessInMemb)
      res.texAccessInMemb.insert(x);
    for(const auto& x : a_kernels[i]->texAccessSampler)
      res.texAccessSampler.insert(x);
  }

  // (3) join shader features
  //
  for(size_t i=0;i<a_kernels.size();i++)
    res.shaderFeatures = res.shaderFeatures || a_kernels[i]->shaderFeatures;
  
  // (4) add used members by CF itself
  //
  {
    for(auto member : cf.usedMembers)
      res.usedMembers.insert(member);
    for(auto cont : cf.usedContainers)
      res.usedContainers.insert(cont);
  }

  res.isMega = true;
  return res;
}

std::string kslicer::GetCFMegaKernelCall(const MainFuncInfo& a_mainFunc)
{
  const clang::CXXMethodDecl* node = a_mainFunc.Node;
  
  std::string fName = node->getNameInfo().getName().getAsString() + "MegaCmd(";
  for(unsigned i=0;i<node->getNumParams();i++)
  {
    auto pParam = node->getParamDecl(i);
    fName += pParam->getNameAsString();

    if(i == node->getNumParams()-1)
      fName += ")";
    else
      fName += ", ";
  }

  if(node->getNumParams() == 0)
    fName += "()";

  return fName + ";";
}
