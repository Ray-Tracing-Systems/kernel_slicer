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

  // (5) join usedMemberFunctions
  //
  {
    res.usedMemberFunctions = cf.usedMemberFunctions;
    for(const auto& k : a_kernels)
      for(const auto& member : k->usedMemberFunctions)
        res.usedMemberFunctions.insert(member);
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
