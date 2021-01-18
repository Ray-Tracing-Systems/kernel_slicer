#include "kslicer.h"
#include "class_gen.h"
#include "ast_matchers.h"

#include <sstream>
#include <algorithm>

bool kslicer::MainFuncASTVisitor::VisitCXXMethodDecl(CXXMethodDecl* f) 
{
  if (f->hasBody())
  {
    // Get name of function
    const DeclarationNameInfo dni = f->getNameInfo();
    const DeclarationName dn      = dni.getName();
    const std::string fname       = dn.getAsString();

    mainFuncCmdName = fname + "Cmd";
    m_rewriter.ReplaceText(dni.getSourceRange(), mainFuncCmdName);
  }

  return true; // returning false aborts the traversal
}

std::string kslicer::MainFuncASTVisitor::MakeKernelCallCmdString(CXXMemberCallExpr* f)
{
  const DeclarationNameInfo dni = f->getMethodDecl()->getNameInfo();
  const DeclarationName dn      = dni.getName();
  const std::string fname       = dn.getAsString();

  const std::string kernName    = m_pCodeInfo->RemoveKernelPrefix(fname);

  // extract arguments to form correct descriptor set
  //
  std::stringstream strOut1;
  strOut1 << "_" << m_dsTagId;                                                           // temporary disable DS hashes due to we didn't account for
  auto args = ExtractArgumentsOfAKernelCall(f);                                          // member vectors that have to be bound also and accounted here!
  std::string callSign = MakeKernellCallSignature(args, m_mainFuncName) + strOut1.str(); //
  auto p2 = dsIdBySignature.find(callSign);
  if(p2 == dsIdBySignature.end())
  {
    dsIdBySignature[callSign] = m_kernCallTypes.size();
    p2 = dsIdBySignature.find(callSign);
    KernelCallInfo call;
    call.kernelName         = kernName;
    call.originKernelName   = fname;
    call.callerName         = m_mainFuncName;
    call.descriptorSetsInfo = args;
    m_kernCallTypes.push_back(call);
  }
  m_dsTagId++;

  std::string textOfCall = GetRangeSourceCode(f->getSourceRange(), m_compiler);
  std::string textOfArgs = textOfCall.substr( textOfCall.find("("));

  std::stringstream strOut;
  {
    // understand if we are inside the loop, or outside of it
    //
    auto pKernel = m_kernels.find(fname);
    assert(pKernel != m_kernels.end()); 

    auto callSourceRangeHash = kslicer::GetHashOfSourceRange(f->getSourceRange());
    auto p3 = m_mainFunc.CallsInsideFor.find(callSourceRangeHash);
    auto p4 = m_mainFunc.ExitExprIfCall.find(callSourceRangeHash);

    std::string flagsVariableName = "";
    if(p3 != m_mainFunc.CallsInsideFor.end())
    {
      flagsVariableName = "inForFlags";
      
      if(pKernel->second.isBoolTyped && p4 == m_mainFunc.ExitExprIfCall.end())
        flagsVariableName += "D";
      else if(p3->second.isNegative)
        flagsVariableName += "N";
    }
    else 
    {
      flagsVariableName = "outOfForFlags";
      if(pKernel->second.isBoolTyped && p4 == m_mainFunc.ExitExprIfCall.end())
        flagsVariableName += "D";
      else if(p4 != m_mainFunc.ExitExprIfCall.end() && p4->second.isNegative)
        flagsVariableName += "N";
    }

    strOut << "vkCmdBindDescriptorSets(a_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ";
    strOut << kernName.c_str() << "Layout," << " 0, 1, " << "&m_allGeneratedDS[" << p2->second << "], 0, nullptr);" << std::endl;
    if(m_pCodeInfo->NeedThreadFlags())
      strOut << "  vkCmdPushConstants(m_currCmdBuffer," << kernName.c_str() << "Layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t)*1, &" << flagsVariableName.c_str() << ");" << std::endl;
    strOut << "  " << kernName.c_str() << "Cmd" << textOfArgs.c_str();
  }
  
  return strOut.str();
}


bool kslicer::MainFuncASTVisitor::VisitCXXMemberCallExpr(CXXMemberCallExpr* f)
{
  // Get name of function
  const DeclarationNameInfo dni = f->getMethodDecl()->getNameInfo();
  const DeclarationName dn      = dni.getName();
  const std::string fname       = dn.getAsString();

  if(m_pCodeInfo->IsKernel(fname))
  {
    std::string callStr = MakeKernelCallCmdString(f);

    auto p2 = m_alreadyProcessedCalls.find(kslicer::GetHashOfSourceRange(f->getSourceRange()));
    if(p2 == m_alreadyProcessedCalls.end())
    {
      m_rewriter.ReplaceText(f->getSourceRange(), callStr); // getExprLoc
      m_kernellCallTagId++;
    }
  }

  return true; 
}

bool kslicer::MainFuncASTVisitor::VisitIfStmt(IfStmt* ifExpr)
{
  Expr* conBody = ifExpr->getCond();
  if(isa<UnaryOperator>(conBody)) // if(!kernel_XXX(...))
  {
    const auto bodyOp = dyn_cast<UnaryOperator>(conBody);
    conBody = bodyOp->getSubExpr();
  }

  if(isa<CXXMemberCallExpr>(conBody))
  {
    CXXMemberCallExpr* f = dyn_cast<CXXMemberCallExpr>(conBody); // extract kernel_XXX(...)
    
    // Get name of function
    const DeclarationNameInfo dni = f->getMethodDecl()->getNameInfo();
    const DeclarationName dn      = dni.getName();
    const std::string fname       = dn.getAsString();
  
    if(m_pCodeInfo->IsKernel(fname))
    {
      std::string callStr = MakeKernelCallCmdString(f);
      m_rewriter.ReplaceText(ifExpr->getSourceRange(), callStr);
      m_kernellCallTagId++;
      m_alreadyProcessedCalls.insert( kslicer::GetHashOfSourceRange(f->getSourceRange()) );
    }
  }

  return true;
}

std::vector<kslicer::ArgReferenceOnCall> kslicer::MainFuncASTVisitor::ExtractArgumentsOfAKernelCall(CXXMemberCallExpr* f)
{
  std::vector<kslicer::ArgReferenceOnCall> args; 
  args.reserve(20);

  auto predefinedNames = GetAllPredefinedThreadIdNamesRTV();
  
  for(size_t i=0;i<f->getNumArgs();i++)
  {
    const Expr* currArgExpr = f->getArgs()[i];
    assert(currArgExpr != nullptr);
    auto sourceRange = currArgExpr->getSourceRange();
    std::string text = GetRangeSourceCode(sourceRange, m_compiler);
  
    ArgReferenceOnCall arg;    
    if(text[0] == '&')
    {
      arg.umpersanned = true;
      text = text.substr(1);

      auto pClassVar = m_allClassMembers.find(text);
      if(pClassVar != m_allClassMembers.end())
        arg.argType = KERN_CALL_ARG_TYPE::ARG_REFERENCE_CLASS_POD;
    }
    else if(text.find(".data()") != std::string::npos) 
    {
      std::string varName = text.substr(0, text.find(".data()"));
      auto pClassVar = m_allClassMembers.find(varName);
      if(pClassVar == m_allClassMembers.end())
        std::cout << "[KernelCallError]: vector<...> variable '" << varName.c_str() << "' was not found in class!" << std::endl; 
      else
        pClassVar->second.usedInMainFn = true;

      arg.argType = KERN_CALL_ARG_TYPE::ARG_REFERENCE_CLASS_VECTOR;
    }

    auto elementId = std::find(predefinedNames.begin(), predefinedNames.end(), text); // exclude predefined names from arguments
    if(elementId != predefinedNames.end())
      arg.argType = KERN_CALL_ARG_TYPE::ARG_REFERENCE_THREAD_ID;

    arg.varName = text;
    args.push_back(arg); 
  }

  for(auto& arg : args)  // in this loop we have to define argument (actual parameter) type
  {
    if(arg.argType != KERN_CALL_ARG_TYPE::ARG_REFERENCE_UNKNOWN_TYPE)
      continue;
    
    auto p2 = m_argsOfMainFunc.find(arg.varName);
    if(p2 != m_argsOfMainFunc.end())
      arg.argType = KERN_CALL_ARG_TYPE::ARG_REFERENCE_ARG;

    auto p3 = m_mainFuncLocals.find(arg.varName);
    if(p3 != m_mainFuncLocals.end())
      arg.argType = KERN_CALL_ARG_TYPE::ARG_REFERENCE_LOCAL;
  }

  // for(auto& arg : args) // check we don't have unknown types of arguments // well, in fact now we can ) Use arguments
  // {
  //   if(arg.argType == KERN_CALL_ARG_TYPE::ARG_REFERENCE_UNKNOWN_TYPE)
  //   {
  //     auto beginLoc        = f->getSourceRange().getBegin();
  //     std::string fileName = m_sm.getFilename(beginLoc);
  //     const auto line      = m_sm.getPresumedLoc(beginLoc).getLine();
  //     std::cout << "  WARNING: expr '" << arg.varName.c_str() << "' was not classified; file: " << fileName.c_str() << ", line: " << line << std::endl; 
  //   }
  //  
  // }

  return args;
}

std::string kslicer::MakeKernellCallSignature(const std::vector<ArgReferenceOnCall>& a_args, const std::string& a_mainFuncName)
{
  std::stringstream strOut;
  for(const auto& arg : a_args)
  {
    switch(arg.argType)
    {
      case KERN_CALL_ARG_TYPE::ARG_REFERENCE_LOCAL:
      strOut << "[L]";
      break;

      case KERN_CALL_ARG_TYPE::ARG_REFERENCE_ARG:
      strOut << "[A][" << a_mainFuncName.c_str() << "]" ;
      break;

      case KERN_CALL_ARG_TYPE::ARG_REFERENCE_CLASS_VECTOR:
      strOut << "[V]";
      break;
      
      case KERN_CALL_ARG_TYPE::ARG_REFERENCE_CLASS_POD:
      strOut << "[P]";
      break;

      case KERN_CALL_ARG_TYPE::ARG_REFERENCE_THREAD_ID:
      strOut << "[T]";
      break;

      default:
      strOut << "[U]";
      break;
    };

    strOut << arg.varName.c_str();
  }

  return strOut.str();
}


bool ReplaceFirst(std::string& str, const std::string& from, const std::string& to) 
{
  size_t start_pos = str.find(from);
  if(start_pos == std::string::npos)
    return false;
  str.replace(start_pos, from.length(), to);
  return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::string kslicer::MainClassInfo::RemoveKernelPrefix(const std::string& a_funcName) const
{
  std::string name = a_funcName;
  if(ReplaceFirst(name, "kernel_", ""))
    return name;
  else
    return a_funcName;
}

bool kslicer::MainClassInfo::IsKernel(const std::string& a_funcName) const
{
  auto pos = a_funcName.find("kernel_");
  return (pos != std::string::npos);
}

std::string kslicer::MainClassInfo::GetCFSourceCodeCmd(MainFuncInfo& a_mainFunc, clang::CompilerInstance& compiler, std::vector<KernelCallInfo>& a_outDSInfo)
{
  const std::string&   a_mainClassName = this->mainClassName;
  auto&                a_outDsInfo     = this->allDescriptorSetsInfo;

  const CXXMethodDecl* a_node          = a_mainFunc.Node;
  const std::string&   a_mainFuncName  = a_mainFunc.Name;
  std::string&         a_outFuncDecl   = a_mainFunc.GeneratedDecl;

  const auto  inOutParamList = kslicer::ListPointerParamsOfMainFunc(a_node);

  Rewriter rewrite2;
  rewrite2.setSourceMgr(compiler.getSourceManager(), compiler.getLangOpts());

  a_mainFunc.startDSNumber = a_outDsInfo.size();

  kslicer::MainFuncASTVisitor rv(rewrite2, compiler, a_mainFunc, inOutParamList, this);

  rv.m_kernCallTypes = a_outDsInfo;
  rv.TraverseDecl(const_cast<clang::CXXMethodDecl*>(a_node));
  a_outDsInfo        = rv.m_kernCallTypes;

  clang::SourceLocation b(a_node->getBeginLoc()), _e(a_node->getEndLoc());
  clang::SourceLocation e(clang::Lexer::getLocForEndOfToken(_e, 0, compiler.getSourceManager(), compiler.getLangOpts()));
  
  std::string sourceCode = rewrite2.getRewrittenText(clang::SourceRange(b,e));
  
  // (1) TestClass::MainFuncCmd --> TestClass_Generated::MainFuncCmd and add input command Buffer as first argument
  // 
  const std::string replaceFrom = a_mainClassName + "::" + rv.mainFuncCmdName;
  const std::string replaceTo   = a_mainClassName + "_Generated" + "::" + rv.mainFuncCmdName;

  assert(ReplaceFirst(sourceCode, replaceFrom, replaceTo));

  if(a_mainFunc.Node->getNumParams() != 0)
  {
    assert(ReplaceFirst(sourceCode, "(", "(VkCommandBuffer a_commandBuffer, "));
  }
  else
  {
    assert(ReplaceFirst(sourceCode, "(", "(VkCommandBuffer a_commandBuffer"));
  }

  // (3) set m_currCmdBuffer with input command bufer and add other prolog to MainFunCmd
  //
  std::stringstream strOut;
  strOut << "{" << std::endl;
  strOut << "  m_currCmdBuffer = a_commandBuffer;" << std::endl;

  if(this->NeedThreadFlags())
  {
    strOut << "  const uint32_t outOfForFlags  = KGEN_FLAG_RETURN;" << std::endl;
    strOut << "  const uint32_t inForFlags     = KGEN_FLAG_RETURN | KGEN_FLAG_BREAK;" << std::endl;
    if(a_mainFunc.needToAddThreadFlags)
    {
      strOut << "  const uint32_t outOfForFlagsN = KGEN_FLAG_RETURN | KGEN_FLAG_SET_EXIT_NEGATIVE;" << std::endl;
      strOut << "  const uint32_t inForFlagsN    = KGEN_FLAG_RETURN | KGEN_FLAG_BREAK | KGEN_FLAG_SET_EXIT_NEGATIVE;" << std::endl;
      strOut << "  const uint32_t outOfForFlagsD = KGEN_FLAG_RETURN | KGEN_FLAG_DONT_SET_EXIT;" << std::endl;
      strOut << "  const uint32_t inForFlagsD    = KGEN_FLAG_RETURN | KGEN_FLAG_BREAK | KGEN_FLAG_DONT_SET_EXIT;" << std::endl;
      strOut << "  vkCmdFillBuffer(a_commandBuffer, " << a_mainFunc.Name.c_str() << "_local.threadFlagsBuffer , 0, VK_WHOLE_SIZE, 0); // zero thread flags, mark all threads to be active" << std::endl;
      strOut << "  VkMemoryBarrier fillBarrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT }; " << std::endl;
      strOut << "  vkCmdPipelineBarrier(a_commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_DEPENDENCY_BY_REGION_BIT, 1, &fillBarrier, 0, nullptr, 0, nullptr); " << std::endl;
    }
    strOut << std::endl; 
  }

  size_t bracePos = sourceCode.find("{");
  sourceCode = (sourceCode.substr(0, bracePos) + strOut.str() + sourceCode.substr(bracePos+2));

  return sourceCode;
}

std::string kslicer::MainClassInfo::GetCFDeclFromSource(const std::string& sourceCode)
{
  std::string mainFuncDecl = sourceCode.substr(0, sourceCode.find(")")+1) + ";";
  assert(ReplaceFirst(mainFuncDecl, mainClassName + "_Generated" + "::", ""));
  return "virtual " + mainFuncDecl;
}


uint32_t kslicer::RTV_Pattern::GetKernelDim(const kslicer::KernelInfo& a_kernel) const
{
  return uint32_t(GetKernelTIDArgs(a_kernel).size());
} 

std::string kslicer::RTV_Pattern::VisitAndRewrite_CF(MainFuncInfo& a_mainFunc, clang::CompilerInstance& compiler)
{
  const std::string&   a_mainClassName = this->mainClassName;
  auto&                a_outDsInfo     = this->allDescriptorSetsInfo;

  const CXXMethodDecl* a_node          = a_mainFunc.Node;
  const std::string&   a_mainFuncName  = a_mainFunc.Name;
  std::string&         a_outFuncDecl   = a_mainFunc.GeneratedDecl;

  std::string sourceCode = GetCFSourceCodeCmd(a_mainFunc, compiler, a_outDsInfo);
  a_outFuncDecl          = GetCFDeclFromSource(sourceCode); 
  a_mainFunc.endDSNumber = a_outDsInfo.size();

  return sourceCode;
}


void kslicer::RTV_Pattern::AddSpecVars_CF(std::vector<MainFuncInfo>& a_mainFuncList, std::vector<KernelInfo>& a_kernelList)
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
  KernelInfo::Arg    tFlagsArg;

  tFlagsLocalVar.name = "threadFlags";
  tFlagsLocalVar.type = "uint";
  tFlagsLocalVar.sizeInBytes = sizeof(uint32_t);
  
  tFlagsArg.name           = "kgen_threadFlags";
  tFlagsArg.needFakeOffset = true;
  tFlagsArg.size           = 1; // array size 
  tFlagsArg.type           = "uint*";
  
  // list kernels
  //
  std::unordered_map<std::string, size_t> kernelIdByName;
  for(size_t i=0;i<a_kernelList.size();i++)
    kernelIdByName[a_kernelList[i].name] = i;
  
  // Add threadFlags to kernel arguments
  //
  for(auto kName : kernelsAddedFlags)
  {
    assert(kernelIdByName.find(kName) != kernelIdByName.end());
    auto& kernel = a_kernelList[kernelIdByName[kName]];
    
    size_t foundId = size_t(-1);
    for(size_t i=0;i<kernel.args.size();i++)
    {
      if(kernel.args[i].name == tFlagsArg.name)
      {
        foundId = i;
        break;
      }
    }

    if(foundId == size_t(-1))
    {
      kernel.args.push_back(tFlagsArg);
      kernel.checkThreadFlags = true;
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

void kslicer::RTV_Pattern::PlugSpecVarsInCalls_CF(const std::vector<MainFuncInfo>&   a_mainFuncList, 
                                                  const std::vector<KernelInfo>&     a_kernelList,
                                                  std::vector<KernelCallInfo>&       a_kernelCalls)
{
  // list kernels and main functions
  //
  std::unordered_map<std::string, size_t> kernelIdByName;
  std::unordered_map<std::string, size_t> mainFuncIdByName;
  for(size_t i=0;i<a_kernelList.size();i++)
    kernelIdByName[a_kernelList[i].name] = i;
  for(size_t i=0;i<a_mainFuncList.size();i++)
    mainFuncIdByName[a_mainFuncList[i].Name] = i;

  ArgReferenceOnCall tFlagsArgRef;
  tFlagsArgRef.argType     = KERN_CALL_ARG_TYPE::ARG_REFERENCE_LOCAL;
  tFlagsArgRef.varName     = "threadFlags";
  tFlagsArgRef.umpersanned = true;

  // add thread flags to MainFuncions and all kernel calls for each MainFunc
  //
  for(auto& call : a_kernelCalls)
  {
    const auto& mainFunc = a_mainFuncList[mainFuncIdByName[call.callerName]];
    if(!mainFunc.needToAddThreadFlags)
      continue;

    auto p2 = kernelIdByName.find(call.originKernelName);
    if(p2 != kernelIdByName.end())
      call.descriptorSetsInfo.push_back(tFlagsArgRef);
  }
  
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

kslicer::RTV_Pattern::MList kslicer::RTV_Pattern::ListMatchers_CF(const std::string& mainFuncName)
{
  std::vector<clang::ast_matchers::StatementMatcher> list;
  list.push_back(kslicer::MakeMatch_LocalVarOfMethod(mainFuncName));
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
  const auto& call = a_call;
  
  // find kernel:
  //
  size_t found = size_t(-1); 
  for(size_t i=0; i<kernels.size(); i++)
  {
    if(kernels[i].name == call.originKernelName)
    {
      found = i;
      break;
    }
  }

  if(found != size_t(-1)) 
  {
    auto& actualParameters = call.descriptorSetsInfo;
    for(size_t argId = 0; argId<actualParameters.size(); argId++)
    {
      if(actualParameters[argId].argType == kslicer::KERN_CALL_ARG_TYPE::ARG_REFERENCE_LOCAL)
        kernels[found].args[argId].needFakeOffset = true; 
    }
  }
}

void kslicer::RTV_Pattern::ProcessKernelArg(KernelInfo::Arg& arg, const KernelInfo& a_kernel) const 
{
  auto pdef = GetAllPredefinedThreadIdNamesRTV();
  auto id   = std::find(pdef.begin(), pdef.end(), arg.name);
  arg.isThreadID = (id != pdef.end()); 
}

std::vector<kslicer::InOutVarInfo> kslicer::ListPointerParamsOfMainFunc(const CXXMethodDecl* a_node)
{
  std::vector<InOutVarInfo> params;
  for(int i=0;i<a_node->getNumParams();i++)
  {
    const ParmVarDecl* currParam = a_node->getParamDecl(i);
    
    const clang::QualType qt = currParam->getType();
    const auto typePtr = qt.getTypePtr(); 
    assert(typePtr != nullptr);
    
    if(!typePtr->isPointerType())
      continue;
    
    InOutVarInfo var;
    var.name = currParam->getNameAsString();
    params.push_back(var);
  }

  return params;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool kslicer::KernelReplacerASTVisitor::VisitMemberExpr(MemberExpr* expr)
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

  // (1) get type of variable itself because we need to cast pointer to this type
  //
  QualType qt = pFieldDecl->getTypeSourceInfo()->getType();
  std::string fieldType = qt.getAsString();
  kslicer::ReplaceOpenCLBuiltInTypes(fieldType);

  // (2) get variable offset in buffer by its name 
  //
  const std::string fieldName = pFieldDecl->getNameAsString(); 
  const auto pMember = m_variables.find(fieldName);
  if(pMember == m_variables.end())
    return true;

  // (3) put *(pointer+offset) instead of variable name, leave containers as they are
  // read only large data structures because small can be readn once in the neggining of kernel
  //
  if(!pMember->second.isContainer && pMember->second.sizeInBytes > kslicer::READ_BEFORE_USE_THRESHOLD) 
  {
    const std::string buffName = kslicer::GetProjPrefix() + "data"; 
    std::stringstream strOut;
    strOut << "*(  "; 
    strOut << "(__global const " << fieldType.c_str() << "*)" << "(" << buffName.c_str() << "+" << (pMember->second.offsetInTargetBuffer/sizeof(uint32_t)) << ")";
    strOut << "  )";
    
    m_rewriter.ReplaceText(expr->getSourceRange(), strOut.str());
  }
  
  return true;
}

bool kslicer::KernelReplacerASTVisitor::VisitCXXMemberCallExpr(CXXMemberCallExpr* f)
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

  if(fname == "size" || fname == "capacity" && isVector)
  {
    const std::string exprContent = GetRangeSourceCode(f->getSourceRange(), m_compiler);
    const auto posOfPoint = exprContent.find(".");
    const std::string memberName = exprContent.substr(0, posOfPoint);
    m_rewriter.ReplaceText(f->getSourceRange(), memberName + "_" + fname);
  }
 
  return true;
}

bool kslicer::KernelReplacerASTVisitor::VisitReturnStmt(ReturnStmt* ret)
{
  Expr* retExpr = ret->getRetValue();
  if (!retExpr || !m_needModifyExitCond)
    return true;

  std::string retExprText = GetRangeSourceCode(retExpr->getSourceRange(), m_compiler);
  std::stringstream strOut;
  strOut << "{" << std::endl;
  strOut << "    const bool exitHappened = (kgen_tFlagsMask & KGEN_FLAG_SET_EXIT_NEGATIVE) != 0 ? !(" <<  retExprText.c_str() << ") : (" << retExprText.c_str() << ");" << std::endl;
  strOut << "    if((kgen_tFlagsMask & KGEN_FLAG_DONT_SET_EXIT) == 0 && exitHappened)" << std::endl;
  strOut << "    {" << std::endl;
  strOut << "      kgen_threadFlags[" << m_fakeOffsetExp.c_str() << "] = ((kgen_tFlagsMask & KGEN_FLAG_BREAK) != 0) ? KGEN_FLAG_BREAK : KGEN_FLAG_RETURN;" << std::endl;
  strOut << "    }" << std::endl;
  strOut << "  }";

  m_rewriter.ReplaceText(ret->getSourceRange(), strOut.str());
  return true;
}


bool kslicer::KernelReplacerASTVisitor::CheckIfExprHasArgumentThatNeedFakeOffset(const std::string& exprStr)
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

bool kslicer::KernelReplacerASTVisitor::VisitUnaryOperator(UnaryOperator* expr)
{
  // detect " *(something)"

  if(expr->canOverflow() || expr->isArithmeticOp()) // -UnaryOperator ...'LiteMath::uint':'unsigned int' lvalue prefix '*' cannot overflow
    return true;
 
  std::string exprAll = GetRangeSourceCode(expr->getSourceRange(), m_compiler);

  if(exprAll.find("*") != 0)
    return true;

  Expr* subExpr =	expr->getSubExpr();
  if(subExpr == nullptr)
    return true;

  std::string exprInside = GetRangeSourceCode(subExpr->getSourceRange(), m_compiler);

  // check if this argument actually need fake Offset
  //
  const bool needOffset = CheckIfExprHasArgumentThatNeedFakeOffset(exprInside);

  if(needOffset)
    m_rewriter.ReplaceText(expr->getSourceRange(), exprInside + "[" + m_fakeOffsetExp + "]");

  return true;
}

//// tid, fakeOffset(tidX,tidY,kgen_iNumElementsX) or fakeOffset2(tidX,tidY,tidX,kgen_iNumElementsX, kgen_iNumElementsY)
//
std::string GetFakeOffsetExpression(const kslicer::KernelInfo& a_funcInfo, const std::vector<kslicer::MainClassInfo::ArgTypeAndNamePair>& threadIds) 
{
  if(threadIds.size() == 1)
    return threadIds[0].argName;
  else if(threadIds.size() == 2)
    return std::string("fakeOffset(") + threadIds[0].argName + "," + threadIds[1].argName + ",kgen_iNumElementsX)";
  else if(threadIds.size() == 3)
    return std::string("fakeOffset(") + threadIds[0].argName + "," + threadIds[1].argName + "," + threadIds[2].argName + ",kgen_iNumElementsX,kgen_iNumElementsY)";
  else
    return "tid";
}

std::string kslicer::MainClassInfo::VisitAndRewrite_KF(KernelInfo& a_funcInfo, const clang::CompilerInstance& compiler)
{
  const CXXMethodDecl* a_node = a_funcInfo.astNode;
  //a_node->dump();

  std::string fakeOffsetExpr = GetFakeOffsetExpression(a_funcInfo, GetKernelTIDArgs(a_funcInfo));

  Rewriter rewrite2;
  rewrite2.setSourceMgr(compiler.getSourceManager(), compiler.getLangOpts());

  kslicer::KernelReplacerASTVisitor rv(rewrite2, compiler, this->mainClassName, this->dataMembers, a_funcInfo.args, fakeOffsetExpr, a_funcInfo.isBoolTyped);
  rv.TraverseDecl(const_cast<clang::CXXMethodDecl*>(a_node));
  
  clang::SourceLocation b(a_node->getBeginLoc()), _e(a_node->getEndLoc());
  clang::SourceLocation e(clang::Lexer::getLocForEndOfToken(_e, 0, compiler.getSourceManager(), compiler.getLangOpts()));
  
  return rewrite2.getRewrittenText(clang::SourceRange(b,e));
}

std::vector<kslicer::MainClassInfo::ArgTypeAndNamePair> kslicer::MainClassInfo::GetKernelTIDArgs(const KernelInfo& a_kernel) const
{
  std::vector<kslicer::MainClassInfo::ArgTypeAndNamePair> args;
  for (const auto& arg : a_kernel.args) 
  {    
    if(arg.isThreadID)
    { 
      std::string typeStr = arg.type;
      kslicer::ReplaceOpenCLBuiltInTypes(typeStr);

      ArgTypeAndNamePair arg2;
      arg2.argName  = arg.name;
      arg2.sizeName = arg.name;
      arg2.typeName = typeStr;
      arg2.id       = 0;
      args.push_back(arg2);
    }
  }

  std::sort(args.begin(), args.end(), [](const auto& a, const auto & b) { return a.argName < b.argName; });

  return args;
}

std::vector<kslicer::MainClassInfo::ArgTypeAndNamePair> kslicer::MainClassInfo::GetKernelCommonArgs(const KernelInfo& a_kernel) const
{
  std::vector<kslicer::MainClassInfo::ArgTypeAndNamePair> args;
  for (const auto& arg : a_kernel.args) 
  {    
    if(!arg.isThreadID && !arg.isLoopSize && !arg.IsUser())
    { 
      std::string typeStr = arg.type;
      kslicer::ReplaceOpenCLBuiltInTypes(typeStr);
      ReplaceFirst(typeStr, this->mainClassName + "::", "");

      ArgTypeAndNamePair arg2;
      arg2.argName  = arg.name;
      arg2.typeName = typeStr;
      args.push_back(arg2);
    }
  }

  return args;
}

void kslicer::ObtainKernelsDecl(std::vector<kslicer::KernelInfo>& a_kernelsData, const clang::CompilerInstance& compiler, const std::string& a_mainClassName, const MainClassInfo& a_codeInfo)
{
  for (auto& k : a_kernelsData)  
  {
    assert(k.astNode != nullptr);
    auto sourceRange = k.astNode->getSourceRange();
    std::string kernelSourceCode = GetRangeSourceCode(sourceRange, compiler);
    
    std::string kernelCmdDecl = kernelSourceCode.substr(0, kernelSourceCode.find(")")+1);
    assert(ReplaceFirst(kernelCmdDecl, a_mainClassName + "::", ""));
    
    kernelCmdDecl = a_codeInfo.RemoveKernelPrefix(kernelCmdDecl);

    assert(ReplaceFirst(kernelCmdDecl,"(", "Cmd("));
    if(k.isBoolTyped)
      ReplaceFirst(kernelCmdDecl,"bool ", "void ");
    k.DeclCmd = kernelCmdDecl;
  }
}
