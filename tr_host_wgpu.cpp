#include "kslicer.h"
#include "template_rendering.h"
#include "class_gen.h"

void kslicer::WGPUCodeGen::GenerateHost(std::string fullSuffix, nlohmann::json jsonHost, kslicer::MainClassInfo& a_mainClass, const kslicer::TextGenSettings& a_settings)
{
  kslicer::ApplyJsonToTemplate("templates_wk/wk_class.h",   fullSuffix + ".h",   jsonHost);
  kslicer::ApplyJsonToTemplate("templates_wk/wk_class.cpp", fullSuffix + ".cpp", jsonHost);
}

void kslicer::WGPUCodeGen::GenerateHostDevFeatures(std::string fullSuffix, nlohmann::json jsonHost, kslicer::MainClassInfo& a_mainClass, const kslicer::TextGenSettings& a_settings)
{
  //kslicer::ApplyJsonToTemplate("templates_wk/wk_class.cpp", fullSuffix + ".cpp", jsonHost);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::string kslicer::GetControlFuncDeclWGPU(const clang::FunctionDecl* fDecl, clang::CompilerInstance& compiler)
{
  std::string text = fDecl->getNameInfo().getName().getAsString() + "Cmd(WGPUCommandEncoder a_commandEncoder";
  if(fDecl->getNumParams()!= 0)
    text += ", ";
  for(unsigned i=0;i<fDecl->getNumParams();i++)
  {
    auto pParam = fDecl->getParamDecl(i);
    text += kslicer::GetRangeSourceCode(pParam->getSourceRange(), compiler);
    if(i!=fDecl->getNumParams()-1)
      text += ", ";
  }
  return text + ")";
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void kslicer::MainFunctionRewriterWGPU::ReplaceTextOrWorkAround(clang::SourceRange a_range, const std::string& a_text)
{
  if(a_range.getBegin().getRawEncoding() == a_range.getEnd().getRawEncoding())
    m_workAround[GetHashOfSourceRange(a_range)] = a_text;
  else
    m_rewriter.ReplaceText(a_range, a_text);
}

bool kslicer::MainFunctionRewriterWGPU::WasNotRewrittenYet(const clang::Stmt* expr) const
{
  if(expr == nullptr)
    return true;
  if(clang::isa<clang::NullStmt>(expr))
    return true;
  const auto exprHash = kslicer::GetHashOfSourceRange(expr->getSourceRange());
  return (m_pRewrittenNodes->find(exprHash) == m_pRewrittenNodes->end());
}

void kslicer::MainFunctionRewriterWGPU::MarkRewritten(const clang::Stmt* expr) { kslicer::MarkRewrittenRecursive(expr, *m_pRewrittenNodes); }

std::string kslicer::MainFunctionRewriterWGPU::RecursiveRewrite(const clang::Stmt* expr)
{
  if(expr == nullptr)
    return "";
  
  MainFunctionRewriterWGPU rvCopy = *this;
  rvCopy.TraverseStmt(const_cast<clang::Stmt*>(expr));
  
  auto range = expr->getSourceRange();
  auto p = rvCopy.m_workAround.find(GetHashOfSourceRange(range));
  if(p != rvCopy.m_workAround.end())
    return p->second;
  else
  {
    std::string text = m_rewriter.getRewrittenText(range);
    return (text != "") ? text : kslicer::GetRangeSourceCode(range, m_compiler);
  }
}

bool kslicer::MainFunctionRewriterWGPU::VisitCXXMethodDecl(CXXMethodDecl* f)
{
  if (f->hasBody())
  {
    // Get name of function
    const DeclarationNameInfo dni = f->getNameInfo();
    const DeclarationName dn      = dni.getName();
    const std::string fname       = dn.getAsString();

    const auto exprHash = kslicer::GetHashOfSourceRange(dni.getSourceRange());
    if(m_pRewrittenNodes->find(exprHash) == m_pRewrittenNodes->end())
    {
      mainFuncCmdName = fname + "Cmd";
      ReplaceTextOrWorkAround(dni.getSourceRange(), mainFuncCmdName);
      m_pRewrittenNodes->insert(exprHash);
    }
  }

  return true; // returning false aborts the traversal
}

struct NameFlagsPair
{
  std::string         name;
  kslicer::TEX_ACCESS flags;
  uint32_t            argId = 0;
  bool                isArg = false;
};

std::string kslicer::MainFunctionRewriterWGPU::MakeKernelCallCmdString(CXXMemberCallExpr* f)
{
  const DeclarationNameInfo dni = f->getMethodDecl()->getNameInfo();
  const DeclarationName dn      = dni.getName();
  const std::string fname       = dn.getAsString();

  const auto pKernelInfo = m_kernels.find(fname);
  assert(pKernelInfo != m_kernels.end());

  const std::string kernName = m_pCodeInfo->RemoveKernelPrefix(fname);

  // extract arguments to form correct descriptor set
  //
  const auto args     = ExtractArgumentsOfAKernelCall(f, m_mainFunc.ExcludeList);
  const auto callSign = MakeKernellCallSignature(m_mainFuncName, args, pKernelInfo->second.usedContainers); // + strOut1.str();
  auto p2 = dsIdBySignature.find(callSign);
  if(p2 == dsIdBySignature.end())
  {
    dsIdBySignature[callSign] = allDescriptorSetsInfo.size();
    p2 = dsIdBySignature.find(callSign);
    KernelCallInfo call;
    call.kernelName         = kernName;
    call.originKernelName   = fname;
    call.callerName         = m_mainFuncName;
    call.descriptorSetsInfo = args;
    allDescriptorSetsInfo.push_back(call);
  }

  //std::string textOfCall = GetRangeSourceCode(f->getSourceRange(), m_compiler);
  //std::string textOfArgs = textOfCall.substr( textOfCall.find("("));

  std::stringstream argsOut;
  argsOut << "(";
  for(size_t i=0;i<f->getNumArgs();i++)
  {
    const Expr* currArgExpr = f->getArgs()[i];
    std::string textDebug = GetRangeSourceCode(currArgExpr->getSourceRange(), m_compiler);
    std::string text = RecursiveRewrite(currArgExpr);
    argsOut << text.c_str();
    if(i < f->getNumArgs()-1)
      argsOut << ", ";
  }
  argsOut << ")";

  std::string textOfArgs = argsOut.str();

  std::stringstream strOut;
  {
    // understand if we are inside the loop, or outside of it
    //
    auto pKernel = m_kernels.find(fname);
    assert(pKernel != m_kernels.end());

    auto callSourceRangeHash = kslicer::GetHashOfSourceRange(f->getSourceRange());
    auto p3 = m_mainFunc.CallsInsideFor.find(callSourceRangeHash);
    auto p4 = m_mainFunc.ExitExprIfCall.find(callSourceRangeHash);

    auto accesedTextures = kslicer::ListAccessedTextures(args, pKernelInfo->second);

    if(pKernel->second.isIndirect)
      strOut << kernName.c_str() << "_UpdateIndirect();" << std::endl << "  ";
    
    strOut << "m_currPCOffset = m_pushConstantStride*" << p2->second << ";" << std::endl;
    strOut << "  wgpuComputePassEncoderSetBindGroup(m_currPassCS, 0, m_allGeneratedDS[" << p2->second << "]" << ", 0, nullptr);" << std::endl;
    strOut << "  " << kernName.c_str() << "Cmd" << textOfArgs.c_str();
  }

  return strOut.str();
}

std::string kslicer::MainFunctionRewriterWGPU::MakeServiceKernelCallCmdString(CallExpr* call, const std::string& a_name)
{
  std::string kernName = "copyKernelFloat"; // extract from 'call' exact name of service function;
  auto originArgs = ExtractArgumentsOfAKernelCall(call, m_mainFunc.ExcludeList);
  const std::string memBarCode = "vkCmdPipelineBarrier(m_currCmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memoryBarrier, 0, nullptr, 0, nullptr)";

  if(a_name == "memcpy")
  {
    kernName = "copyKernelFloat";
    std::vector<ArgReferenceOnCall> args(2); // extract corretc arguments from memcpy (CallExpr* call)
    {
      args[0].argType = originArgs[0].argType;
      args[0].name    = originArgs[0].name;
      args[0].kind    = DATA_KIND::KIND_POINTER;

      args[1].argType = originArgs[1].argType;
      args[1].name    = originArgs[1].name;
      args[1].kind    = DATA_KIND::KIND_POINTER;
    }

    const auto callSign = MakeKernellCallSignature(m_mainFuncName, args, std::map<std::string, kslicer::UsedContainerInfo>()); // + strOut1.str();
    auto p2 = dsIdBySignature.find(callSign);
    if(p2 == dsIdBySignature.end())
    {
      dsIdBySignature[callSign] = allDescriptorSetsInfo.size();
      p2 = dsIdBySignature.find(callSign);
      KernelCallInfo call;
      call.kernelName         = kernName;
      call.originKernelName   = kernName;
      call.callerName         = m_mainFuncName;
      call.descriptorSetsInfo = args;
      call.isService          = true;
      allDescriptorSetsInfo.push_back(call);
    }

    std::stringstream strOut;
    strOut << "vkCmdBindDescriptorSets(a_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ";
    strOut << kernName.c_str() << "Layout," << " 0, 1, " << "&m_allGeneratedDS[" << p2->second << "], 0, nullptr);" << std::endl;
    strOut << "  " << kernName.c_str() << "Cmd(" << originArgs[2].name << " / sizeof(float));" << std::endl;
    strOut << "  " << memBarCode.c_str();
    return strOut.str();
  }
  else if(a_name == "exclusive_scan" || a_name == "inclusive_scan")
  {
    std::string commandName  = "ExclusiveScan";
    std::string dsLayoutName = "";
    if(a_name == "exclusive_scan")
      commandName = "ExclusiveScan";
    else
      commandName = "InclusiveScan";
    kernName = "m_scan.internal";

    std::string launchSize = kslicer::ExtractSizeFromArgExpression(originArgs[1].name);
    std::vector<ArgReferenceOnCall> args(3);
    {
      args[0].argType = originArgs[0].argType;
      args[0].name    = kslicer::ClearNameFromBegin(originArgs[0].name);
      args[0].kind    = DATA_KIND::KIND_POINTER;

      //std::cout << "  originArgs[1].name = " << originArgs[1].name.c_str() << std::endl;

      args[1].argType = originArgs[2].argType;
      args[1].name    = kslicer::ClearNameFromBegin(originArgs[2].name);
      args[1].kind    = DATA_KIND::KIND_POINTER;

      args[2].argType = kslicer::KERN_CALL_ARG_TYPE::ARG_REFERENCE_SERVICE_DATA;
      args[2].name    = "m_scan.m_scanTempData";
      args[2].kind    = DATA_KIND::KIND_VECTOR;
    }

    kslicer::ServiceCall call;
    call.opName       = "scan";
    const std::string dataType = kslicer::SubstrBetween(originArgs[0].type, "<", ">");
    call.dataTypeName = m_pCodeInfo->pShaderFuncRewriter->RewriteStdVectorTypeStr(dataType);
    ReplaceFirst(call.dataTypeName, "const ", "");

    call.lambdaSource = "+";
    m_pCodeInfo->serviceCalls[call.key()] = call;

    commandName  = "m_scan_" + call.dataTypeName + "." + commandName;
    kernName     = "m_scan_" + call.dataTypeName + ".internal";
    dsLayoutName = "m_scan_" + call.dataTypeName;
    args[2].name = "m_scan_" + call.dataTypeName + ".m_scanTempData";

    const auto callSign = MakeKernellCallSignature(m_mainFuncName, args, std::map<std::string, kslicer::UsedContainerInfo>()); // + strOut1.str();
    auto p2 = dsIdBySignature.find(callSign);
    if(p2 == dsIdBySignature.end())
    {
      dsIdBySignature[callSign] = allDescriptorSetsInfo.size();
      p2 = dsIdBySignature.find(callSign);
      KernelCallInfo call;
      call.kernelName         = kernName;
      call.originKernelName   = kernName;
      call.callerName         = m_mainFuncName;
      call.descriptorSetsInfo = args;
      call.isService          = true;
      allDescriptorSetsInfo.push_back(call);
    }

    std::stringstream strOut;
    strOut << "vkCmdBindDescriptorSets(a_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, " << dsLayoutName.c_str() << ".scanFwdLayout," << " 0, 1, " << "&m_allGeneratedDS[" << p2->second << "], 0, nullptr);" << std::endl;
    strOut << "  " << commandName.c_str() << "Cmd(m_currCmdBuffer, " << launchSize.c_str() << ");" << std::endl;
    strOut << "  " << memBarCode.c_str();
    return strOut.str();
  }
  else if(a_name == "sort")
  {
    std::string commandName  = "m_sort.BitonicSort";
    std::string dsLayoutName = "";
    kernName = "m_sort.sort";

    std::string launchSize = kslicer::ExtractSizeFromArgExpression(originArgs[1].name);
    std::vector<ArgReferenceOnCall> args(1);
    {
      args[0].argType = originArgs[0].argType;
      args[0].name    = kslicer::ClearNameFromBegin(originArgs[0].name);
      args[0].kind    = DATA_KIND::KIND_POINTER;
    }

    kslicer::ServiceCall call;
    call.opName       = a_name;
    call.dataTypeName = m_pCodeInfo->pShaderFuncRewriter->RewriteStdVectorTypeStr(kslicer::SubstrBetween(originArgs[0].type, "<", ">"));
    call.lambdaSource = kslicer::FixLamdbaSourceCode(m_pCodeInfo->pShaderFuncRewriter->RecursiveRewrite(originArgs[2].node));
    call.lambdaSource = call.lambdaSource.substr(call.lambdaSource.find("[]")+2); // eliminate "[]" from source code
    m_pCodeInfo->serviceCalls[call.key()] = call;

    commandName  = "m_sort_" + call.dataTypeName + ".BitonicSort";
    kernName     = "m_sort_" + call.dataTypeName + ".sort";
    dsLayoutName = "m_sort_" + call.dataTypeName;

    const auto callSign = MakeKernellCallSignature(m_mainFuncName, args, std::map<std::string, kslicer::UsedContainerInfo>()); // + strOut1.str();
    auto p2 = dsIdBySignature.find(callSign);
    if(p2 == dsIdBySignature.end())
    {
      dsIdBySignature[callSign] = allDescriptorSetsInfo.size();
      p2 = dsIdBySignature.find(callSign);
      KernelCallInfo call;
      call.kernelName         = kernName;
      call.originKernelName   = kernName;
      call.callerName         = m_mainFuncName;
      call.descriptorSetsInfo = args;
      call.isService          = true;
      allDescriptorSetsInfo.push_back(call);
    }

    std::stringstream strOut;
    strOut << "vkCmdBindDescriptorSets(a_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, " << dsLayoutName.c_str() << ".bitonicPassLayout," << " 0, 1, " << "&m_allGeneratedDS[" << p2->second << "], 0, nullptr);" << std::endl;
    strOut << "  " << commandName.c_str() << "Cmd(m_currCmdBuffer, " << launchSize.c_str() << ", m_devProps.limits.maxComputeWorkGroupSize[0]);" << std::endl;
    strOut << "  " << memBarCode.c_str();
    return strOut.str();
  }
  else if (a_name == "MatMulTranspose")
  {
    kernName = "matMulTranspose";
    std::vector<ArgReferenceOnCall> args(3); // extract corretc arguments from memcpy (CallExpr* call)
    for (int i = 0; i < 3; ++i)
    {
      args[i].argType = originArgs[i * 2].argType;
      args[i].name    = originArgs[i * 2].name;
      args[i].kind    = DATA_KIND::KIND_POINTER;
    }

    const auto callSign = kslicer::MakeKernellCallSignature(m_mainFuncName, args, std::map<std::string, kslicer::UsedContainerInfo>()); // + strOut1.str();
    auto p2 = dsIdBySignature.find(callSign);
    if(p2 == dsIdBySignature.end())
    {
      dsIdBySignature[callSign] = allDescriptorSetsInfo.size();
      p2 = dsIdBySignature.find(callSign);
      KernelCallInfo call;
      call.kernelName         = kernName;
      call.originKernelName   = kernName;
      call.callerName         = m_mainFuncName;
      call.descriptorSetsInfo = args;
      call.isService          = true;
      allDescriptorSetsInfo.push_back(call);
    }

    std::stringstream strOut;
    strOut << "vkCmdBindDescriptorSets(a_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ";
    strOut << kernName.c_str() << "Layout," << " 0, 1, " << "&m_allGeneratedDS[" << p2->second << "], 0, nullptr);" << std::endl;
    strOut << "  " << kernName.c_str() << "Cmd(";
    for (int i = 0; i < 3; ++i)
    {
      strOut << originArgs[i * 2 + 1].name << ", ";
    }
    for (int i = 6; i < 8; ++i)
    {
      strOut << originArgs[i].name << ", ";
    }
    strOut << originArgs[8].name << ");" << std::endl;
    strOut << "  " << memBarCode.c_str();
    return strOut.str();
  }

  return "";
}

bool kslicer::MainFunctionRewriterWGPU::VisitCXXMemberCallExpr(CXXMemberCallExpr* f)
{
  // Get name of function
  const DeclarationNameInfo dni = f->getMethodDecl()->getNameInfo();
  const DeclarationName dn      = dni.getName();
  const std::string fname       = dn.getAsString();

  if(m_pCodeInfo->IsKernel(fname) && WasNotRewrittenYet(f))
  {
    auto pKernel = m_kernels.find(fname);
    if(pKernel != m_kernels.end())
    {
      //std::string debugText = kslicer::GetRangeSourceCode(f->getSourceRange(), m_compiler);
      if(pKernel->second.be.enabled)
        kslicer::ExtractBlockSizeFromCall(f, pKernel->second, m_compiler);
      std::string callStr = MakeKernelCallCmdString(f);
      ReplaceTextOrWorkAround(f->getSourceRange(), callStr); // getExprLoc
      MarkRewritten(f);
    }
    else
    {
      //std::string callStr = MakeKernelCallCmdString(f);
      std::cout << "  [MainFunctionRewriterWGPU::VisitCXXMemberCallExpr]: can't process kernel call for " << fname.c_str() << std::endl;
    }
  }

  return true;
}

bool kslicer::MainFunctionRewriterWGPU::VisitCallExpr(CallExpr* call)
{
  //if(isa<CXXMemberCallExpr>(call)) // because we process them in "VisitCXXMemberCallExpr"
  //  return true;

  const FunctionDecl* fDecl = call->getDirectCallee();
  if(fDecl == nullptr)             // definitely can't process nullpointer
    return true;

  // Get name of function
  //
  const DeclarationNameInfo dni = fDecl->getNameInfo();
  const DeclarationName dn      = dni.getName();
  const std::string fname       = dn.getAsString();

  if((fname == "memcpy" || fname == "exclusive_scan" || fname == "inclusive_scan" || fname == "sort" || fname == "MatMulTranspose") && WasNotRewrittenYet(call))
  {
    m_pCodeInfo->usedServiceCalls.insert(fname);
    std::string testStr = MakeServiceKernelCallCmdString(call, fname);
    ReplaceTextOrWorkAround(call->getSourceRange(), testStr);
    MarkRewritten(call);
  }

  //std::cout << "  [CF::Vulkan]:" << m_mainFunc.Name << " --> " << fname << std::endl;
  if(m_pCodeInfo->persistentRTV && !m_mainFunc.usePersistentThreads && (fname == "RTVPersistent_Iters" || fname == "RTVPersistent_SetIter"))
  {
    std::cout << "    --> Enable Persistent Threads for '" << m_mainFunc.Name << "'" << std::endl;
    m_mainFunc.usePersistentThreads = true;
  }

  return true;
}

bool kslicer::MainFunctionRewriterWGPU::VisitMemberExpr(MemberExpr* expr)
{
  if(!WasNotRewrittenYet(expr))
    return true;

  std::string setter, containerName;
  if(CheckSettersAccess(expr, m_pCodeInfo, m_compiler, &setter, &containerName) && WasNotRewrittenYet(expr))
  {
    std::string name = setter + "Vulkan." + containerName;
    ReplaceTextOrWorkAround(expr->getSourceRange(), name);
    MarkRewritten(expr);
  }
  return true;
}

std::vector<kslicer::ArgReferenceOnCall> kslicer::MainFunctionRewriterWGPU::ExtractArgumentsOfAKernelCall(CallExpr* f, const std::unordered_set<std::string>& a_excludeList)
{
  std::vector<kslicer::ArgReferenceOnCall> args;
  args.reserve(20);

  auto predefinedNames = GetAllPredefinedThreadIdNamesRTV();

  for(size_t i=0;i<f->getNumArgs();i++)
  {
    const Expr* currArgExpr = f->getArgs()[i];
    const clang::QualType q = currArgExpr->getType().getCanonicalType();
    std::string text        = GetRangeSourceCode(currArgExpr->getSourceRange(), m_compiler);

    // check if this is conbst variable which is declared inside control func
    //
    bool isConstFound = false;
    bool isLiteral    = false;
    for(size_t i=0; i<m_pCodeInfo->mainFunc.size();i++)
    {
      if(m_pCodeInfo->mainFunc[i].Name == m_mainFuncName)
      {
        auto& localVars = m_pCodeInfo->mainFunc[i].LocalConst;
        auto p          = localVars.find(text);
        if(p != localVars.end())
          isConstFound = p->second.isConst;
        break;
      }
    }

    auto checkExpr = kslicer::RemoveImplicitCast(currArgExpr);
    if(clang::isa<clang::IntegerLiteral>(checkExpr) || clang::isa<clang::FloatingLiteral>(checkExpr) || clang::isa<clang::CXXBoolLiteralExpr>(checkExpr) || clang::isa<clang::CompoundLiteralExpr>(checkExpr))
    {
      isConstFound = true;
      isLiteral    = true;
    }

    ArgReferenceOnCall arg;
    arg.type          = q.getAsString();
    arg.isConst       = q.isConstQualified() || isConstFound;
    arg.isExcludedRTV = (a_excludeList.find(text) != a_excludeList.end());
    arg.node          = checkExpr;
    if(text[0] == '&')
    {
      text = text.substr(1);
      auto pClassVar = m_allClassMembers.find(text);
      if(pClassVar != m_allClassMembers.end())
      {
        arg.argType = KERN_CALL_ARG_TYPE::ARG_REFERENCE_CLASS_POD;
        arg.kind    = DATA_KIND::KIND_POD;
      }
    }
    else if(q->isPointerType() && text.find(".data()") != std::string::npos) // TODO: add check for reference, fo the case if we want to pass vectors by reference
    {
      std::string varName = text.substr(0, text.find(".data()"));
      auto pClassVar = m_allClassMembers.find(varName);
      if(pClassVar == m_allClassMembers.end())
        std::cout << "[KernelCallError]: vector<...> variable '" << varName.c_str() << "' was not found in class!" << std::endl;
      else
        pClassVar->second.usedInMainFn = true;

      arg.argType = KERN_CALL_ARG_TYPE::ARG_REFERENCE_CLASS_VECTOR;
      arg.kind    = DATA_KIND::KIND_VECTOR;
    }
    else if(kslicer::IsTexture(q))
    {
      auto pClassVar = m_allClassMembers.find(text);
      if(pClassVar != m_allClassMembers.end()) // if not found, probably this is an argument of control function. Not an error. Process in later.
        pClassVar->second.usedInMainFn = true;
      arg.kind      = DATA_KIND::KIND_TEXTURE;
    }
    else if(isConstFound || isLiteral)
    {
      arg.argType = KERN_CALL_ARG_TYPE::ARG_REFERENCE_CONST_OR_LITERAL;
      arg.kind    = DATA_KIND::KIND_POD;
    }

    auto elementId = std::find(predefinedNames.begin(), predefinedNames.end(), text); // exclude predefined names from arguments
    if(elementId != predefinedNames.end())
    {
      arg.argType = KERN_CALL_ARG_TYPE::ARG_REFERENCE_THREAD_ID;
      arg.kind    = DATA_KIND::KIND_POD;
    }

    arg.name = text;
    args.push_back(arg);
  }

  for(auto& arg : args)  // in this loop we have to define argument (actual parameter) type
  {
    if(arg.argType != KERN_CALL_ARG_TYPE::ARG_REFERENCE_UNKNOWN_TYPE)
      continue;

    auto p2 = m_argsOfMainFunc.find(arg.name);
    if(p2 != m_argsOfMainFunc.end())
      arg.argType = KERN_CALL_ARG_TYPE::ARG_REFERENCE_ARG;

    auto p3 = m_mainFuncLocals.find(arg.name);
    if(p3 != m_mainFuncLocals.end())
      arg.argType = KERN_CALL_ARG_TYPE::ARG_REFERENCE_LOCAL;
  }

  return args;
}
