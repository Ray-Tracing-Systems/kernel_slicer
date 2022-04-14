#include "kslicer.h"
#include "class_gen.h"
#include "ast_matchers.h"

#include <sstream>
#include <algorithm>

bool kslicer::MainFunctionRewriter::VisitCXXMethodDecl(CXXMethodDecl* f) 
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
      m_rewriter.ReplaceText(dni.getSourceRange(), mainFuncCmdName);
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

std::vector<NameFlagsPair> ListAccessedTextures(const std::vector<kslicer::ArgReferenceOnCall>& args, const kslicer::KernelInfo& kernel)
{
  std::vector<NameFlagsPair> accesedTextures;
  accesedTextures.reserve(16);
  for(uint32_t i=0;i<uint32_t(args.size());i++)
  {
    if(args[i].isTexture())
    {
      std::string argNameInKernel = kernel.args[i].name;
      auto pFlags = kernel.texAccessInArgs.find(argNameInKernel);
      NameFlagsPair tex;
      tex.name  = args[i].name;
      if(pFlags != kernel.texAccessInArgs.end())
        tex.flags = pFlags->second;
      tex.isArg = true;
      tex.argId = i;
      accesedTextures.push_back(tex);
    }
  }
  for(const auto& container : kernel.usedContainers)
  {
    if(container.second.isTexture())
    {
      auto pFlags = kernel.texAccessInMemb.find(container.second.name);
      NameFlagsPair tex; 
      tex.name  = container.second.name;
      if(pFlags != kernel.texAccessInMemb.end())
        tex.flags = pFlags->second;
      else
        tex.flags = kslicer::TEX_ACCESS::TEX_ACCESS_SAMPLE;
      tex.isArg = false;
      accesedTextures.push_back(tex);
    }
  }
  return accesedTextures;
}


std::string kslicer::MainFunctionRewriter::MakeKernelCallCmdString(CXXMemberCallExpr* f)
{
  const DeclarationNameInfo dni = f->getMethodDecl()->getNameInfo();
  const DeclarationName dn      = dni.getName();
  const std::string fname       = dn.getAsString();

  const auto pKernelInfo = m_kernels.find(fname);
  assert(pKernelInfo != m_kernels.end());

  const std::string kernName    = m_pCodeInfo->RemoveKernelPrefix(fname);

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
  m_dsTagId++;

  //std::string textOfCall = GetRangeSourceCode(f->getSourceRange(), m_compiler);
  //std::string textOfArgs = textOfCall.substr( textOfCall.find("("));
  
  std::stringstream argsOut;
  argsOut << "(";
  for(size_t i=0;i<f->getNumArgs();i++)
  {
    const Expr* currArgExpr = f->getArgs()[i];
    //std::string text = GetRangeSourceCode(currArgExpr->getSourceRange(), m_compiler);
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
    
    auto accesedTextures = ListAccessedTextures(args, pKernelInfo->second);

    if(pKernel->second.isMaker)
      strOut << "nullptr;" << std::endl << "  ";
    if(pKernel->second.isIndirect)
      strOut << kernName.c_str() << "_UpdateIndirect();" << std::endl << "  ";
    
    //if(accesedTextures.size() != 0)
    //{
    //  strOut << "TrackTextureAccess({";
    //  for(size_t i=0; i < accesedTextures.size();i++)
    //  {
    //    std::string texObjName, accessFlags;
    //    auto pFlagsMemb = pKernel->second.texAccessInMemb.find(accesedTextures[i].name); 
    //    if(pFlagsMemb == pKernel->second.texAccessInMemb.end() && accesedTextures[i].isArg)
    //    {
    //      auto pData = m_pCodeInfo->allDataMembers.find(accesedTextures[i].name);
    //      if(pData != m_pCodeInfo->allDataMembers.end())
    //        texObjName  = "m_vdata." + accesedTextures[i].name + "Texture"; 
    //      else
    //        texObjName  = m_mainFuncName + "_local." + accesedTextures[i].name + "Text"; 
    //
    //      auto argName = pKernel->second.args[accesedTextures[i].argId].name;
    //      auto pFlags  = pKernel->second.texAccessInArgs.find(argName); 
    //      accessFlags  = kslicer::GetDSVulkanAccessMask(pFlags->second);
    //    }
    //    else
    //    {
    //      texObjName  = "m_vdata." + accesedTextures[i].name + "Texture"; 
    //      accessFlags = kslicer::GetDSVulkanAccessMask(pFlagsMemb->second);
    //    }
    //
    //    strOut << "{" << texObjName.c_str() << "," << accessFlags.c_str() << "}";
    //    if(i != accesedTextures.size()-1)
    //      strOut << ", ";
    //    else
    //      strOut << "}, texAccessInfo);";
    //  }
    //  strOut << std::endl << "  ";
    //}
    
    strOut << "vkCmdBindDescriptorSets(a_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ";
    strOut << kernName.c_str() << "Layout," << " 0, 1, " << "&m_allGeneratedDS[" << p2->second << "], 0, nullptr);" << std::endl;
    if(m_pCodeInfo->NeedThreadFlags())
      strOut << "  m_currThreadFlags = " << flagsVariableName.c_str() << ";" << std::endl;
    strOut << "  " << kernName.c_str() << "Cmd" << textOfArgs.c_str() << ";" << std::endl;
    strOut << "  " << "vkCmdPipelineBarrier(m_currCmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memoryBarrier, 0, nullptr, 0, nullptr)";
  }
  
  return strOut.str();
}

std::unordered_set<std::string> kslicer::GetAllServiceKernels()
{
  std::unordered_set<std::string> names;
  names.insert("copyKernelFloat");
  return names;
}


std::string kslicer::MainFunctionRewriter::MakeServiceKernelCallCmdString(CallExpr* call, const std::string& a_name)
{
  std::string kernName = "copyKernelFloat"; // extract from 'call' exact name of service function;
                                            // replace it with actual name we are going to used in generated HOST(!!!) code. 
                                            // for example it can be 'MyMemcpy' for 'memcpy' if in host code we have (MyMemcpyLayout, MyMemcpyPipeline, MyMemcpyDSLayout)
                                            // please note that you should init MyMemcpyLayout, MyMemcpyPipeline, MyMemcpyDSLayout yourself in the generated code!                                      
  
  if(a_name == "memcpy")
    kernName = "copyKernelFloat";

  auto memCpyArgs = ExtractArgumentsOfAKernelCall(call, m_mainFunc.ExcludeList);

  std::vector<ArgReferenceOnCall> args(2); // TODO: extract corretc arguments from memcpy (CallExpr* call)
  {
    args[0].argType = memCpyArgs[0].argType;
    args[0].name    = memCpyArgs[0].name;
    args[0].kind    = DATA_KIND::KIND_POINTER;

    args[1].argType = memCpyArgs[1].argType;
    args[1].name    = memCpyArgs[1].name;
    args[1].kind    = DATA_KIND::KIND_POINTER;
  }

  const auto callSign = MakeKernellCallSignature(m_mainFuncName, args, std::unordered_map<std::string, kslicer::UsedContainerInfo>()); // + strOut1.str();
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
  m_dsTagId++;

  std::stringstream strOut;
  strOut << "vkCmdBindDescriptorSets(a_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ";
  strOut << kernName.c_str() << "Layout," << " 0, 1, " << "&m_allGeneratedDS[" << p2->second << "], 0, nullptr);" << std::endl;
  strOut << "  " << kernName.c_str() << "Cmd(" << memCpyArgs[2].name << " / sizeof(float));" << std::endl;
  strOut << "  " << "vkCmdPipelineBarrier(m_currCmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memoryBarrier, 0, nullptr, 0, nullptr)";

  return strOut.str();
}



bool kslicer::MainFunctionRewriter::VisitCXXMemberCallExpr(CXXMemberCallExpr* f)
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
      std::string callStr = MakeKernelCallCmdString(f);
      m_rewriter.ReplaceText(f->getSourceRange(), callStr); // getExprLoc
    }
    else
    {
      //std::string callStr = MakeKernelCallCmdString(f);
      std::cout << "  [MainFunctionRewriter::VisitCXXMemberCallExpr]: can't process kernel call for " << fname.c_str() << std::endl; 
    }
  }

  return true; 
}

bool kslicer::MainFunctionRewriter::VisitCallExpr(CallExpr* call)
{
  if(isa<CXXMemberCallExpr>(call)) // because we process them in "VisitCXXMemberCallExpr"
    return true;

  const FunctionDecl* fDecl = call->getDirectCallee();  
  if(fDecl == nullptr)             // definitely can't process nullpointer 
    return true;

  // Get name of function
  //
  const DeclarationNameInfo dni = fDecl->getNameInfo();
  const DeclarationName dn      = dni.getName();
  const std::string fname       = dn.getAsString();
  
  if(fname == "memcpy")
  {
    m_pCodeInfo->usedServiceCalls.insert("memcpy");
    std::string testStr = MakeServiceKernelCallCmdString(call, fname);
    m_rewriter.ReplaceText(call->getSourceRange(), testStr);
  }

  return true;
}

bool kslicer::MainFunctionRewriter::VisitIfStmt(IfStmt* ifExpr)
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
    const DeclarationName     dn  = dni.getName();
    const std::string      fname  = dn.getAsString();
  
    if(m_pCodeInfo->IsKernel(fname) && WasNotRewrittenYet(ifExpr))
    {
      std::string callStr = MakeKernelCallCmdString(f);
      m_rewriter.ReplaceText(ifExpr->getSourceRange(), callStr);
      MarkRewritten(ifExpr);
    }
  }

  return true;
}

bool kslicer::MainFunctionRewriter::VisitMemberExpr(MemberExpr* expr)
{
  if(!WasNotRewrittenYet(expr))
    return true;

  std::string setter, containerName;
  if(CheckSettersAccess(expr, m_pCodeInfo, m_compiler, &setter, &containerName))
  {
    std::string name = setter + "Vulkan." + containerName;            
    m_rewriter.ReplaceText(expr->getSourceRange(), name);
    MarkRewritten(expr);
  }
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool kslicer::IsTexture(clang::QualType a_qt)
{
  if(a_qt->isReferenceType())
    a_qt = a_qt.getNonReferenceType();

  auto typeDecl = a_qt->getAsRecordDecl();
  if(typeDecl == nullptr || !clang::isa<clang::ClassTemplateSpecializationDecl>(typeDecl))
    return false;

  const std::string typeName = a_qt.getAsString();
  return (typeName.find("Texture") != std::string::npos || typeName.find("Image") != std::string::npos);
}

bool kslicer::IsAccelStruct(clang::QualType a_qt)
{
  return IsAccelStruct(a_qt.getAsString());
}

bool kslicer::IsAccelStruct(const std::string& a_typeName)
{
  return (a_typeName == "std::shared_ptr<ISceneObject>") || (a_typeName == "std::unique_ptr<ISceneObject>") || 
         (a_typeName == "const std::shared_ptr<ISceneObject>") || (a_typeName == "const std::unique_ptr<ISceneObject>");
}

bool kslicer::IsVectorContainer(const std::string& a_typeName)
{
  return (a_typeName == "vector") || (a_typeName == "std::vector") || (a_typeName == "cvex::vector");  
}

bool kslicer::IsPointerContainer(const std::string& a_typeName)
{
  return (a_typeName == "shared_ptr") || (a_typeName == "unique_ptr") || 
         (a_typeName == "std::shared_ptr") || (a_typeName == "std::unique_ptr");
}


std::vector<kslicer::ArgReferenceOnCall> kslicer::MainFunctionRewriter::ExtractArgumentsOfAKernelCall(CallExpr* f, const std::unordered_set<std::string>& a_excludeList)
{
  std::vector<kslicer::ArgReferenceOnCall> args; 
  args.reserve(20);

  auto predefinedNames = GetAllPredefinedThreadIdNamesRTV();
  
  for(size_t i=0;i<f->getNumArgs();i++)
  {
    const Expr* currArgExpr = f->getArgs()[i];
    const clang::QualType q = currArgExpr->getType();
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
    
    auto checkExpr = currArgExpr;
    if(clang::isa<clang::ImplicitCastExpr>(checkExpr))
    {
      const clang::ImplicitCastExpr* cast = clang::dyn_cast<clang::ImplicitCastExpr>(checkExpr);
      checkExpr = cast->getSubExpr();
    }

    if(clang::isa<clang::IntegerLiteral>(checkExpr) || clang::isa<clang::FloatingLiteral>(checkExpr) || clang::isa<clang::CXXBoolLiteralExpr>(checkExpr) || clang::isa<clang::CompoundLiteralExpr>(checkExpr))
    {
      isConstFound = true;
      isLiteral    = true;
    }

    ArgReferenceOnCall arg; 
    arg.type          = q.getAsString();
    arg.isConst       = q.isConstQualified() || isConstFound;
    arg.isExcludedRTV = (a_excludeList.find(text) != a_excludeList.end());
    if(text[0] == '&')
    {
      arg.umpersanned = true;
      text = text.substr(1);
      auto pClassVar = m_allClassMembers.find(text);
      if(pClassVar != m_allClassMembers.end())
      {
        arg.argType = KERN_CALL_ARG_TYPE::ARG_REFERENCE_CLASS_POD;
        arg.kind    = DATA_KIND::KIND_POD;
      }
    }
    else if(q->isPointerType() && text.find(".data()") != std::string::npos) // TODO: add check for reference, fo the case if we want to pas vectors by reference
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

  // for(auto& arg : args) // check we don't have unknown types of arguments // well, in fact now we can ) Use arguments
  // {
  //   if(arg.argType == KERN_CALL_ARG_TYPE::ARG_REFERENCE_UNKNOWN_TYPE)
  //   {
  //     std::stringstream strOut;
  //     strOut << "  WARNING: expr '" << arg.varName.c_str() << "' was not classified"; 
  //     kslicer::PrintError(strOut.str(), f->getSourceRange(), m_sm);  
  //   }
  // }

  return args;
}

std::string kslicer::MakeKernellCallSignature(const std::string& a_mainFuncName, const std::vector<ArgReferenceOnCall>& a_args, const std::unordered_map<std::string, UsedContainerInfo>& a_usedContainers)
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

    strOut << arg.name.c_str();
  }

  for(const auto& vecName : a_usedContainers)
    strOut << "[MV][" << vecName.second.name.c_str() << "]";

  return strOut.str();
}
