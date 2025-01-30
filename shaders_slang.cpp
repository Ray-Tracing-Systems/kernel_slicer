#include "kslicer.h"
#include "template_rendering.h"

#ifdef _WIN32
  #include <sys/types.h>
#endif


bool kslicer::SlangRewriter::NeedToRewriteMemberExpr(const clang::MemberExpr* expr, std::string& out_text)
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
  
  //std::string debugText = kslicer::GetRangeSourceCode(expr->getSourceRange(), m_compiler);

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
    
    bool isKernel = m_codeInfo->IsKernel(m_pCurrKernel->name);

    if(foundId != size_t(-1)) // else we didn't found 'payload' in kernel arguments, so just ignore it
    {
      // now split 'payload->xxx' to 'payload' (baseName) and 'xxx' (memberName); 
      // 
      const std::string exprContent = GetRangeSourceCode(expr->getSourceRange(), m_compiler);
      auto pos = exprContent.find("->");

      if(pos != std::string::npos)
      {    
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
  const bool isInLoopInitPart   = !m_codeInfo->IsRTV() && (expr->getSourceRange().getEnd()   < m_pCurrKernel->loopInsides.getBegin());
  const bool isInLoopFinishPart = !m_codeInfo->IsRTV() && (expr->getSourceRange().getBegin() > m_pCurrKernel->loopInsides.getEnd());
  const bool hasLargeSize     = true; // (pMember->second.sizeInBytes > kslicer::READ_BEFORE_USE_THRESHOLD);
  const bool inMegaKernel     = m_codeInfo->megakernelRTV;
  const bool subjectedToRed   = m_pCurrKernel->subjectedToReduction.find(fieldName) != m_pCurrKernel->subjectedToReduction.end();
  
  if(m_codeInfo->pShaderCC->IsISPC() && subjectedToRed)
    return false;
  
  if(!pMember->second.isContainer && WasNotRewrittenYet(expr) && (isInLoopInitPart || isInLoopFinishPart || !subjectedToRed) && 
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

bool kslicer::SlangRewriter::CheckIfExprHasArgumentThatNeedFakeOffset(const std::string& exprStr)
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

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool kslicer::SlangRewriter::VisitFunctionDecl_Impl(clang::FunctionDecl* fDecl)        
{
  auto hash = kslicer::GetHashOfSourceRange(fDecl->getBody()->getSourceRange());
  if(m_codeInfo->m_functionsDone.find(hash) == m_codeInfo->m_functionsDone.end()) // it is important to put functions in 'm_functionsDone'
  {
    kslicer::RewrittenFunction done;
    done.funDecl = kslicer::GetRangeSourceCode(fDecl->getSourceRange(),            m_compiler); 
    auto posBrace = done.funDecl.find("{");
    if(posBrace != std::string::npos)
      done.funDecl = done.funDecl.substr(0,posBrace); // discard func body source code
    done.funBody = kslicer::GetRangeSourceCode(fDecl->getBody()->getSourceRange(), m_compiler);
    m_codeInfo->m_functionsDone[hash] = done;
  }  
  return true; 
}

bool kslicer::SlangRewriter::VisitCXXMethodDecl_Impl(clang::CXXMethodDecl* fDecl)      { return true; }
bool kslicer::SlangRewriter::VisitMemberExpr_Impl(clang::MemberExpr* expr)             
{
  if(m_kernelMode)
  {
    std::string originalText = kslicer::GetRangeSourceCode(expr->getSourceRange(), m_compiler);
    std::string rewrittenText;
    if(NeedToRewriteMemberExpr(expr, rewrittenText))
    {
      //ReplaceTextOrWorkAround(expr->getSourceRange(), rewrittenText);
      m_rewriter.ReplaceText(expr->getSourceRange(), rewrittenText);
      MarkRewritten(expr);
    }
  }

  return true; 
}
bool kslicer::SlangRewriter::VisitCXXMemberCallExpr_Impl(clang::CXXMemberCallExpr* f)  { return true; } 
bool kslicer::SlangRewriter::VisitFieldDecl_Impl(clang::FieldDecl* decl)               { return true; }
bool kslicer::SlangRewriter::VisitCXXConstructExpr_Impl(clang::CXXConstructExpr* call) { return true; } 
bool kslicer::SlangRewriter::VisitCallExpr_Impl(clang::CallExpr* f)                    { return true; }

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// unually both for kernel and functions
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// unually both for kernel and functions

bool kslicer::SlangRewriter::VisitUnaryOperator_Impl(clang::UnaryOperator* expr)
{ 
  if(m_kernelMode)
  {
    clang::Expr* subExpr = expr->getSubExpr();
    if(subExpr == nullptr)
      return true;

    const auto op = expr->getOpcodeStr(expr->getOpcode());
    if(op == "++" || op == "--") // detect ++ and -- for reduction
    {
      auto opRange = expr->getSourceRange();
      if(opRange.getEnd()   <= m_pCurrKernel->loopInsides.getBegin() || 
         opRange.getBegin() >= m_pCurrKernel->loopInsides.getEnd() ) // not inside loop
        return true;     
      
      const auto op = expr->getOpcodeStr(expr->getOpcode());
      std::string leftStr = kslicer::GetRangeSourceCode(subExpr->getSourceRange(), m_compiler);
  
      auto p = m_pCurrKernel->usedMembers.find(leftStr);
      if(p != m_pCurrKernel->usedMembers.end() && WasNotRewrittenYet(expr))
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
        
        std::string leftStr2   = RecursiveRewrite(expr->getSubExpr()); 
        std::string localIdStr = m_codeInfo->pShaderCC->LocalIdExpr(m_pCurrKernel->GetDim(), m_pCurrKernel->wgSize);
        ReplaceTextOrWorkAround(expr->getSourceRange(), leftStr2 + "Shared[" + localIdStr + "]++");
        MarkRewritten(expr);
      }
    }

    // detect " *(something)"
    //
    const std::string exprInside = RecursiveRewrite(subExpr);

    if(op == "*" && !expr->canOverflow() && CheckIfExprHasArgumentThatNeedFakeOffset(exprInside) && WasNotRewrittenYet(expr)) 
    {
      if(m_codeInfo->megakernelRTV || m_fakeOffsetExp == "")
        ReplaceTextOrWorkAround(expr->getSourceRange(), exprInside);
      else
        ReplaceTextOrWorkAround(expr->getSourceRange(), exprInside + "[" + m_fakeOffsetExp + "]");
      MarkRewritten(expr);
    }
  }

  return true; 
}

bool kslicer::SlangRewriter::VisitCXXOperatorCallExpr_Impl(clang::CXXOperatorCallExpr* expr) 
{ 
  if(m_kernelMode)
  {
    // ...
  }

  return true; 
}

bool kslicer::SlangRewriter::VisitVarDecl_Impl(clang::VarDecl* decl) 
{ 
  if(m_kernelMode)
  {
    // ...
  }

  return true; 
}

bool kslicer::SlangRewriter::VisitCStyleCastExpr_Impl(clang::CStyleCastExpr* cast)           
{ 
  if(m_kernelMode)
  {
    // ...
  }

  return true;   
}

bool kslicer::SlangRewriter::VisitImplicitCastExpr_Impl(clang::ImplicitCastExpr* cast) 
{ 
  if(m_kernelMode)
  {
    // ...
  }

  return true; 
}

bool kslicer::SlangRewriter::VisitDeclStmt_Impl(clang::DeclStmt* decl)             
{ 
  if(m_kernelMode)
  {
    // ...
  }

  return true; 
}

bool kslicer::SlangRewriter::VisitArraySubscriptExpr_Impl(clang::ArraySubscriptExpr* arrayExpr) 
{ 
  if(m_kernelMode)
  {
    // ...
  }

  return true; 
}

bool kslicer::SlangRewriter::VisitUnaryExprOrTypeTraitExpr_Impl(clang::UnaryExprOrTypeTraitExpr* szOfExpr) 
{ 
  if(m_kernelMode)
  {
    // ...
  }

  return true; 
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// usually only for kernel
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// usually only for kernel

bool kslicer::SlangRewriter::VisitCompoundAssignOperator_Impl(clang::CompoundAssignOperator* expr) 
{ 
  if(m_kernelMode)
  {
    // ...
  }

  return true; 
}

bool kslicer::SlangRewriter::VisitBinaryOperator_Impl(clang::BinaryOperator* expr)
{
  if(m_kernelMode)
  {
    // ...
  }

  return true;
}

bool  kslicer::SlangRewriter::VisitDeclRefExpr_Impl(clang::DeclRefExpr* expr) 
{
  if(m_kernelMode)
  {
    const clang::ValueDecl* pDecl = expr->getDecl();
    if(!clang::isa<clang::ParmVarDecl>(pDecl))
      return true;
  
    clang::QualType qt = pDecl->getType();
    if(qt->isPointerType() || qt->isReferenceType()) // we can't put references to push constants
      return true;
  
    const std::string textOri = kslicer::GetRangeSourceCode(expr->getSourceRange(), m_compiler); //
    //const std::string textRes = RecursiveRewrite(expr);
    if(m_kernelUserArgs.find(textOri) != m_kernelUserArgs.end() && WasNotRewrittenYet(expr))
    {
      if(!m_codeInfo->megakernelRTV || m_pCurrKernel->isMega)
      {
        //ReplaceTextOrWorkAround(expr->getSourceRange(), std::string("kgenArgs.") + textOri);
        m_rewriter.ReplaceText(expr->getSourceRange(), std::string("kgenArgs.") + textOri);
        MarkRewritten(expr);
      }
    }
  }

  return true;
}

std::string kslicer::SlangRewriter::RecursiveRewrite(const clang::Stmt* expr)
{
  if(expr == nullptr)
    return "";
    
  SlangRewriter rvCopy = *this;
  rvCopy.TraverseStmt(const_cast<clang::Stmt*>(expr));

  auto range = expr->getSourceRange();
  auto p = rvCopy.m_workAround.find(GetHashOfSourceRange(range));
  if(p != rvCopy.m_workAround.end())
    return p->second;
  else
    return m_rewriter.getRewrittenText(range);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


kslicer::SlangCompiler::SlangCompiler(const std::string& a_prefix) : m_suffix(a_prefix)
{

}


std::string kslicer::SlangCompiler::LocalIdExpr(uint32_t a_kernelDim, uint32_t a_wgSize[3]) const
{
  // uint3 a_localTID  : SV_GroupThreadID
  // uint3 a_globalTID : SV_DispatchThreadID
  if(a_kernelDim == 1)
    return "a_localTID.x";
  else if(a_kernelDim == 2)
  {
    std::stringstream strOut;
    strOut << "a_localTID.x + " << a_wgSize[0] << "*a_localTID.y";
    return strOut.str();
  }
  else if(a_kernelDim == 3)
  {
    std::stringstream strOut;
    strOut << "a_localTID.x + " << a_wgSize[0] << "*a_localTID.y + " << a_wgSize[0]*a_wgSize[1] << "*a_localTID.z";
    return strOut.str();
  }
  else
  {
    std::cout << "  [SlangCompiler::LocalIdExpr]: Error, bad kernelDim = " << a_kernelDim << std::endl;
    return "a_localTID.x";
  }
}

void kslicer::SlangCompiler::GetThreadSizeNames(std::string a_strs[3]) const
{
  a_strs[0] = "iNumElementsX";
  a_strs[1] = "iNumElementsY";
  a_strs[2] = "iNumElementsZ";
}

std::string kslicer::SlangCompiler::ProcessBufferType(const std::string& a_typeName) const
{
  std::string type = kslicer::CleanTypeName(a_typeName);
  ReplaceFirst(type, "*", "");
  if(type[type.size()-1] == ' ')
    type = type.substr(0, type.size()-1);

  return type;
}

std::string kslicer::SlangCompiler::RewritePushBack(const std::string& memberNameA, const std::string& memberNameB, const std::string& newElemValue) const
{
  return std::string("{ uint offset = atomicAdd(") + UBOAccess(memberNameB) + ", 1); " + memberNameA + "[offset] = " + newElemValue + ";}";
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::shared_ptr<kslicer::FunctionRewriter> kslicer::SlangCompiler::MakeFuncRewriter(clang::Rewriter &R, const clang::CompilerInstance& a_compiler, 
                                                                                    MainClassInfo* a_codeInfo, kslicer::ShittyFunction a_shit)
{
  return std::make_shared<SlangRewriter>(R, a_compiler, a_codeInfo);
}

std::shared_ptr<kslicer::KernelRewriter> kslicer::SlangCompiler::MakeKernRewriter(clang::Rewriter &R, const clang::CompilerInstance& a_compiler, 
                                                                                  MainClassInfo* a_codeInfo, kslicer::KernelInfo& a_kernel, const std::string& fakeOffs)
{
  auto pFunc = std::make_shared<SlangRewriter>(R, a_compiler, a_codeInfo);
  pFunc->InitKernelData(a_kernel, fakeOffs);
  return std::make_shared<KernelRewriter2>(R, a_compiler, a_codeInfo, a_kernel, fakeOffs, pFunc);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void kslicer::SlangCompiler::GenerateShaders(nlohmann::json& a_kernelsJson, const MainClassInfo* a_codeInfo, const kslicer::TextGenSettings& a_settings)
{
  const auto& mainClassFileName = a_codeInfo->mainClassFileName;
  const auto& ignoreFolders     = a_codeInfo->ignoreFolders;

  #ifdef _WIN32
  const std::string scriptName = "build_slang.bat";
  #else
  const std::string scriptName = "build_slang.sh";
  #endif

  std::filesystem::path folderPath = mainClassFileName.parent_path();
  std::filesystem::path shaderPath = folderPath / this->ShaderFolder();
  std::filesystem::path incUBOPath = folderPath / "include";
  std::filesystem::create_directory(shaderPath);
  std::filesystem::create_directory(incUBOPath);

  // generate header for all used functions in GLSL code
  //
  std::string headerCommon = "common" + ToLowerCase(m_suffix) + "_slang.h";
  std::filesystem::path templatesFolder("templates_slang");
  kslicer::ApplyJsonToTemplate(templatesFolder / "common_generated_slang.h", shaderPath / headerCommon, a_kernelsJson);

  const std::filesystem::path templatePath       = templatesFolder / (a_codeInfo->megakernelRTV ? "generated_mega.slang" : "generated.slang");
  const std::filesystem::path templatePathUpdInd = templatesFolder / "update_indirect.slang";
  const std::filesystem::path templatePathRedFin = templatesFolder / "reduction_finish.slang";
  
  nlohmann::json copy, kernels, intersections;
  for (auto& el : a_kernelsJson.items())
  {
    //std::cout << el.key() << std::endl;
    if(std::string(el.key()) == "Kernels")
      kernels = a_kernelsJson[el.key()];
    else
      copy[el.key()] = a_kernelsJson[el.key()];
  }

  std::ofstream buildSH(shaderPath / scriptName);
  #if not __WIN32__
  buildSH << "#!/bin/sh" << std::endl;
  #endif
  for(auto& kernel : kernels.items())
  {
    nlohmann::json currKerneJson = copy;
    currKerneJson["Kernel"] = kernel.value();

    std::string kernelName     = std::string(kernel.value()["Name"]);
    bool useRayTracingPipeline = kernel.value()["UseRayGen"];
    const bool vulkan11        = kernel.value()["UseSubGroups"];
    const bool vulkan12        = useRayTracingPipeline;

    std::string outFileName           = kernelName + ".slang";
    std::filesystem::path outFilePath = shaderPath / outFileName;
    kslicer::ApplyJsonToTemplate(templatePath.c_str(), outFilePath, currKerneJson);
    
    buildSH << "slangc " << outFileName.c_str() << " -o " << kernelName.c_str() << ".comp.spv" << " -I.. ";
    for(auto folder : ignoreFolders)
      buildSH << "-I" << folder.c_str() << " ";
    //if(vulkan12)
    //  buildSH << "--target-env vulkan1.2 ";
    //else if(vulkan11)
    //  buildSH << "--target-env vulkan1.1 ";
    //if(useRayTracingPipeline)
    //  buildSH << "-S rgen ";
    buildSH << std::endl;
  
    //if(kernel.value()["IsIndirect"])
    //{
    //  outFileName = kernelName + "_UpdateIndirect.slang";
    //  outFilePath = shaderPath / outFileName;
    //  kslicer::ApplyJsonToTemplate(templatePathUpdInd.c_str(), outFilePath, currKerneJson);
    //  buildSH << "glslangValidator -V ";
    //  if(vulkan11)
    //    buildSH << "--target-env vulkan1.1 ";
    //  buildSH << outFileName.c_str() << " -o " << outFileName.c_str() << ".spv" << " -DGLSL -I.. ";
    //  for(auto folder : ignoreFolders)
    //   buildSH << "-I" << folder.c_str() << " ";
    //  buildSH << std::endl;
    //}

    //if(kernel.value()["FinishRed"])
    //{
    //  outFileName = kernelName + "_Reduction.slang";
    //  outFilePath = shaderPath / outFileName;
    //  kslicer::ApplyJsonToTemplate(templatePathRedFin.c_str(), outFilePath, currKerneJson);
    //  buildSH << "glslangValidator -V ";
    //  if(vulkan11)
    //    buildSH << "--target-env vulkan1.1 ";
    //  buildSH << outFileName.c_str() << " -o " << outFileName.c_str() << ".spv" << " -DGLSL -I.. ";
    //  for(auto folder : ignoreFolders)
    //   buildSH << "-I" << folder.c_str() << " ";
    //  buildSH << std::endl;
    //}
  }

}


std::string kslicer::SlangCompiler::PrintHeaderDecl(const DeclInClass& a_decl, const clang::CompilerInstance& a_compiler, std::shared_ptr<kslicer::FunctionRewriter> a_pRewriter)
{
  std::string typeInCL = a_decl.type;
  ReplaceFirst(typeInCL, "struct ", "");

  std::string result = "";
  switch(a_decl.kind)
  {
    case kslicer::DECL_IN_CLASS::DECL_STRUCT:
    result = kslicer::GetRangeSourceCode(a_decl.srcRange, a_compiler) + ";";
    break;
    case kslicer::DECL_IN_CLASS::DECL_CONSTANT:
    ReplaceFirst(typeInCL, "_Bool", "bool");
    result = "static " + typeInCL + " " + a_decl.name + " = " + kslicer::GetRangeSourceCode(a_decl.srcRange, a_compiler) + ";";
    break;
    case kslicer::DECL_IN_CLASS::DECL_TYPEDEF:
    result = "typealias " + a_decl.name + " = " + typeInCL + ";";
    break;
    default:
    break;
  };
  return result;
}

std::string kslicer::SlangCompiler::RTVGetFakeOffsetExpression(const kslicer::KernelInfo& a_funcInfo, const std::vector<kslicer::ArgFinal>& threadIds)
{
  std::string names[3];
  this->GetThreadSizeNames(names);

  const std::string names0 = std::string("kgenArgs.") + names[0];
  const std::string names1 = std::string("kgenArgs.") + names[1];
  const std::string names2 = std::string("kgenArgs.") + names[2];

  if(threadIds.size() == 1)
    return threadIds[0].name;
  else if(threadIds.size() == 2)
    return std::string("fakeOffset(") + threadIds[0].name + "," + threadIds[1].name + "," + names0 + ")";
  else if(threadIds.size() == 3)
    return std::string("fakeOffset2(") + threadIds[0].name + "," + threadIds[1].name + "," + threadIds[2].name + "," + names0 + "," + names1 + ")";
  else
    return "a_globalTID.x";
} 
