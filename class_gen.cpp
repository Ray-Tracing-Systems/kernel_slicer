#include "kslicer.h"
#include "class_gen.h"
#include "ast_matchers.h"

#include <sstream>
#include <algorithm>

void kslicer::MainClassInfo::AddTempBufferToKernel(const std::string buffName, const std::string a_elemTypeName, KernelInfo& a_kernel)
{
  // (1) append vector to this->allDataMembers
  //
  auto pFoundMember = allDataMembers.find(buffName);
  if(pFoundMember == allDataMembers.end())
  {
    DataMemberInfo vecMemberTmp;
    vecMemberTmp.name          = buffName;
    vecMemberTmp.type          = std::string("std::vector<") + a_elemTypeName + ">";
    vecMemberTmp.sizeInBytes   = 4; // not used by containers
    vecMemberTmp.isContainer   = true;
    vecMemberTmp.usedInKernel  = true;
    vecMemberTmp.containerType = "std::vector";
    vecMemberTmp.containerDataType = a_elemTypeName;
    vecMemberTmp.usage  = kslicer::DATA_USAGE::USAGE_SLICER_REDUCTION; // we dont have to generate code for update of vector or vector size for such vectors
    pFoundMember = allDataMembers.insert({buffName, vecMemberTmp}).first;
  }

  // (2) append vector to a_kernel.usedContainers 
  //
  kslicer::UsedContainerInfo container;
  container.type              = std::string("std::vector<") + a_elemTypeName + ">";
  container.name              = buffName; 
  container.isTexture         = false;
  container.isConst           = false;
  a_kernel.usedContainers[container.name] = container;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool ReplaceFirst(std::string& str, const std::string& from, const std::string& to) 
{
  size_t start_pos = str.find(from);
  if(start_pos == std::string::npos)
    return false;
  str.replace(start_pos, from.length(), to);
  return true;
}

bool CheckTextureAccessFlags(kslicer::TEX_ACCESS a_flags, const std::string& argName, const std::string& a_kernName)
{
  if(a_flags == kslicer::TEX_ACCESS::TEX_ACCESS_READ   || 
     a_flags == kslicer::TEX_ACCESS::TEX_ACCESS_WRITE  || 
     a_flags == kslicer::TEX_ACCESS::TEX_ACCESS_SAMPLE || 
     a_flags == kslicer::TEX_ACCESS::TEX_ACCESS_NOTHING)
    return true;

  //if(int(a_flags) == (int(kslicer::TEX_ACCESS::TEX_ACCESS_READ) | int(kslicer::TEX_ACCESS::TEX_ACCESS_WRITE)))
  //  return true;

  std::cout << "  ERROR: bad ACCESS(Read,Write,Sample) flags: " << int(a_flags) << std::endl;
  std::cout << "  ERROR: texture " << argName.c_str() << ", in kernel " << a_kernName.c_str() << std::endl;

  return false;
}

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

bool kslicer::IsTextureContainer(const std::string& a_typeName)
{
  if(a_typeName == "Texture1D" || a_typeName == "Image1D")
    return true;
  if(a_typeName == "Texture2D" || a_typeName == "Image2D")
    return true;
  if(a_typeName == "Texture3D" || a_typeName == "Image3D")
    return true;
  if(a_typeName == "TextureCube" || a_typeName == "ImageCube")
    return true;

  return false;
} 

bool kslicer::MainClassInfo::IsIndirect(const KernelInfo& a_kernel) const
{
  bool isIndirect = false;
  for(auto& arg : a_kernel.loopIters)
  {
    bool foundSize = (arg.sizeText.find(".size()") != std::string::npos);
    bool foundCap  = (arg.sizeText.find(".capacity()") != std::string::npos);
    bool isMember  = (allDataMembers.find(arg.sizeText) != allDataMembers.end());

    if(foundSize)
    {
      auto pos        = arg.sizeText.find(".size()");
      auto memberName = arg.sizeText.substr(0, pos);
      if(allDataMembers.find(memberName) == allDataMembers.end())
      {
        std::cout << "[ERROR]: Use non-member .size() expression '" << arg.sizeText.c_str() << "' as loop boundary for kernel " <<  a_kernel.name << std::endl;
        std::cout << "[ERROR]: Only class members and member vectors.size() are allowed." << std::endl;
      }
    }

    if(foundCap)
    {
      auto pos        = arg.sizeText.find(".capacity()");
      auto memberName = arg.sizeText.substr(0, pos);
      if(allDataMembers.find(memberName) == allDataMembers.end())
      {
        std::cout << "[ERROR]: Use non-member .capacity() expression '" << arg.sizeText.c_str() << "' as loop boundary for kernel " <<  a_kernel.name << std::endl;
        std::cout << "[ERROR]: Only class members and member vectors.capacity() are allowed." << std::endl;
      }
    }

    isIndirect = isIndirect || foundSize || foundCap || isMember;
  }
  return isIndirect;
} 

static std::string GetControlFuncDeclText(const clang::FunctionDecl* fDecl, clang::CompilerInstance& compiler)
{
  std::string text = fDecl->getNameInfo().getName().getAsString() + "Cmd(VkCommandBuffer a_commandBuffer";
  if(fDecl->getNumParams()!= 0)
    text += ", ";
  for(unsigned i=0;i<fDecl->getNumParams();i++)
  {
    auto pParam = fDecl->getParamDecl(i);
    const clang::QualType typeOfParam =	pParam->getType();
    std::string typeStr = typeOfParam.getAsString();
    //if(!typeOfParam->isPointerType())
    //{
      text += kslicer::GetRangeSourceCode(pParam->getSourceRange(), compiler);
      if(i!=fDecl->getNumParams()-1)
        text += ", ";
    //}
  }

  return text + ")";
}


std::string kslicer::MainClassInfo::GetCFSourceCodeCmd(MainFuncInfo& a_mainFunc, clang::CompilerInstance& compiler)
{
  const std::string&   a_mainClassName = this->mainClassName;
  const CXXMethodDecl* a_node          = a_mainFunc.Node;
  a_mainFunc.GeneratedDecl  = GetCFDeclFromSource(kslicer::GetRangeSourceCode(a_node->getCanonicalDecl()->getSourceRange(), compiler));
  const auto inOutParamList = kslicer::ListPointerParamsOfMainFunc(a_node);

  Rewriter rewrite2;
  rewrite2.setSourceMgr(compiler.getSourceManager(), compiler.getLangOpts());

  a_mainFunc.startDSNumber = allDescriptorSetsInfo.size();

  kslicer::MainFunctionRewriter rv(rewrite2, compiler, a_mainFunc, inOutParamList, this); // ==> write this->allDescriptorSetsInfo during 'TraverseDecl'
  rv.TraverseDecl(const_cast<clang::CXXMethodDecl*>(a_node));
  
  const auto funcBody = a_node->getBody();
  clang::SourceLocation b(funcBody->getBeginLoc()), _e(funcBody->getEndLoc());
  clang::SourceLocation e(clang::Lexer::getLocForEndOfToken(_e, 0, compiler.getSourceManager(), compiler.getLangOpts()));
  
  // (1) TestClass::MainFuncCmd --> TestClass_Generated::MainFuncCmd
  // 
  std::string funcDecl   = a_node->getReturnType().getAsString() + " " + a_mainClassName + "_Generated" + "::" + GetControlFuncDeclText(a_node, compiler);
  std::string sourceCode = rewrite2.getRewrittenText(clang::SourceRange(b,e));

  // (3) set m_currCmdBuffer with input command bufer and add other prolog to MainFunCmd
  //
  std::stringstream strOut;
  strOut << "{" << std::endl;
  strOut << "  m_currCmdBuffer = a_commandBuffer;" << std::endl;
  strOut << "  VkMemoryBarrier memoryBarrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT }; " << std::endl;
  //strOut << "  std::unordered_map<uint64_t, VkAccessFlags> texAccessInfo; " << std::endl;
  strOut << std::endl;

  if(this->NeedThreadFlags())
  {
    strOut << "  const uint32_t outOfForFlags  = KGEN_FLAG_RETURN;" << std::endl;
    strOut << "  const uint32_t inForFlags     = KGEN_FLAG_RETURN | KGEN_FLAG_BREAK;" << std::endl;
    if(a_mainFunc.needToAddThreadFlags)
    {
      const std::string buffName = a_mainFunc.Name + "_local.threadFlagsBuffer";

      strOut << "  const uint32_t outOfForFlagsN = KGEN_FLAG_RETURN | KGEN_FLAG_SET_EXIT_NEGATIVE;" << std::endl;
      strOut << "  const uint32_t inForFlagsN    = KGEN_FLAG_RETURN | KGEN_FLAG_BREAK | KGEN_FLAG_SET_EXIT_NEGATIVE;" << std::endl;
      strOut << "  const uint32_t outOfForFlagsD = KGEN_FLAG_RETURN | KGEN_FLAG_DONT_SET_EXIT;" << std::endl;
      strOut << "  const uint32_t inForFlagsD    = KGEN_FLAG_RETURN | KGEN_FLAG_BREAK | KGEN_FLAG_DONT_SET_EXIT;" << std::endl;
      strOut << "  vkCmdFillBuffer(a_commandBuffer, " << buffName.c_str() << ", 0, VK_WHOLE_SIZE, 0); // zero thread flags, mark all threads to be active" << std::endl;
      strOut << "  VkBufferMemoryBarrier fillBarrier = BarrierForClearFlags(" << buffName.c_str() << "); " << std::endl;
      strOut << "  vkCmdPipelineBarrier(a_commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_DEPENDENCY_BY_REGION_BIT, 0, nullptr, 1, &fillBarrier, 0, nullptr); " << std::endl;
    }
    strOut << std::endl; 
  }

  size_t bracePos = sourceCode.find("{");
  sourceCode = (sourceCode.substr(0, bracePos) + strOut.str() + sourceCode.substr(bracePos+2));
  return funcDecl + sourceCode;
}

std::string kslicer::MainClassInfo::GetCFDeclFromSource(const std::string& sourceCode)
{
  const auto  posOfBracket     = sourceCode.find("(");
  std::string mainFuncDeclHead = sourceCode.substr(0, posOfBracket);
  std::string mainFuncDeclTail = sourceCode.substr(posOfBracket+1);

  while(mainFuncDeclHead[mainFuncDeclHead.size()-1] == ' ')
    mainFuncDeclHead = mainFuncDeclHead.substr(0, mainFuncDeclHead.size()-1);

  return std::string("virtual ") + mainFuncDeclHead + "Cmd(VkCommandBuffer a_commandBuffer, " + mainFuncDeclTail + ";";
}

std::vector<kslicer::InOutVarInfo> kslicer::ListPointerParamsOfMainFunc(const CXXMethodDecl* a_node)
{
  std::vector<InOutVarInfo> params;
  for(unsigned i=0;i<a_node->getNumParams();i++)
  {
    const ParmVarDecl* currParam = a_node->getParamDecl(i);
    const clang::QualType qt     = currParam->getType();
    
    if(qt->isPointerType())
    {
      InOutVarInfo var;
      var.name      = currParam->getNameAsString();
      var.isTexture = false;
      var.isConst   = qt.isConstQualified();
      params.push_back(var);
    }
    else if(qt->isReferenceType() && kslicer::IsTexture(qt))
    {
      InOutVarInfo var;
      var.name      = currParam->getNameAsString();
      var.isTexture = true;
      var.isConst   = qt.isConstQualified();
      params.push_back(var);
    }
  }

  return params;
}

std::string kslicer::MainClassInfo::VisitAndRewrite_KF(KernelInfo& a_funcInfo, const clang::CompilerInstance& compiler, std::string& a_outLoopInitCode, std::string& a_outLoopFinishCode)
{
  const CXXMethodDecl* a_node = a_funcInfo.astNode;
  //a_node->dump();
  
  std::string names[3];
  pShaderCC->GetThreadSizeNames(names);
  if(pShaderCC->IsGLSL())
  {
    names[0] = std::string("kgenArgs.") + names[0];
    names[1] = std::string("kgenArgs.") + names[1];
    names[2] = std::string("kgenArgs.") + names[2];
  }
  std::string fakeOffsetExpr = kslicer::GetFakeOffsetExpression(a_funcInfo, GetKernelTIDArgs(a_funcInfo), names);

  Rewriter rewrite2;
  rewrite2.setSourceMgr(compiler.getSourceManager(), compiler.getLangOpts());
  
  auto pVisitor = pShaderCC->MakeKernRewriter(rewrite2, compiler, this, a_funcInfo, fakeOffsetExpr, false);
  pVisitor->TraverseDecl(const_cast<clang::CXXMethodDecl*>(a_node));
  a_funcInfo.shaderFeatures = a_funcInfo.shaderFeatures || pVisitor->GetKernelShaderFeatures();

  clang::SourceLocation b(a_node->getBeginLoc()), _e(a_node->getEndLoc());
  clang::SourceLocation e(clang::Lexer::getLocForEndOfToken(_e, 0, compiler.getSourceManager(), compiler.getLangOpts()));
  
  return rewrite2.getRewrittenText(clang::SourceRange(b,e));
}

std::vector<kslicer::ArgFinal> kslicer::MainClassInfo::GetKernelTIDArgs(const KernelInfo& a_kernel) const
{
  std::vector<kslicer::ArgFinal> args;
  for (const auto& arg : a_kernel.args) 
  {   
    if(arg.isThreadID)
    { 
      ArgFinal arg2;
      arg2.type = pShaderFuncRewriter->RewriteStdVectorTypeStr(arg.type);
      arg2.name = arg.name;
      arg2.loopIter.sizeText = arg.name;
      arg2.loopIter.id       = 0;
      args.push_back(arg2);
    }
  }

  std::sort(args.begin(), args.end(), [](const auto& a, const auto & b) { return a.name < b.name; });

  return args;
}

std::vector<kslicer::ArgFinal> kslicer::MainClassInfo::GetKernelCommonArgs(const KernelInfo& a_kernel) const
{
  std::vector<kslicer::ArgFinal> args;
  for (const auto& arg : a_kernel.args) 
  { 
    if(!arg.isThreadID && !arg.isLoopSize && !arg.IsUser())
    { 
      ArgFinal arg2;
      arg2.name  = arg.name;
      
      if(arg.IsTexture())
      {
        auto pAccessFlags = a_kernel.texAccessInArgs.find(arg.name);
        CheckTextureAccessFlags(pAccessFlags->second, arg.name, a_kernel.name);
        arg2.isImage   = true;
        arg2.isSampler = (pAccessFlags->second == kslicer::TEX_ACCESS::TEX_ACCESS_SAMPLE);
        arg2.imageType = pShaderFuncRewriter->RewriteImageType(arg.containerType, arg.containerDataType, pAccessFlags->second, arg2.imageFormat);
      }
      else
      {
        arg2.type             = pShaderFuncRewriter->RewriteStdVectorTypeStr(arg.type);
        arg2.isDefinedInClass = (arg.type.find(std::string("class ")  + mainClassName) != std::string::npos || arg.type.find(std::string("struct ") + mainClassName) != std::string::npos);
        arg2.isThreadFlags    = arg.isThreadFlags;
      }
      args.push_back(arg2);
    }
  }

  return args;
}

void kslicer::ObtainKernelsDecl(std::unordered_map<std::string, KernelInfo>& a_kernelsData, const clang::CompilerInstance& compiler, const std::string& a_mainClassName, const MainClassInfo& a_codeInfo)
{
  for (auto& k : a_kernelsData)  
  {
    assert(k.second.astNode != nullptr);
    auto sourceRange = k.second.astNode->getSourceRange();
    
    std::string funcName         = k.second.astNode->getNameInfo().getAsString();
    std::string kernelSourceCode = GetRangeSourceCode(sourceRange, compiler);
    
    auto posBeg      = kernelSourceCode.find(funcName);
    //auto posEnd      = posBeg + funcName.size();
    auto posEndBrace = kernelSourceCode.find(")");

    std::string kernelCmdDecl = kernelSourceCode.substr(posBeg, posEndBrace+1-posBeg);   
    kernelCmdDecl = a_codeInfo.RemoveKernelPrefix(kernelCmdDecl);

    ReplaceFirst(kernelCmdDecl,"(", "Cmd(");
    k.second.DeclCmd = kernelCmdDecl;
    k.second.RetType = kernelSourceCode.substr(0, posBeg);
    ReplaceFirst(k.second.RetType, a_mainClassName + "::", "");
    if(k.second.isBoolTyped)
      ReplaceFirst(k.second.RetType,"bool ", "void ");
  }
}


void kslicer::MainClassInfo::ProcessCallArs_KF(const KernelCallInfo& a_call)
{
  auto pKernel = kernels.find(a_call.originKernelName);
  if(pKernel == kernels.end())
    return;

  for(size_t argId = 0; argId < a_call.descriptorSetsInfo.size(); argId++)
  {
    const auto& currArg = a_call.descriptorSetsInfo[argId];
    const auto& kernArg = pKernel->second.args[argId];

    auto pDataMember = allDataMembers.find(currArg.varName);
    auto pMask       = pKernel->second.texAccessInArgs.find(kernArg.name);
        
    if(pDataMember == allDataMembers.end() || pMask == pKernel->second.texAccessInArgs.end())
      continue;
      
    pDataMember->second.tmask = kslicer::TEX_ACCESS( int(pDataMember->second.tmask) | int(pMask->second) );
  }
}


void kslicer::FunctionRewriter::MarkRewritten(const clang::Stmt* expr) { kslicer::MarkRewrittenRecursive(expr, *m_pRewrittenNodes); }

bool kslicer::FunctionRewriter::WasNotRewrittenYet(const clang::Stmt* expr)
{
  if(expr == nullptr)
    return true;
  if(clang::isa<clang::NullStmt>(expr))
    return true;
  const auto exprHash = kslicer::GetHashOfSourceRange(expr->getSourceRange());
  return (m_pRewrittenNodes->find(exprHash) == m_pRewrittenNodes->end());
}
