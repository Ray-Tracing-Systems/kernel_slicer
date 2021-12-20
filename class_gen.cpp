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
    vecMemberTmp.usage = kslicer::DATA_USAGE::USAGE_SLICER_REDUCTION; // we dont have to generate code for update of vector or vector size for such vectors
    vecMemberTmp.kind  = kslicer::DATA_KIND::KIND_VECTOR;
    pFoundMember = allDataMembers.insert({buffName, vecMemberTmp}).first;
  }

  // (2) append vector to a_kernel.usedContainers 
  //
  kslicer::UsedContainerInfo container;
  container.type              = std::string("std::vector<") + a_elemTypeName + ">";
  container.name              = buffName; 
  container.kind              = DATA_KIND::KIND_VECTOR;
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
    //const clang::QualType typeOfParam =	pParam->getType();
    //std::string typeStr = typeOfParam.getAsString();
    text += kslicer::GetRangeSourceCode(pParam->getSourceRange(), compiler);
    if(i!=fDecl->getNumParams()-1)
      text += ", ";
  }

  return text + ")";
}

static std::string GetOriginalDeclText(const clang::FunctionDecl* fDecl, clang::CompilerInstance& compiler, bool isRTV)
{
  std::string text = fDecl->getNameInfo().getName().getAsString();
  if(isRTV)
    text += "Block";
  text += "(";
  for(unsigned i=0;i<fDecl->getNumParams();i++)
  {
    auto pParam = fDecl->getParamDecl(i);
    //const clang::QualType typeOfParam =	pParam->getType();
    //std::string typeStr = typeOfParam.getAsString();
    text += kslicer::GetRangeSourceCode(pParam->getSourceRange(), compiler);
    if(i!=fDecl->getNumParams()-1)
      text += ", ";
  }
  
  if(isRTV)
  {
    if(fDecl->getNumParams() != 0)
      text += ", ";
    text += "uint32_t a_numPasses";
  }
  return text + ")";
}

void kslicer::MainClassInfo::GetCFSourceCodeCmd(MainFuncInfo& a_mainFunc, clang::CompilerInstance& compiler, bool a_megakernelRTV)
{
  //const std::string&   a_mainClassName = this->mainClassName;
  const CXXMethodDecl* a_node = a_mainFunc.Node;
  const auto inOutParamList   = kslicer::ListParamsOfMainFunc(a_node, compiler);

  const auto funcBody = a_node->getBody();
  clang::SourceLocation b(funcBody->getBeginLoc()), _e(funcBody->getEndLoc());
  clang::SourceLocation e(clang::Lexer::getLocForEndOfToken(_e, 0, compiler.getSourceManager(), compiler.getLangOpts()));

  a_mainFunc.ReturnType    = a_node->getReturnType().getAsString();
  a_mainFunc.GeneratedDecl = GetControlFuncDeclText(a_node, compiler);
  a_mainFunc.startDSNumber = allDescriptorSetsInfo.size();
  a_mainFunc.OriginalDecl  = GetOriginalDeclText(a_node, compiler, IsRTV());
  
  if(a_megakernelRTV)
  {
    // (1) just add allDescriptorSetsInfo for current megakernel 
    //
    a_mainFunc.CodeGenerated = "";
    KernelCallInfo dsInfo;
    dsInfo.callerName = a_mainFunc.Name;
    dsInfo.isService  = false;
    dsInfo.kernelName       = a_mainFunc.Name + "Mega";
    dsInfo.originKernelName = a_mainFunc.Name + "Mega"; // postpone "descriptorSetsInfo" process untill megakernels will be formed at the end
    allDescriptorSetsInfo.push_back(dsInfo);
  }
  else
  {
    // (1) TestClass::MainFunc --> TestClass_Generated::MainFuncCmd
    //
    Rewriter rewrite2;
    rewrite2.setSourceMgr(compiler.getSourceManager(), compiler.getLangOpts());
  
    kslicer::MainFunctionRewriter rv(rewrite2, compiler, a_mainFunc, inOutParamList, this); // ==> write this->allDescriptorSetsInfo during 'TraverseDecl'
    rv.TraverseDecl(const_cast<clang::CXXMethodDecl*>(a_node));
    
    std::string sourceCode   = rewrite2.getRewrittenText(clang::SourceRange(b,e));
    size_t bracePos          = sourceCode.find("{");
    std::string src2         = sourceCode.substr(bracePos+2);
    a_mainFunc.CodeGenerated = src2.substr(0, src2.find_last_of("}"));
  }
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


std::vector<std::string> ParseSizeAttributeText(const std::string& text)
{
  std::string middleText = text.substr(5, text.size()-5-1); // size("w","h") --> "w","h"
  std::vector<std::string> res;
  std::stringstream test(middleText);
  std::string segment;

  while(std::getline(test, segment, ','))
    res.push_back(segment);

  for(auto& attr : res)
  {
    attr.erase(std::remove(attr.begin(), attr.end(), '\"'), attr.end());
    attr.erase(std::remove(attr.begin(), attr.end(), ' '), attr.end());
  }

  return res;
}


kslicer::InOutVarInfo kslicer::GetParamInfo(const clang::ParmVarDecl* currParam, const clang::CompilerInstance& compiler)
{
  auto tidNames = GetAllPredefinedThreadIdNamesRTV();
  const clang::QualType qt = currParam->getType();
  
  auto argInfo = kslicer::ProcessParameter(currParam); 

  InOutVarInfo var;
  var.name      = currParam->getNameAsString();
  var.type      = qt.getAsString();
  auto id       = std::find(tidNames.begin(), tidNames.end(), var.name);
  if(qt->isPointerType())
  {
    var.kind    = DATA_KIND::KIND_POINTER;
    var.isConst = qt->getPointeeType().isConstQualified();
  }
  else if(qt->isReferenceType() && kslicer::IsTexture(qt))
  {
    auto objType          = qt.getNonReferenceType(); 
    var.kind              = DATA_KIND::KIND_TEXTURE;
    var.isConst           = objType.isConstQualified();
    var.containerType     = argInfo.containerType;
    var.containerDataType = argInfo.containerDataType;
  }
  else if(id != tidNames.end())
  {
    var.kind       = DATA_KIND::KIND_POD;
    var.isConst    = qt.isConstQualified();
    var.isThreadId = true;
  }
  
  var.paramNode = currParam;
  if(currParam->hasAttrs())
  {
    auto attrs = currParam->getAttrs();
    for(const auto& attr : attrs)
    {
      const std::string text = kslicer::GetRangeSourceCode(attr->getRange(), compiler);
      if(text.find("size(") != std::string::npos)
        var.sizeUserAttr = ParseSizeAttributeText(text);
    }
  }
  return var;
}

std::vector<kslicer::InOutVarInfo> kslicer::ListParamsOfMainFunc(const CXXMethodDecl* a_node, const clang::CompilerInstance& compiler)
{
  std::vector<InOutVarInfo> params;
  for(unsigned i=0;i<a_node->getNumParams();i++)
  {
    auto var = GetParamInfo(a_node->getParamDecl(i), compiler);
    params.push_back(var);
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
        arg2.isDefinedInClass = (arg.type.find(std::string("class ")  + mainClassName) != std::string::npos || 
                                 arg.type.find(std::string("struct ") + mainClassName) != std::string::npos || 
                                 arg.type.find(mainClassName) != std::string::npos);
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
    auto posEndBrace = kernelSourceCode.find(")");

    std::string kernelCmdDecl = kernelSourceCode.substr(posBeg, posEndBrace+1-posBeg);   
    kernelCmdDecl = a_codeInfo.RemoveKernelPrefix(kernelCmdDecl);
    
    if(k.second.isMega)
      ReplaceFirst(kernelCmdDecl,"(", "MegaCmd(");
    else
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

    auto pDataMember = allDataMembers.find(currArg.name);
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
