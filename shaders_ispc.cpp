#include "kslicer.h"
#include "template_rendering.h"
#include <fstream>

#ifdef _WIN32
  #include <sys/types.h>
#endif

void kslicer::ISPCRewriter::Init()
{ 
  m_funReplacements.clear();
  //m_funReplacements["atomicAdd"] = "InterlockedAdd";
  //m_funReplacements["AtomicAdd"] = "InterlockedAdd";
}

std::string kslicer::ISPCRewriter::RewriteStdVectorTypeStr(const std::string& a_str) const
{
  const bool isConst = (a_str.find("const ") != std::string::npos);
  std::string typeStr = a_str;
  ReplaceFirst(typeStr, "struct LiteMath::", "");
  ReplaceFirst(typeStr, "LiteMath::", "");
  ReplaceFirst(typeStr, "struct glm::", "");
  ReplaceFirst(typeStr, "glm::", "");
  ReplaceFirst(typeStr, "const ",    "");
  ReplaceFirst(typeStr, m_codeInfo->mainClassName + "::", "");
  ReplaceFirst(typeStr, "struct float4x4", "float4x4");       // small inconvinience in math library
  if(isConst)
    typeStr = std::string("const ") + typeStr;
  return typeStr;
}

// process arrays: 'float[3] data' --> 'float data[3]' 
std::string kslicer::ISPCRewriter::RewriteStdVectorTypeStr(const std::string& a_typeName, std::string& varName) const
{
  auto typeNameR = a_typeName;
  auto posArrayBegin = typeNameR.find("[");
  auto posArrayEnd   = typeNameR.find("]");
  if(posArrayBegin != std::string::npos && posArrayEnd != std::string::npos)
  {
    varName   = varName + typeNameR.substr(posArrayBegin, posArrayEnd-posArrayBegin+1);
    typeNameR = typeNameR.substr(0, posArrayBegin);
  }

  return RewriteStdVectorTypeStr(typeNameR);
}

std::string kslicer::ISPCRewriter::RewriteFuncDecl(clang::FunctionDecl* fDecl)
{
  std::string retT   = RewriteStdVectorTypeStr(fDecl->getReturnType().getAsString());
  std::string fname  = fDecl->getNameInfo().getName().getAsString();

  if(m_pCurrFuncInfo != nullptr && m_pCurrFuncInfo->hasPrefix)          // alter function name if it has any prefix
  { 
    if(fname.find(m_pCurrFuncInfo->prefixName) == std::string::npos)
      fname = m_pCurrFuncInfo->prefixName + "_" + fname;
  }
  else if(m_pCurrFuncInfo != nullptr && m_pCurrFuncInfo->name != fname) // alter function name if was changed
    fname = m_pCurrFuncInfo->name;

  std::string result = retT + " " + fname + "(";

  for(uint32_t i=0; i < fDecl->getNumParams(); i++)
  {
    const clang::ParmVarDecl* pParam  = fDecl->getParamDecl(i);
    const clang::QualType typeOfParam =	pParam->getType();
    std::string typeStr = typeOfParam.getAsString();
    if(typeOfParam->isPointerType())
      typeStr += "*";
    else if(typeOfParam->isReferenceType())
      typeStr += "&";
    
    result += RewriteStdVectorTypeStr(typeStr) + " " + pParam->getNameAsString();
    if(i!=fDecl->getNumParams()-1)
      result += ", ";
  }

  return result + ") ";
}

bool kslicer::ISPCRewriter::NeedsVectorTypeRewrite(const std::string& a_str) // TODO: make this implementation more smart, bad implementation actually!
{
  std::string typeStr = kslicer::CleanTypeName(a_str);
  return (m_typesReplacement.find(typeStr) != m_typesReplacement.end());
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool kslicer::ISPCRewriter::VisitFunctionDecl_Impl(clang::FunctionDecl* fDecl)        
{
  auto hash = kslicer::GetHashOfSourceRange(fDecl->getBody()->getSourceRange());
  if(m_codeInfo->m_functionsDone.find(hash) == m_codeInfo->m_functionsDone.end())
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

bool kslicer::ISPCRewriter::VisitCXXMethodDecl_Impl(clang::CXXMethodDecl* fDecl)      
{ 
  return true; 
}

bool kslicer::ISPCRewriter::VisitMemberExpr_Impl(clang::MemberExpr* expr)             
{
  if(m_kernelMode)
  {
    std::string originalText = kslicer::GetRangeSourceCode(expr->getSourceRange(), m_compiler);
    std::string rewrittenText;
    if(NeedToRewriteMemberExpr(expr, rewrittenText))
    {
      //ReplaceTextOrWorkAround(expr->getSourceRange(), rewrittenText);
      kslicer::ReplaceTextSafe(m_rewriter, expr->getSourceRange(), rewrittenText);
      MarkRewritten(expr);
    }
  }

  return true; 
}

bool kslicer::ISPCRewriter::VisitCXXMemberCallExpr_Impl(clang::CXXMemberCallExpr* f)  
{ 
  if(m_kernelMode)
  {
    // Get name of function
    //
    const clang::DeclarationNameInfo dni = f->getMethodDecl()->getNameInfo();
    const clang::DeclarationName dn      = dni.getName();
          std::string fname       = dn.getAsString();

    // Get name of "this" type; we should check wherther this member is std::vector<T>  
    //
    const clang::QualType qt = f->getObjectType();
    const auto& thisTypeName = qt.getAsString();
    clang::CXXRecordDecl* typeDecl  = f->getRecordDecl(); 
    const std::string cleanTypeName = kslicer::CleanTypeName(thisTypeName);
    
    const bool isVector   = (typeDecl != nullptr && clang::isa<clang::ClassTemplateSpecializationDecl>(typeDecl)) && thisTypeName.find("vector<") != std::string::npos; 
    const bool isRTX      = (thisTypeName == "struct ISceneObject") && (fname.find("RayQuery_") != std::string::npos);
    const auto pPrefix    = m_codeInfo->composPrefix.find(cleanTypeName);
    const bool isPrefixed = (pPrefix != m_codeInfo->composPrefix.end());
    
    if(isVector && WasNotRewrittenYet(f))
    {
      const std::string exprContent = GetRangeSourceCode(f->getSourceRange(), m_compiler);
      const auto posOfPoint         = exprContent.find(".");
      std::string memberNameA       = exprContent.substr(0, posOfPoint);
      
      if(processFuncMember && m_pCurrFuncInfo != nullptr && m_pCurrFuncInfo->hasPrefix)
        memberNameA = m_pCurrFuncInfo->prefixName + "_" + memberNameA;
  
      if(fname == "size" || fname == "capacity")
      {
        const std::string memberNameB = memberNameA + "_" + fname;
        ReplaceTextOrWorkAround(f->getSourceRange(), m_codeInfo->pShaderCC->UBOAccess(memberNameB) );
        MarkRewritten(f);
      }
      else if(fname == "resize")
      {
        //if(f->getSourceRange().getBegin() <= m_currKernel.loopOutsidesInit.getEnd()) // TODO: SEEMS INCORECT LOGIC
        //{
        //  assert(f->getNumArgs() == 1);
        //  const clang::Expr* currArgExpr  = f->getArgs()[0];
        //  std::string newSizeValue = RecursiveRewrite(currArgExpr); 
        //  std::string memberNameB  = memberNameA + "_size = " + newSizeValue;
        //  ReplaceTextOrWorkAround(f->getSourceRange(), m_codeInfo->pShaderCC->UBOAccess(memberNameB) );
        //  MarkRewritten(f);
        //}
      }
      else if(fname == "push_back")
      {
        assert(f->getNumArgs() == 1);
        const clang::Expr* currArgExpr  = f->getArgs()[0];
        std::string newElemValue = RecursiveRewrite(currArgExpr);
  
        std::string memberNameB  = memberNameA + "_size";
        std::string resulingText = m_codeInfo->pShaderCC->RewritePushBack(memberNameA, memberNameB, newElemValue);
        ReplaceTextOrWorkAround(f->getSourceRange(), resulingText);
        MarkRewritten(f);
      }
      else if(fname == "data")
      {
        ReplaceTextOrWorkAround(f->getSourceRange(), memberNameA);
        MarkRewritten(f);
      }
      else
      {
        kslicer::PrintError(std::string("Unsuppoted std::vector method") + fname, f->getSourceRange(), m_compiler.getSourceManager());
      }
    }
    else if(isRTX && WasNotRewrittenYet(f)) // 'm_pAccelStruct->RayQuery_NearestHit(a,b)' ==> 'm_pAccelStruct_RayQuery_NearestHit(a,b)', see 'templates_ispc/generated.ispc'
    {
      const std::string exprContent = GetRangeSourceCode(f->getSourceRange(), m_compiler);
      const std::string memberNameA = exprContent.substr(0, exprContent.find("->"));

      std::string resCallText = memberNameA + "_" + fname + "(";
      for(unsigned i=0;i<f->getNumArgs(); i++)
      {
        resCallText += RecursiveRewrite(f->getArg(i));
        if(i!=f->getNumArgs()-1)
          resCallText += ", ";
      }
      resCallText += ")";
      ReplaceTextOrWorkAround(f->getSourceRange(), resCallText);
      MarkRewritten(f);
    }
  }

  return true; 
} 

bool kslicer::ISPCRewriter::VisitFieldDecl_Impl(clang::FieldDecl* decl) { return true; }

std::string kslicer::ISPCRewriter::VectorTypeContructorReplace(const std::string& fname, const std::string& callText)
{
  return fname + callText;
}

// 'float3(a,b,c)' ==> 'make_float3(a,b,c)', 'float3(a)' ==> '(float3)(a)'; ISPC has no constructors for short vectors
//
bool kslicer::ISPCRewriter::VisitCXXConstructExpr_Impl(clang::CXXConstructExpr* call) 
{ 
  const clang::CXXConstructorDecl* ctorDecl = call->getConstructor();
  if(ctorDecl == nullptr || ctorDecl->isCopyOrMoveConstructor() || call->getNumArgs() == 0 || !WasNotRewrittenYet(call))
    return true;

  const std::string typeName = ctorDecl->getNameInfo().getName().getAsString();
  static const std::unordered_set<std::string> vecTypes = {"float2", "float3", "float4", "int2", "int3", "int4", "uint2", "uint3", "uint4"};
  if(vecTypes.find(typeName) == vecTypes.end())
    return true;

  const uint32_t vecSize = uint32_t(typeName[typeName.size()-1] - '0');
  std::string text;
  if(call->getNumArgs() == vecSize)
  {
    text = "make_" + typeName + "(";
    for(unsigned i=0;i<call->getNumArgs();i++)
    {
      text += RecursiveRewrite(call->getArg(i));
      if(i != call->getNumArgs()-1)
        text += ", ";
    }
    text += ")";
  }
  else if(call->getNumArgs() == 1 && call->getArg(0)->getType()->isBuiltinType())
    text = "(" + typeName + ")(" + RecursiveRewrite(call->getArg(0)) + ")";
  else
    return true;

  ReplaceTextOrWorkAround(call->getSourceRange(), text);
  MarkRewritten(call);
  return true; 
} 

// 'float(x)' ==> '(float)(x)'
//
bool kslicer::ISPCRewriter::VisitCXXFunctionalCastExpr_Impl(clang::CXXFunctionalCastExpr* cast)
{
  if(!cast->getType()->isBuiltinType() || !WasNotRewrittenYet(cast))
    return true;
  const std::string typeName = RewriteStdVectorTypeStr(cast->getTypeAsWritten().getAsString());
  ReplaceTextOrWorkAround(cast->getSourceRange(), "(" + typeName + ")(" + RecursiveRewrite(cast->getSubExpr()) + ")");
  MarkRewritten(cast);
  return true;
}

bool kslicer::ISPCRewriter::VisitCallExpr_Impl(clang::CallExpr* call)                    
{ 
  if(m_kernelMode && WasNotRewrittenYet(call))
  {
    clang::FunctionDecl* fDecl = call->getDirectCallee();
    if(fDecl == nullptr)
      return true;

    const std::string fname    = fDecl->getNameInfo().getName().getAsString();
    const std::string callText = GetRangeSourceCode(call->getSourceRange(), m_compiler);
    const auto ddPos = callText.find("::");
    if(ddPos != std::string::npos)
    {
      std::string funcName = fname;
      const std::string baseClassName = callText.substr(0, ddPos);
      if(baseClassName != m_codeInfo->mainClassName && m_codeInfo->mainClassNames.find(baseClassName) != m_codeInfo->mainClassNames.end())
      {
        funcName = baseClassName + "_" + fname;
      }
      const std::string lastRewrittenText = funcName + "(" + CompleteFunctionCallRewrite(call);
      ReplaceTextOrWorkAround(call->getSourceRange(), lastRewrittenText);
      MarkRewritten(call);
    }
  }

  return true; 
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// unually both for kernel and functions
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// unually both for kernel and functions

bool kslicer::ISPCRewriter::VisitUnaryOperator_Impl(clang::UnaryOperator* expr)
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
        
        //std::string leftStr2   = RecursiveRewrite(expr->getSubExpr()); 
        //std::string localIdStr = m_codeInfo->pShaderCC->LocalIdExpr(m_pCurrKernel->GetDim(), m_pCurrKernel->wgSize);
        //ReplaceTextOrWorkAround(expr->getSourceRange(), leftStr2 + "Shared[" + localIdStr + "]++");
        //MarkRewritten(expr);
      }
    }

    // detect " *something and &something"
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

bool kslicer::ISPCRewriter::VisitCXXOperatorCallExpr_Impl(clang::CXXOperatorCallExpr* expr) 
{ 
  if(m_kernelMode)
  {
    FunctionRewriter2::VisitCXXOperatorCallExpr_Impl(expr);
  }

  return true; 
}

bool kslicer::ISPCRewriter::VisitVarDecl_Impl(clang::VarDecl* decl) 
{ 
  if(clang::isa<clang::ParmVarDecl>(decl)) // process else-where (VisitFunctionDecl_Impl)
    return true;

  if(m_kernelMode)
  {
    // ...
  }

  return true; 
}

bool kslicer::ISPCRewriter::VisitCStyleCastExpr_Impl(clang::CStyleCastExpr* cast)           
{ 
  if(m_kernelMode)
  {
    // ...
  }

  return true;   
}

bool kslicer::ISPCRewriter::VisitImplicitCastExpr_Impl(clang::ImplicitCastExpr* cast) 
{ 
  if(m_kernelMode)
  {
    // ...
  }

  return true; 
}

bool kslicer::ISPCRewriter::VisitDeclStmt_Impl(clang::DeclStmt* decl)             
{ 
  if(m_kernelMode)
  {
    // ...
  }

  return true; 
}

bool kslicer::ISPCRewriter::VisitFloatingLiteral_Impl(clang::FloatingLiteral* expr) 
{
  return true;
}

bool kslicer::ISPCRewriter::VisitArraySubscriptExpr_Impl(clang::ArraySubscriptExpr* arrayExpr) 
{ 
  if(m_kernelMode)
  {
    // ...
  }

  return true; 
}

bool kslicer::ISPCRewriter::VisitUnaryExprOrTypeTraitExpr_Impl(clang::UnaryExprOrTypeTraitExpr* szOfExpr) 
{ 
  if(m_kernelMode)
  {
    // ...
  }

  return true; 
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// usually only for kernel
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// usually only for kernel

bool kslicer::ISPCRewriter::VisitCompoundAssignOperator_Impl(clang::CompoundAssignOperator* expr) 
{ 
  if(m_kernelMode)
  {
    
  }

  return true; 
}

bool kslicer::ISPCRewriter::VisitBinaryOperator_Impl(clang::BinaryOperator* expr)
{
  if(m_kernelMode)
  {
    
  }

  return true;
}

bool  kslicer::ISPCRewriter::VisitDeclRefExpr_Impl(clang::DeclRefExpr* expr) 
{
  if(m_kernelMode)
  { 

  }

  return true;
}

std::string kslicer::ISPCRewriter::RecursiveRewrite(const clang::Stmt* expr)
{
  if(expr == nullptr)
    return "";
  
  std::string shallow;
  if(DetectAndRewriteShallowPattern(expr, shallow)) 
    return shallow;
    
  ISPCRewriter rvCopy = *this;
  rvCopy.TraverseStmt(const_cast<clang::Stmt*>(expr));

  auto range = expr->getSourceRange();
  auto p = rvCopy.m_workAround.find(GetHashOfSourceRange(range));
  if(p != rvCopy.m_workAround.end())
    return p->second;
  else
    return kslicer::GetRewrittenText(m_rewriter, range);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

constexpr static bool NEW_REWRITER = true;

kslicer::ISPCCompiler::ISPCCompiler(bool a_useCPP, const std::string& a_prefix) : ClspvCompiler(a_useCPP, a_prefix)
{

}

std::shared_ptr<kslicer::FunctionRewriter> kslicer::ISPCCompiler::MakeFuncRewriter(clang::Rewriter &R, const clang::CompilerInstance& a_compiler, MainClassInfo* a_codeInfo, kslicer::ShittyFunction a_shit)
{
  if(NEW_REWRITER)
  {
    auto pFunc = std::make_shared<ISPCRewriter>(R, a_compiler, a_codeInfo);
    pFunc->m_shit = a_shit;
    return pFunc;
  }
  else
    return std::make_shared<kslicer::FunctionRewriter>(R, a_compiler, a_codeInfo);
}

std::shared_ptr<kslicer::KernelRewriter> kslicer::ISPCCompiler::MakeKernRewriter(clang::Rewriter &R, const clang::CompilerInstance& a_compiler, MainClassInfo* a_codeInfo,
                                                                                 kslicer::KernelInfo& a_kernel, const std::string& fakeOffs)
{
  if(NEW_REWRITER)
  {
    auto pFunc = std::make_shared<ISPCRewriter>(R, a_compiler, a_codeInfo);
    pFunc->InitKernelData(a_kernel, fakeOffs);
    return std::make_shared<KernelRewriter2>(R, a_compiler, a_codeInfo, a_kernel, fakeOffs, pFunc);
  }
  else
    return std::make_shared<kslicer::KernelRewriter>(R, a_compiler, a_codeInfo, a_kernel, fakeOffs);
}


std::string kslicer::ISPCCompiler::BuildCommand(const std::string& a_inputFile) const
{
  return std::string("ispc ") + a_inputFile + " --target=" + m_target + " -O2 ";
}

void kslicer::ISPCCompiler::GenerateShaders(nlohmann::json& a_kernelsJson, const MainClassInfo* a_codeInfo, const kslicer::TextGenSettings& a_settings)
{
  const auto& ignoreFolders     = a_codeInfo->ignoreFolders;

  // output names are based on original input file name (without extension), not on merged '*_temp.cpp' file
  //
  const std::filesystem::path rawName = a_kernelsJson["MainRawName"].get<std::string>();
  a_kernelsJson["MainISPCFile"] = rawName.filename().u8string() + "_kernels.h";

  // member functions access class data directly, so all member buffers and acceleration structures used by kernels
  // are also placed to global variables of ISPC file; each kernel sets them before work
  //
  std::set<std::string> globalNames;
  nlohmann::json memberBuffers = std::vector<nlohmann::json>();
  nlohmann::json accelStructs  = std::vector<nlohmann::json>();
  for(const auto& kernel : a_kernelsJson["Kernels"])
  {
    for(const auto& arg : kernel["Args"])
    {
      const std::string name = arg["Name"];
      if(!arg["IsMember"].get<bool>() || globalNames.find(name) != globalNames.end())
        continue;
      globalNames.insert(name);
      if(arg["IsAccelStruct"].get<bool>())
        accelStructs.push_back(name);
      else if(arg["IsPointer"].get<bool>())
      {
        nlohmann::json buff;
        buff["Name"] = name;
        buff["Type"] = arg["Type"];
        memberBuffers.push_back(buff);
      }
    }
  }
  a_kernelsJson["MemberBuffersISPC"] = memberBuffers;
  a_kernelsJson["AccelStructsISPC"]  = accelStructs;
  a_kernelsJson["UseAccelStruct"]    = (accelStructs.size() != 0);

  std::filesystem::path folderPath = rawName.parent_path();
  std::filesystem::path incUBOPath = folderPath / "include";
  std::filesystem::create_directory(incUBOPath);

  const std::string templatePath = "templates_ispc/generated.ispc";
  std::filesystem::path outFileName = rawName;
  outFileName.concat("_kernels.ispc");
  std::filesystem::path outCppName = rawName;
  outCppName.concat("_ispc.cpp");

  kslicer::ApplyJsonToTemplate(templatePath, outFileName, a_kernelsJson);
  kslicer::ApplyJsonToTemplate("templates_ispc/ispc_class.cpp", outCppName, a_kernelsJson);

  #ifdef _WIN32
  std::ofstream buildSH(folderPath / "z_build_ispc.bat");
  #else
  std::ofstream buildSH(folderPath / "z_build_ispc.sh");
  #endif
  #ifndef _WIN32
  buildSH << "#!/bin/sh" << std::endl;
  #endif
  std::filesystem::path kernelTarget = rawName;
  kernelTarget.concat("_kernels.o");
  std::filesystem::path kernelHeader = rawName;
  kernelHeader.concat("_kernels.h");
  std::string build = this->BuildCommand(outFileName.u8string()) + " -o " + kernelTarget.u8string() + " -h " + kernelHeader.u8string();
  if(a_kernelsJson["UseAccelStruct"].get<bool>())
  {
    // Embree ISPC headers: 'embree3/*.isph' are stored near 'CrossRT.h'; for Embree 4 set EMBREE4_FLAGS="-I<embree4>/include -DUSE_EMBREE4"
    //
    buildSH << "# for Embree 4 set EMBREE4_FLAGS=\"-I<embree 4 install>/include -DUSE_EMBREE4\"" << std::endl;
    for(auto folder : a_codeInfo->processFolders)
      if(std::filesystem::exists(folder / "CrossRT.h"))
        build += " -I" + folder.u8string();
    build += " $EMBREE4_FLAGS";
  }
  buildSH << build.c_str() << " ";
  for(auto folder : ignoreFolders)
    buildSH << "-I" << folder.c_str() << " ";
  buildSH << std::endl;
  buildSH.close();
}

std::unordered_map<std::string, std::string> ListISPCVectorReplacements()
{
  std::unordered_map<std::string, std::string> m_vecReplacements;
  m_vecReplacements["float2"] = "float2";
  m_vecReplacements["float3"] = "float3";
  m_vecReplacements["float4"] = "float4";
  m_vecReplacements["int2"]   = "int2";
  m_vecReplacements["int3"]   = "int3";
  m_vecReplacements["int4"]   = "int4";
  m_vecReplacements["uint2"]  = "uint2";
  m_vecReplacements["uint3"]  = "uint3";
  m_vecReplacements["uint4"]  = "uint4";
  //m_vecReplacements["float4x4"] = "mat4";
  m_vecReplacements["_Bool"] = "bool";
  return m_vecReplacements;
}

const std::string ConvertVecTypesToISPC(const std::string& a_typeName,
                                        const std::string& a_argName)

{
  static const auto vecTypes = ListISPCVectorReplacements();
  std::string nameToSearch = a_typeName;
  ReplaceFirst(nameToSearch, "const ", "");
  while(nameToSearch[nameToSearch.size()-1] == ' ')
    nameToSearch = nameToSearch.substr(0, nameToSearch.size()-1);
  static const std::unordered_set<std::string> scalarTypes = {"float", "double", "int", "uint", "unsigned", "unsigned int", "bool", "_Bool", "char", "unsigned char", "short", "unsigned short",
                                                               "int8_t", "uint8_t", "int16_t", "uint16_t", "int32_t", "uint32_t", "int64_t", "uint64_t", "size_t"};
  if(scalarTypes.find(nameToSearch) == scalarTypes.end()) // vector types and user structures have their own declarations in ISPC header 
  {
    if (nameToSearch.find("struct") != std::string::npos)
      ReplaceFirst(nameToSearch, "struct ", "");

    if(a_typeName.find("const ") != std::string::npos)
      return "(const ispc::" + nameToSearch + "*)" + a_argName;
    else
      return "(ispc::" + nameToSearch + "*)" + a_argName;
  }
  return a_argName;
}

std::string kslicer::ISPCCompiler::PrintHeaderDecl(const DeclInClass& a_decl, const clang::CompilerInstance& a_compiler, std::shared_ptr<kslicer::FunctionRewriter> a_pRewriter)
{
  std::string typeInCL = a_decl.type;
  std::string result = "";
  switch(a_decl.kind)
  {
    case kslicer::DECL_IN_CLASS::DECL_STRUCT:
    result = kslicer::GetRangeSourceCode(a_decl.srcRange, a_compiler) + ";";
    break;
    case kslicer::DECL_IN_CLASS::DECL_CONSTANT:
    ReplaceFirst(typeInCL, "_Bool", "bool");
    result = typeInCL + " " + a_decl.name + " = " + kslicer::GetRangeSourceCode(a_decl.srcRange, a_compiler) + ";";
    break;
    case kslicer::DECL_IN_CLASS::DECL_TYPEDEF:
    ReplaceFirst(typeInCL, "LiteMath::", "");
    result = "typedef " + typeInCL + " " + a_decl.name + ";";
    break;
    default:
    break;
  };
  return result;
}

std::string kslicer::ISPCCompiler::ReplaceCallFromStdNamespace(const std::string& a_call, const std::string& a_typeName) const
{
  std::string call = a_call;
  ReplaceFirst(call, "std::", "");
  return call;
}
