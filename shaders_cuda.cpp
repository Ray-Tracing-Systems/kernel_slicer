#include "kslicer.h"
#include "template_rendering.h"

#ifdef _WIN32
  #include <sys/types.h>
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void kslicer::CudaRewriter::Init()
{ 
  m_funReplacements.clear();
  m_funReplacements["atomicAdd"] = "InterlockedAdd";
  m_funReplacements["AtomicAdd"] = "InterlockedAdd";
}

std::string kslicer::CudaRewriter::RewriteStdVectorTypeStr(const std::string& a_str) const
{
  const bool isConst  = (a_str.find("const") != std::string::npos);
  std::string copy = a_str;
  ReplaceFirst(copy, "const ", "");
  while(ReplaceFirst(copy, " ", ""));

  auto sFeatures2 = kslicer::GetUsedShaderFeaturesFromTypeName(a_str);
  sFeatures = sFeatures || sFeatures2;

  std::string resStr;
  std::string typeStr = kslicer::CleanTypeName(a_str);

  if(typeStr.size() > 0 && typeStr[typeStr.size()-1] == ' ')
    typeStr = typeStr.substr(0, typeStr.size()-1);

  if(typeStr.size() > 0 && typeStr[0] == ' ')
    typeStr = typeStr.substr(1, typeStr.size()-1);

  auto p = m_typesReplacement.find(typeStr);
  if(p == m_typesReplacement.end())
    resStr = typeStr;
  else
    resStr = p->second;

  if(isConst)
    resStr = std::string("const ") + resStr;

  return resStr;
}

// process arrays: 'float[3] data' --> 'float data[3]' 
std::string kslicer::CudaRewriter::RewriteStdVectorTypeStr(const std::string& a_typeName, std::string& varName) const
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

std::string kslicer::CudaRewriter::RewriteFuncDecl(clang::FunctionDecl* fDecl)
{
  std::string retT   = RewriteStdVectorTypeStr(fDecl->getReturnType().getAsString());
  std::string fname  = fDecl->getNameInfo().getName().getAsString();

  if(m_pCurrFuncInfo != nullptr && m_pCurrFuncInfo->hasPrefix) // alter function name if it has any prefix
    if(fname.find(m_pCurrFuncInfo->prefixName) == std::string::npos)
      fname = m_pCurrFuncInfo->prefixName + "_" + fname;

  std::string result = retT + " " + fname + "(";

  for(uint32_t i=0; i < fDecl->getNumParams(); i++)
  {
    const clang::ParmVarDecl* pParam  = fDecl->getParamDecl(i);
    const clang::QualType typeOfParam =	pParam->getType();
    std::string typeStr = typeOfParam.getAsString();
    if(typeOfParam->isPointerType())
      typeStr += "*";
    result += RewriteStdVectorTypeStr(typeStr) + " " + pParam->getNameAsString();
    if(i!=fDecl->getNumParams()-1)
      result += ", ";
  }

  return result + ") ";
}

bool kslicer::CudaRewriter::NeedsVectorTypeRewrite(const std::string& a_str) // TODO: make this implementation more smart, bad implementation actually!
{
  std::string typeStr = kslicer::CleanTypeName(a_str);
  return (m_typesReplacement.find(typeStr) != m_typesReplacement.end());
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool kslicer::CudaRewriter::VisitFunctionDecl_Impl(clang::FunctionDecl* fDecl)        
{
  return FunctionRewriter2::VisitFunctionDecl_Impl(fDecl); 
}

bool kslicer::CudaRewriter::VisitCXXMethodDecl_Impl(clang::CXXMethodDecl* fDecl)      
{ 
  return true; 
}

bool kslicer::CudaRewriter::VisitMemberExpr_Impl(clang::MemberExpr* expr)             
{
  if(m_kernelMode)
  {
   
  }

  return true; 
}

bool kslicer::CudaRewriter::VisitCXXMemberCallExpr_Impl(clang::CXXMemberCallExpr* f)  { return true; } 
bool kslicer::CudaRewriter::VisitFieldDecl_Impl(clang::FieldDecl* decl)               { return true; }

std::string kslicer::CudaRewriter::VectorTypeContructorReplace(const std::string& fname, const std::string& callText)
{
  return fname + callText;
}

bool kslicer::CudaRewriter::VisitCXXConstructExpr_Impl(clang::CXXConstructExpr* call) 
{ 
  return true; 
} 

bool kslicer::CudaRewriter::VisitCallExpr_Impl(clang::CallExpr* call)                    
{ 
  return true; 
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// unually both for kernel and functions
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// unually both for kernel and functions

bool kslicer::CudaRewriter::VisitUnaryOperator_Impl(clang::UnaryOperator* expr)
{ 
  if(m_kernelMode)
  {
  }
  return true; 
}

bool kslicer::CudaRewriter::VisitCXXOperatorCallExpr_Impl(clang::CXXOperatorCallExpr* expr) 
{ 
  if(m_kernelMode)
  {
    FunctionRewriter2::VisitCXXOperatorCallExpr_Impl(expr);
  }

  return true; 
}

bool kslicer::CudaRewriter::VisitVarDecl_Impl(clang::VarDecl* decl) 
{ 
  if(clang::isa<clang::ParmVarDecl>(decl)) // process else-where (VisitFunctionDecl_Impl)
    return true;

  if(m_kernelMode)
  {
    // ...
  }

  return true; 
}

bool kslicer::CudaRewriter::VisitCStyleCastExpr_Impl(clang::CStyleCastExpr* cast)           
{ 
  if(m_kernelMode)
  {
    // ...
  }

  return true;   
}

bool kslicer::CudaRewriter::VisitImplicitCastExpr_Impl(clang::ImplicitCastExpr* cast) 
{ 
  if(m_kernelMode)
  {
    // ...
  }

  return true; 
}

bool kslicer::CudaRewriter::VisitDeclStmt_Impl(clang::DeclStmt* decl)             
{ 
  if(m_kernelMode)
  {
    // ...
  }

  return true; 
}

bool kslicer::CudaRewriter::VisitFloatingLiteral_Impl(clang::FloatingLiteral* expr) 
{
  return true;
}

bool kslicer::CudaRewriter::VisitArraySubscriptExpr_Impl(clang::ArraySubscriptExpr* arrayExpr) 
{ 
  if(m_kernelMode)
  {
    // ...
  }

  return true; 
}

bool kslicer::CudaRewriter::VisitUnaryExprOrTypeTraitExpr_Impl(clang::UnaryExprOrTypeTraitExpr* szOfExpr) 
{ 
  if(m_kernelMode)
  {
    // ...
  }

  return true; 
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// usually only for kernel
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// usually only for kernel

bool kslicer::CudaRewriter::VisitCompoundAssignOperator_Impl(clang::CompoundAssignOperator* expr) 
{ 
  if(m_kernelMode)
  {
    return FunctionRewriter2::VisitCompoundAssignOperator_Impl(expr);
  }

  return true; 
}

bool kslicer::CudaRewriter::VisitBinaryOperator_Impl(clang::BinaryOperator* expr)
{
  if(m_kernelMode)
  {
    return FunctionRewriter2::VisitBinaryOperator_Impl(expr);
  }

  return true;
}

bool  kslicer::CudaRewriter::VisitDeclRefExpr_Impl(clang::DeclRefExpr* expr) 
{
  if(m_kernelMode)
  { 

  }

  return true;
}

std::string kslicer::CudaRewriter::RecursiveRewrite(const clang::Stmt* expr)
{
  if(expr == nullptr)
    return "";
  
  std::string shallow;
  if(DetectAndRewriteShallowPattern(expr, shallow)) 
    return shallow;
    
  CudaRewriter rvCopy = *this;
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

kslicer::CudaCompiler::CudaCompiler(const std::string& a_prefix) : m_suffix(a_prefix)
{
  m_typesReplacement.clear(); 
}

std::string kslicer::CudaCompiler::LocalIdExpr(uint32_t a_kernelDim, uint32_t a_wgSize[3]) const
{
  if(a_kernelDim == 1)
    return "threadIdx.x";
  else if(a_kernelDim == 2)
  {
    std::stringstream strOut;
    strOut << "threadIdx.x + " << a_wgSize[0] << "*threadIdx.y";
    return strOut.str();
  }
  else if(a_kernelDim == 3)
  {
    std::stringstream strOut;
    strOut << "threadIdx.x + " << a_wgSize[0] << "*threadIdx.y + " << a_wgSize[0]*a_wgSize[1] << "*threadIdx.z";
    return strOut.str();
  }
  else
  {
    std::cout << "  [CudaCompiler::LocalIdExpr]: Error, bad kernelDim = " << a_kernelDim << std::endl;
    return "threadIdx.x";
  }
}

void kslicer::CudaCompiler::GetThreadSizeNames(std::string a_strs[3]) const
{
  a_strs[0] = "iNumElementsX";
  a_strs[1] = "iNumElementsY";
  a_strs[2] = "iNumElementsZ";
}

std::string kslicer::CudaCompiler::GetSubgroupOpCode(const kslicer::KernelInfo::ReductionAccess& a_access) const
{
  std::string res = "WaveActiveSumUnknown"; 
  switch(a_access.type)
  {
    case KernelInfo::REDUCTION_TYPE::ADD_ONE:
    case KernelInfo::REDUCTION_TYPE::ADD:
    res = "WarpReduceSum";
    break;

    case KernelInfo::REDUCTION_TYPE::SUB:
    case KernelInfo::REDUCTION_TYPE::SUB_ONE:
    res = "WarpReduceSum";
    break;

    case KernelInfo::REDUCTION_TYPE::FUNC:
    {
      if(a_access.funcName == "min" || a_access.funcName == "std::min") res = "WarpReduceMin";
      if(a_access.funcName == "max" || a_access.funcName == "std::max") res = "WarpReduceMax";
    }
    break;

    case KernelInfo::REDUCTION_TYPE::MUL:
    res = "WarpReduceMul";
    break;

    default:
    break;
  };
  return res;
}

std::string kslicer::CudaCompiler::GetAtomicImplCode(const kslicer::KernelInfo::ReductionAccess& a_access) const
{
  std::string res = "atomicUnknown";
  switch(a_access.type)
  {
    case KernelInfo::REDUCTION_TYPE::ADD_ONE:
    case KernelInfo::REDUCTION_TYPE::ADD:
    res = "atomicAdd";
    break;

    case KernelInfo::REDUCTION_TYPE::SUB:
    case KernelInfo::REDUCTION_TYPE::SUB_ONE:
    res = "atomicAdd";
    break;

    case KernelInfo::REDUCTION_TYPE::FUNC:
    {
      if(a_access.funcName == "min" || a_access.funcName == "std::min") res = "atomicMin";
      if(a_access.funcName == "max" || a_access.funcName == "std::max") res = "atomicMax";
    }
    break;

    default:
    break;
  };
  return res;
}

std::string kslicer::CudaCompiler::ProcessBufferType(const std::string& a_typeName) const
{
  std::string type = kslicer::CleanTypeName(a_typeName);
  ReplaceFirst(type, "*", "");
  
  if(type[type.size()-1] == ' ')
    type = type.substr(0, type.size()-1);

  return type;
}

std::string kslicer::CudaCompiler::RewritePushBack(const std::string& memberNameA, const std::string& memberNameB, const std::string& newElemValue) const
{
  return std::string("{ uint offset = 0; InterlockedAdd(") + UBOAccess(memberNameB) + ", 1, offset); " + memberNameA + "[offset] = " + newElemValue + ";}";
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::shared_ptr<kslicer::FunctionRewriter> kslicer::CudaCompiler::MakeFuncRewriter(clang::Rewriter &R, const clang::CompilerInstance& a_compiler, 
                                                                                    MainClassInfo* a_codeInfo, kslicer::ShittyFunction a_shit)
{
  auto pFunc = std::make_shared<CudaRewriter>(R, a_compiler, a_codeInfo);
  pFunc->m_shit = a_shit;
  return pFunc;
}

std::shared_ptr<kslicer::KernelRewriter> kslicer::CudaCompiler::MakeKernRewriter(clang::Rewriter &R, const clang::CompilerInstance& a_compiler, 
                                                                                 MainClassInfo* a_codeInfo, kslicer::KernelInfo& a_kernel, const std::string& fakeOffs)
{
  auto pFunc = std::make_shared<CudaRewriter>(R, a_compiler, a_codeInfo);
  pFunc->InitKernelData(a_kernel, fakeOffs);
  return std::make_shared<KernelRewriter2>(R, a_compiler, a_codeInfo, a_kernel, fakeOffs, pFunc);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void kslicer::CudaCompiler::GenerateShaders(nlohmann::json& a_kernelsJson, const MainClassInfo* a_codeInfo, const kslicer::TextGenSettings& a_settings)
{
  
}


std::string kslicer::CudaCompiler::PrintHeaderDecl(const DeclInClass& a_decl, const clang::CompilerInstance& a_compiler, std::shared_ptr<kslicer::FunctionRewriter> a_pRewriter)
{
  std::string typeInCL = a_decl.type;
  ReplaceFirst(typeInCL, "struct ", "");
  ReplaceFirst(typeInCL, "LiteMath::", "");

  std::string originalText = kslicer::GetRangeSourceCode(a_decl.srcRange, a_compiler);

  std::string result = "";
  switch(a_decl.kind)
  {
    case kslicer::DECL_IN_CLASS::DECL_STRUCT:
    result = originalText + ";";
    break;
    case kslicer::DECL_IN_CLASS::DECL_CONSTANT:
    //ReplaceFirst(typeInCL, "unsigned int", "uint");
    //ReplaceFirst(typeInCL, "unsigned", "uint");
    //ReplaceFirst(typeInCL, "_Bool", "bool");
    for(const auto& pair : m_typesReplacement)
      ReplaceFirst(typeInCL, pair.first, pair.second);
    if(originalText == "")
      originalText = a_decl.lostValue;
    result = "static " + typeInCL + " " + a_decl.name + " = " + originalText + ";";
    //if(a_decl.name.find("CRT_GEOM_MASK_AABB_BIT") != std::string::npos)
    //{
    //  int a = 2;
    //}
    break;
    case kslicer::DECL_IN_CLASS::DECL_TYPEDEF:
    for(const auto& pair : m_typesReplacement)
      ReplaceFirst(typeInCL, pair.first, pair.second);
    result = "typealias " + a_decl.name + " = " + typeInCL + ";";
    break;
    default:
    break;
  };
  return result;
}

std::string kslicer::CudaCompiler::RTVGetFakeOffsetExpression(const kslicer::KernelInfo& a_funcInfo, const std::vector<kslicer::ArgFinal>& threadIds)
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
