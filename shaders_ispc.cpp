#include "kslicer.h"
#include "template_rendering.h"
#include "clang/Lex/Lexer.h"

#ifdef _WIN32
  #include <sys/types.h>
#endif

void kslicer::ISPCRewriter::Init()
{ 
  m_removeStdQualifier = true;
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
  ReplaceFirst(typeStr, "class ", "");                        // 'class EyeRayData': ISPC has only structures
  if(typeStr == "_Bool")
    typeStr = "bool";
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

namespace
{
  /// names which are keywords or types in ISPC but not in C++: local variables and parameters with them get '_' at the end
  std::string ISPCSafeName(const std::string& a_name)
  {
    static const std::set<std::string> keywords = {"uniform", "varying", "foreach", "foreach_tiled", "foreach_active", "foreach_unique",
      "task", "launch", "sync", "export", "cif", "cdo", "cfor", "cwhile", "soa", "unmasked", "int8", "int16", "int32", "int64",
      "uint8", "uint16", "uint32", "uint64", "float16", "half"};
    return (keywords.find(a_name) != keywords.end()) ? a_name + "_" : a_name;
  }

  bool IsLocalOrParam(const clang::ValueDecl* a_decl)
  {
    auto pVar = clang::dyn_cast_or_null<clang::VarDecl>(a_decl);
    return pVar != nullptr && (pVar->isLocalVarDecl() || clang::isa<clang::ParmVarDecl>(pVar));
  }
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
    if(clang::isa<clang::DecayedType>(typeOfParam))                               // 'float a_data[3]' is kept as is
    {
      std::string text = kslicer::GetRangeSourceCode(pParam->getSourceRange(), m_compiler);
      while(ReplaceFirst(text, "LiteMath::", ""));
      result += text;
    }
    else if(typeOfParam->isPointerType() && IsGlobalPointerParam(pParam))        // 'const float* a_data' ==> 'uniform const float a_data[]', pointer to buffer
    {
      std::string elemType = RewriteStdVectorTypeStr(typeOfParam->getPointeeType().getAsString());
      result += "uniform " + elemType + " " + ISPCSafeName(pParam->getNameAsString()) + "[]";
    }
    else if(typeOfParam->isPointerType() && (IsReferenceParam(pParam) || typeOfParam->getPointeeType()->isRecordType())) // 'Varyings* a_out' ==> 'Varyings& a_out'
      result += RewriteStdVectorTypeStr(typeOfParam->getPointeeType().getAsString()) + "& " + ISPCSafeName(pParam->getNameAsString());
    else
      result += RewriteStdVectorTypeStr(typeStr) + " " + ISPCSafeName(pParam->getNameAsString());
    if(i!=fDecl->getNumParams()-1)
      result += ", ";
  }

  return result + ") ";
}

bool kslicer::ISPCRewriter::IsGlobalPointerParam(const clang::ParmVarDecl* a_param) const
{
  auto pFunc = clang::dyn_cast<clang::FunctionDecl>(a_param->getDeclContext());
  if(pFunc == nullptr || pFunc->getNameAsString() != m_shit.originalName)
    return false;
  for(const auto& p : m_shit.pointers)
    if(p.formal == a_param->getNameAsString())
      return true;
  return false;
}

bool kslicer::ISPCRewriter::IsReferenceParam(const clang::ValueDecl* a_decl) const
{
  auto pParam = clang::dyn_cast_or_null<clang::ParmVarDecl>(a_decl);
  if(pParam == nullptr || !pParam->getType()->isPointerType() || clang::isa<clang::DecayedType>(pParam->getType()))
    return false;
  auto pFunc = clang::dyn_cast<clang::FunctionDecl>(pParam->getDeclContext());
  if(pFunc == nullptr || IsGlobalPointerParam(pParam))
    return false;
  auto pKernel = m_codeInfo->kernels.find(pFunc->getNameAsString()); // kernels get buffers, RTV kernels get payload by reference
  if(pKernel != m_codeInfo->kernels.end())
  {
    if(pKernel->second.pattern != kslicer::PATTERN_TP::PATTERN_RTV)
      return false;
    if(m_codeInfo->megakernelRTV)
      return true;
    for(const auto& arg : pKernel->second.args) // wavefront: payload, other pointers are buffers
      if(arg.name == pParam->getNameAsString())
        return arg.needFakeOffset;
    return false;
  }
  const std::string fileName = std::string(m_compiler.getSourceManager().getFilename(pFunc->getSourceRange().getBegin()));
  return !m_codeInfo->IsInExcludedFolder(fileName);
}

bool kslicer::ISPCRewriter::IsAddressOfLocalForCall(const clang::UnaryOperator* a_expr)
{
  if(a_expr->getOpcode() != clang::UO_AddrOf)
    return false;

  // '&x', '&x.y', '&a[i]' of local variable or parameter passed by value or by reference
  //
  const clang::Expr* pBase = a_expr->getSubExpr()->IgnoreParenImpCasts();
  while(true)
  {
    if(auto pMember = clang::dyn_cast<clang::MemberExpr>(pBase); pMember != nullptr && !pMember->isArrow())
      pBase = pMember->getBase()->IgnoreParenImpCasts();
    else if(auto pSubscr = clang::dyn_cast<clang::ArraySubscriptExpr>(pBase); pSubscr != nullptr && pSubscr->getBase()->IgnoreParenImpCasts()->getType()->isArrayType())
      pBase = pSubscr->getBase()->IgnoreParenImpCasts();
    else
      break;
  }
  auto pRef = clang::dyn_cast<clang::DeclRefExpr>(pBase);
  if(pRef == nullptr)
    return false;
  auto pVar = clang::dyn_cast<clang::VarDecl>(pRef->getDecl());
  if(pVar == nullptr || !pVar->hasLocalStorage() || (pVar->getType()->isPointerType() && !IsReferenceParam(pVar)))
    return false;

  // it must be argument of user function or RTV kernel of megakernel
  //
  auto parents = m_compiler.getASTContext().getParents(*a_expr);
  while(parents.size() != 0 && (parents[0].get<clang::ImplicitCastExpr>() != nullptr || parents[0].get<clang::ParenExpr>() != nullptr))
    parents = m_compiler.getASTContext().getParents(parents[0]);
  if(parents.size() == 0)
    return false;
  auto pCall = parents[0].get<clang::CallExpr>();
  if(pCall == nullptr || pCall->getDirectCallee() == nullptr)
    return false;
  const clang::FunctionDecl* pCallee = pCall->getDirectCallee();
  for(unsigned i = 0; i < pCall->getNumArgs() && i < pCallee->getNumParams(); i++)
  {
    if(pCall->getArg(i)->IgnoreParenImpCasts() == a_expr)
      return IsReferenceParam(pCallee->getParamDecl(i));
  }
  return false;
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
  if(clang::isa<clang::CXXMethodDecl>(fDecl)) // methods are processed separately
    return true;

  if(fDecl->hasBody() && WasNotRewrittenYet(fDecl->getBody()))  // bodies of local functions are rewritten: 'float(x)' ==> '(float)(x)', 'std::pow' ==> 'pow', ...
  {
    RewrittenFunction done = RewriteFunction(fDecl);
    done.funDecl = "static inline " + done.funDecl; // pointers to local variables are references, see IsReferenceParam
    while(ReplaceFirst(done.funDecl, "LiteMath::", "")); // ISPC has no namespaces
    const auto hash = GetHashOfSourceRange(fDecl->getBody()->getSourceRange());
    if(m_codeInfo->m_functionsDone.find(hash) == m_codeInfo->m_functionsDone.end())
      m_codeInfo->m_functionsDone[hash] = done;
    MarkRewritten(fDecl->getBody());
  }
  return true; 
}

bool kslicer::ISPCRewriter::VisitCXXMethodDecl_Impl(clang::CXXMethodDecl* fDecl)      
{ 
  return true; 
}

bool kslicer::ISPCRewriter::VisitMemberExpr_Impl(clang::MemberExpr* expr)             
{
  // 'a_x->y' ==> 'a_x.y' for reference parameter (see IsReferenceParam)
  //
  if(expr->isArrow() && WasNotRewrittenYet(expr))
  {
    auto pRef = clang::dyn_cast<clang::DeclRefExpr>(expr->getBase()->IgnoreParenImpCasts());
    if(pRef != nullptr && IsReferenceParam(pRef->getDecl()))
    {
      const std::string text = kslicer::GetRangeSourceCode(expr->getSourceRange(), m_compiler); // member of anonymous struct of LiteMath vector has no name
      const auto posArrow    = text.find("->");
      if(posArrow != std::string::npos)
        ReplaceText(expr->getSourceRange(), RecursiveRewrite(expr->getBase()) + "." + text.substr(posArrow + 2));
      MarkRewritten(expr);
      return true;
    }
  }

  if(m_kernelMode)
  {
    std::string rewrittenText;
    if(NeedToRewriteMemberExpr(expr, rewrittenText))
    {
      ReplaceText(expr->getSourceRange(), rewrittenText);
      MarkRewritten(expr);
    }
  }

  return true; 
}

/// name of Image2D member in expression 'm_img' or 'this->m_img', empty string otherwise
static std::string ImageMemberName(const clang::Expr* a_expr)
{
  const clang::Expr* expr = kslicer::RemoveImplicitCast(a_expr);
  if(expr == nullptr || expr->getType()->isPointerType() || !kslicer::IsTexture(expr->getType()))
    return "";
  if(auto pMember = clang::dyn_cast<clang::MemberExpr>(expr))
    return pMember->getMemberDecl()->getNameAsString();
  return "";
}

/// 'm_textures[id]' of 'm_textures[id]->sample(uv)' for vector of 'std::shared_ptr<ICombinedImageSampler>' or 'const ICombinedImageSampler*',
/// returns name of member ('m_textures') and index expression
static std::string TextureArrayElement(const clang::Expr* a_object, const clang::Expr** a_pIndex)
{
  const clang::Expr* expr = a_object->IgnoreParenImpCasts();
  if(auto pArrow = clang::dyn_cast<clang::CXXOperatorCallExpr>(expr); pArrow != nullptr && pArrow->getOperator() == clang::OO_Arrow) // 'shared_ptr::operator->'
    expr = pArrow->getArg(0)->IgnoreParenImpCasts();
  auto pSubscr = clang::dyn_cast<clang::CXXOperatorCallExpr>(expr);
  if(pSubscr == nullptr || pSubscr->getOperator() != clang::OO_Subscript)
    return "";
  auto pMember = clang::dyn_cast<clang::MemberExpr>(pSubscr->getArg(0)->IgnoreParenImpCasts());
  if(pMember == nullptr)
    return "";
  (*a_pIndex) = pSubscr->getArg(1);
  return pMember->getMemberDecl()->getNameAsString();
}

bool kslicer::ISPCRewriter::VisitCXXMemberCallExpr_Impl(clang::CXXMemberCallExpr* f)
{
  // 'm_textures[id]->sample(uv)' ==> 'LiteTexArraySample(m_textures, id, uv)', see LiteMathAux/LiteImageISPC.h
  //
  if(f->getMethodDecl() != nullptr && f->getMethodDecl()->getNameAsString() == "sample" && f->getNumArgs() == 1 && WasNotRewrittenYet(f) &&
     f->getMethodDecl()->getParent()->getNameAsString() == "ICombinedImageSampler")
  {
    const clang::Expr* pIndex = nullptr;
    const std::string arrayName = TextureArrayElement(f->getImplicitObjectArgument(), &pIndex);
    auto pMember = m_codeInfo->allDataMembers.find(arrayName);
    if(arrayName != "" && pMember != m_codeInfo->allDataMembers.end())
    {
      ReplaceText(f->getSourceRange(), "LiteTexArraySample(" + arrayName + ", " + RecursiveRewrite(pIndex) + ", " + RecursiveRewrite(f->getArg(0)) + ")");
      MarkRewritten(f);
      return true;
    }
  }

  // 'm_img.sample(m_sampler, uv)' ==> 'LiteImageSample_<T>(m_img, m_img_info, m_sampler, uv)', see LiteMathAux/LiteImageISPC.h
  //
  if(f->getMethodDecl() != nullptr && f->getMethodDecl()->getNameAsString() == "sample" && f->getNumArgs() == 2 && WasNotRewrittenYet(f))
  {
    const std::string imageName = ImageMemberName(f->getImplicitObjectArgument());
    auto pSampler = clang::dyn_cast<clang::MemberExpr>(kslicer::RemoveImplicitCast(f->getArg(0)));
    if(imageName != "" && pSampler != nullptr)
    {
      auto pMember = m_codeInfo->allDataMembers.find(imageName);
      const std::string elemType = (pMember != m_codeInfo->allDataMembers.end()) ? kslicer::ImageElemTypeISPC(RewriteStdVectorTypeStr(pMember->second.containerDataType)) : std::string("float4");
      const std::string uv       = RecursiveRewrite(f->getArg(1));
      ReplaceText(f->getSourceRange(), "LiteImageSample_" + elemType + "(" + imageName + ", " + imageName + "_info, " + pSampler->getMemberDecl()->getNameAsString() + ", " + uv + ")");
      MarkRewritten(f);
      return true;
    }
  }

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
        ReplaceText(f->getSourceRange(), m_codeInfo->pShaderCC->UBOAccess(memberNameB) );
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
        //  ReplaceText(f->getSourceRange(), m_codeInfo->pShaderCC->UBOAccess(memberNameB) );
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
        ReplaceText(f->getSourceRange(), resulingText);
        MarkRewritten(f);
      }
      else if(fname == "data")
      {
        ReplaceText(f->getSourceRange(), memberNameA);
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
      ReplaceText(f->getSourceRange(), resCallText);
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
namespace
{
  /// 'float4', 'int4', 'uint4' for a_size == '4' (LiteMath vector of the same number of components, another type of elements)
  bool IsVectorOfSize(clang::QualType a_type, char a_size)
  {
    const clang::RecordDecl* pRecord = a_type.getNonReferenceType().getCanonicalType()->getAsRecordDecl();
    if(pRecord == nullptr)
      return false;
    const std::string name = pRecord->getNameAsString();
    return (name == std::string("float") + a_size || name == std::string("int") + a_size || name == std::string("uint") + a_size);
  }
}

bool kslicer::ISPCRewriter::VisitCXXConstructExpr_Impl(clang::CXXConstructExpr* call) 
{ 
  const clang::CXXConstructorDecl* ctorDecl = call->getConstructor();
  if(ctorDecl == nullptr || ctorDecl->isCopyOrMoveConstructor() || call->getNumArgs() == 0 || !WasNotRewrittenYet(call))
    return true;

  const std::string typeName = ctorDecl->getNameInfo().getName().getAsString();
  static const kslicer::uset<std::string> vecTypes = {"float2", "float3", "float4", "int2", "int3", "int4", "uint2", "uint3", "uint4", "complex"};
  if(vecTypes.find(typeName) == vecTypes.end())
    return true;

  const uint32_t vecSize = (typeName == "complex") ? call->getNumArgs() : uint32_t(typeName[typeName.size()-1] - '0'); // 'complex(re)', 'complex(re, im)'
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
  else if(call->getNumArgs() == 1 && IsVectorOfSize(call->getArg(0)->getType(), typeName.back())) // 'int4(float4)' ==> '(int4)(...)', element by element
    text = "(" + typeName + ")(" + RecursiveRewrite(call->getArg(0)) + ")";
  else
    return true;

  // 'float3 v(a, b, c)' ==> 'float3 v = make_float3(a, b, c)': the range of constructor call starts at the name of variable
  //
  auto parents = m_compiler.getASTContext().getParents(*call);
  if(parents.size() != 0)
  {
    auto pVar = parents[0].get<clang::VarDecl>();
    if(pVar != nullptr && pVar->getInit() != nullptr && pVar->getInit()->IgnoreImplicit() == call && call->getBeginLoc() == pVar->getLocation())
      text = pVar->getNameAsString() + " = " + text;
  }

  ReplaceText(call->getSourceRange(), text);
  MarkRewritten(call);
  return true; 
} 

// 'float(x)' ==> '(float)(x)'
//
bool kslicer::ISPCRewriter::VisitCXXFunctionalCastExpr_Impl(clang::CXXFunctionalCastExpr* cast)
{
  if(!WasNotRewrittenYet(cast))
    return true;

  // 'half(x)': LiteMath::half is a class (half_float::half), in ISPC it is 'float16'
  //
  const clang::RecordDecl* pRecord = cast->getType()->getAsRecordDecl();
  if(pRecord != nullptr && pRecord->getNameAsString() == "half")
  {
    const clang::Expr* sub = cast->getSubExpr()->IgnoreImplicit();
    if(auto pCtor = clang::dyn_cast<clang::CXXConstructExpr>(sub); pCtor != nullptr && pCtor->getNumArgs() == 1)
      sub = pCtor->getArg(0);
    ReplaceText(cast->getSourceRange(), "(half)(" + RecursiveRewrite(sub) + ")");
    MarkRewritten(cast);
    return true;
  }

  if(!cast->getType()->isBuiltinType())
    return true;
  const std::string typeName = RewriteStdVectorTypeStr(cast->getTypeAsWritten().getAsString());
  ReplaceText(cast->getSourceRange(), "(" + typeName + ")(" + RecursiveRewrite(cast->getSubExpr()) + ")");
  MarkRewritten(cast);
  return true;
}

bool kslicer::ISPCRewriter::VisitCallExpr_Impl(clang::CallExpr* call)                    
{ 
  // 'InterlockedAdd(mem, value[, res])' ==> '[res = ]atomic_add_global(&(mem), value)'; for arrays of the block of block expansion kernel
  // 'atomic_add_local': a block is processed by one task, other tasks never access its arrays
  //
  const clang::FunctionDecl* pAtomic = call->getDirectCallee();
  if(pAtomic != nullptr && pAtomic->getNameAsString() == "InterlockedAdd" && (call->getNumArgs() == 2 || call->getNumArgs() == 3) && WasNotRewrittenYet(call))
  {
    const clang::Expr* base = call->getArg(0)->IgnoreParenImpCasts();
    while(auto pSub = clang::dyn_cast<clang::ArraySubscriptExpr>(base))
      base = pSub->getBase()->IgnoreParenImpCasts();
    auto pRef = clang::dyn_cast<clang::DeclRefExpr>(base);
    auto pVar = (pRef == nullptr) ? nullptr : clang::dyn_cast<clang::VarDecl>(pRef->getDecl());
    const bool local = m_kernelMode && m_pCurrKernel != nullptr && m_pCurrKernel->be.enabled && pVar != nullptr && m_pCurrKernel->be.sharedVars.contains(pVar);

    std::string text = std::string(local ? "atomic_add_local" : "atomic_add_global") + "(&(" + RecursiveRewrite(call->getArg(0)) + "), " + RecursiveRewrite(call->getArg(1)) + ")";
    if(call->getNumArgs() == 3)
      text = RecursiveRewrite(call->getArg(2)) + " = " + text;
    ReplaceText(call->getSourceRange(), text);
    MarkRewritten(call);
    return true;
  }

  if(m_kernelMode && WasNotRewrittenYet(call) && !clang::isa<clang::CXXOperatorCallExpr>(call)) // 'a*b' of LiteMath types is an operator call
  {
    clang::FunctionDecl* fDecl = call->getDirectCallee();
    if(fDecl == nullptr)
      return true;

    // buffers passed to functions: such pointer parameters stay pointers (see IsGlobalPointerParam), the others become references
    //
    {
      std::vector<kslicer::ArgMatch> usedArgMatches = kslicer::MatchCallArgsForKernel(call, (*m_pCurrKernel), m_compiler);
      std::vector<kslicer::ArgMatch> globalPointers;
      for(const auto& x : usedArgMatches)
        if(x.isPointer && !NameNeedsFakeOffset(x.actual))
          globalPointers.push_back(x);
      if(globalPointers.size() > 0)
      {
        kslicer::ShittyFunction func;
        func.pointers     = globalPointers;
        func.originalName = fDecl->getNameInfo().getName().getAsString();
        m_pCurrKernel->shittyFunctions.push_back(func);
      }
    }

    // 'Base::Func(...)' of composition; namespaces 'std::', 'LiteMath::' are removed in VisitDeclRefExpr_Impl
    //
    auto pCallee = clang::dyn_cast<clang::DeclRefExpr>(call->getCallee()->IgnoreParenImpCasts());
    if(pCallee == nullptr || !pCallee->hasQualifier())
      return true;
#ifdef KSLICER_USE_LLVM22
    if(pCallee->getQualifier().getKind() == clang::NestedNameSpecifier::Kind::Namespace)
#else
    if(pCallee->getQualifier()->getAsNamespace() != nullptr)
#endif
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
      ReplaceText(call->getSourceRange(), lastRewrittenText);
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
        //ReplaceText(expr->getSourceRange(), leftStr2 + "Shared[" + localIdStr + "]++");
        //MarkRewritten(expr);
      }
    }

    // detect " *something and &something"
    //
    const std::string exprInside = RecursiveRewrite(subExpr);

    if(op == "*" && !expr->canOverflow() && CheckIfExprHasArgumentThatNeedFakeOffset(exprInside) && WasNotRewrittenYet(expr)) 
    {
      if(m_codeInfo->megakernelRTV || m_fakeOffsetExp == "")
        ReplaceText(expr->getSourceRange(), exprInside);
      else
        ReplaceText(expr->getSourceRange(), exprInside + "[" + m_fakeOffsetExp + "]");
      MarkRewritten(expr);
    }
  }

  // '*a_x' ==> 'a_x' for reference parameter, 'f(&local)' ==> 'f(local)' (see IsReferenceParam)
  //
  if(WasNotRewrittenYet(expr))
  {
    const clang::Expr* subExpr = expr->getSubExpr()->IgnoreParenImpCasts();
    auto pRef = clang::dyn_cast<clang::DeclRefExpr>(subExpr);
    const bool derefOfRef = (expr->getOpcode() == clang::UO_Deref) && pRef != nullptr && IsReferenceParam(pRef->getDecl());
    if(derefOfRef || IsAddressOfLocalForCall(expr))
    {
      ReplaceText(expr->getSourceRange(), RecursiveRewrite(expr->getSubExpr()));
      MarkRewritten(expr);
    }
  }

  return true;
}

bool kslicer::ISPCRewriter::VisitCXXOperatorCallExpr_Impl(clang::CXXOperatorCallExpr* expr) 
{ 
  // 'm_img[int2(x,y)]' ==> 'm_img[LiteImageOffset(m_img_info, make_int2(x,y))]', see LiteMathAux/LiteImageISPC.h
  //
  if(expr->getOperator() == clang::OO_Subscript && expr->getNumArgs() == 2 && WasNotRewrittenYet(expr))
  {
    const std::string imageName = ImageMemberName(expr->getArg(0));
    if(imageName != "")
    {
      const std::string coord = RecursiveRewrite(expr->getArg(1));
      ReplaceText(expr->getSourceRange(), imageName + "[LiteImageOffset(" + imageName + "_info, " + coord + ")]");
      MarkRewritten(expr);
      return true;
    }
  }

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
  if(decl->isLocalVarDecl() && ISPCSafeName(decl->getNameAsString()) != decl->getNameAsString()) // 'int half = n/2' ==> 'int half_ = n/2'
    m_rewriter.ReplaceText(decl->getLocation(), decl->getNameAsString().size(), ISPCSafeName(decl->getNameAsString()));
  if(decl->isLocalVarDecl() && WasNotRewrittenYet(decl->getInit()))
    RemoveLiteNamespaceQualifier(decl); // 'const LiteMath::float4x4 m' ==> 'const float4x4 m'

  // block expansion: variables of the block (shared arrays and variables of single thread code) are 'uniform',
  // as well as local variables of uniform overloads of functions called from this code
  //
  const bool uniformInKernel = m_kernelMode && m_pCurrKernel != nullptr && m_pCurrKernel->be.enabled && m_pCurrKernel->be.uniformVars.contains(decl);
  if(decl->isLocalVarDecl() && (uniformInKernel || m_uniformFunction))
  {
    const unsigned loc = decl->getBeginLoc().getRawEncoding(); // 'int a, b;': both declarations start at 'int'
    if(m_uniformDeclLocs->insert(loc).second)
      m_rewriter.InsertTextBefore(decl->getBeginLoc(), "uniform ");
  }

  // 'auto x = f();' ==> 'MatIdWeightPair x = f();': ISPC has no 'auto'
  //
  if(decl->isLocalVarDecl() && decl->getTypeSourceInfo() != nullptr && decl->getType()->getContainedAutoType() != nullptr)
  {
    const clang::AutoType* pAuto = decl->getType()->getContainedAutoType();
    auto autoLoc = decl->getTypeSourceInfo()->getTypeLoc().getContainedAutoTypeLoc();
    if(pAuto->isDeduced() && !autoLoc.isNull())
    {
      const unsigned loc = autoLoc.getBeginLoc().getRawEncoding(); // 'auto a = x, b = y;': both declarations have the same 'auto'
      if(m_uniformDeclLocs->insert(loc).second)
        m_rewriter.ReplaceText(autoLoc.getSourceRange(), RewriteStdVectorTypeStr(kslicer::CleanTypeName(pAuto->getDeducedType().getAsString())));
    }
  }

  // ISPC needs a literal for array size, 'float a[TILE*TILE]' ==> 'float a[256]'; size depending on template parameter is a macro, see templates
  //
  if(decl->isLocalVarDecl() && decl->getTypeSourceInfo() != nullptr)
  {
    auto arrayLoc = decl->getTypeSourceInfo()->getTypeLoc().getAs<clang::ConstantArrayTypeLoc>();
    auto pArray   = clang::dyn_cast<clang::ConstantArrayType>(decl->getType().getTypePtr());
    const clang::Expr* sizeExpr = arrayLoc.isNull() ? nullptr : arrayLoc.getSizeExpr();
    if(pArray != nullptr && sizeExpr != nullptr && !clang::isa<clang::IntegerLiteral>(sizeExpr->IgnoreParenImpCasts()) && WasNotRewrittenYet(sizeExpr))
    {
      ReplaceText(sizeExpr->getSourceRange(), std::to_string(pArray->getSize().getZExtValue()));
      MarkRewritten(sizeExpr);
    }
  }

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

namespace
{
  bool ReferencesDecl(const clang::Stmt* a_stmt, const clang::ValueDecl* a_decl)
  {
    if(auto pRef = clang::dyn_cast<clang::DeclRefExpr>(a_stmt); pRef != nullptr && pRef->getDecl() == a_decl)
      return true;
    for(const clang::Stmt* child : a_stmt->children())
      if(child != nullptr && ReferencesDecl(child, a_decl))
        return true;
    return false;
  }
}

bool kslicer::ISPCRewriter::VisitDeclStmt_Impl(clang::DeclStmt* decl)
{
  // 'float a = x, b = a*y;': in ISPC 'a' is not visible in the initializer of 'b' of the same declaration,
  // so it is split to 'float a = x; float b = a*y;' (not in 'for(int i = 0, j = i; ...)')
  //
  if(decl->isSingleDecl())
    return true;
  for(const auto& parent : m_compiler.getASTContext().getParents(*decl))
    if(parent.get<clang::ForStmt>() != nullptr)
      return true;

  std::vector<const clang::VarDecl*> vars;
  for(const clang::Decl* d : decl->decls())
  {
    auto pVar = clang::dyn_cast<clang::VarDecl>(d);
    if(pVar == nullptr)
      return true;
    vars.push_back(pVar);
  }
  bool usesPrevious = false;
  for(size_t i = 1; i < vars.size() && !usesPrevious; i++)
    for(size_t j = 0; j < i && !usesPrevious && vars[i]->getInit() != nullptr; j++)
      usesPrevious = ReferencesDecl(vars[i]->getInit(), vars[j]);
  if(!usesPrevious)
    return true;

  const auto& sm = m_compiler.getSourceManager();
  const auto& lo = m_compiler.getLangOpts();
  const std::string typeText = kslicer::GetRangeSourceCode(clang::SourceRange(decl->getBeginLoc(), vars[0]->getTypeSpecEndLoc()), m_compiler);
  for(size_t i = 1; i < vars.size(); i++)
  {
    auto comma = clang::Lexer::findNextToken(vars[i-1]->getEndLoc(), sm, lo);
    if(!comma || comma->isNot(clang::tok::comma))
      return true;
    m_rewriter.ReplaceText(comma->getLocation(), 1, "; " + typeText + " ");
  }
  return true;
}

bool kslicer::ISPCRewriter::VisitFloatingLiteral_Impl(clang::FloatingLiteral* expr) 
{
  return true;
}

namespace
{
  bool HasSubtraction(const clang::Stmt* a_expr)
  {
    if(auto pBin = clang::dyn_cast<clang::BinaryOperator>(a_expr); pBin != nullptr && pBin->getOpcode() == clang::BO_Sub)
      return true;
    for(const clang::Stmt* child : a_expr->children())
      if(child != nullptr && HasSubtraction(child))
        return true;
    return false;
  }
}

bool kslicer::ISPCRewriter::VisitArraySubscriptExpr_Impl(clang::ArraySubscriptExpr* arrayExpr) 
{ 
  // block expansion, local arrays: ISPC evaluates both sides of '?:' (and both branches of 'if') with varying condition under a mask,
  // but the address of a vector load is computed for masked lanes too, so 'c ? a[i - 1] : 0' with unsigned 'i == 0' crashes;
  // with a signed index such address is only a bit before the array
  //
  if(m_kernelMode && m_pCurrKernel != nullptr && m_pCurrKernel->be.enabled)
  {
    auto pRef = clang::dyn_cast<clang::DeclRefExpr>(arrayExpr->getBase()->IgnoreParenImpCasts());
    auto pVar = (pRef == nullptr) ? nullptr : clang::dyn_cast<clang::VarDecl>(pRef->getDecl());
    const clang::Expr* idx = arrayExpr->getIdx();
    if(pVar != nullptr && pVar->hasLocalStorage() && pVar->getType()->isArrayType() &&
       idx->getType()->isUnsignedIntegerType() && HasSubtraction(idx) && WasNotRewrittenYet(idx))
    {
      ReplaceText(idx->getSourceRange(), "(int)(" + RecursiveRewrite(idx) + ")");
      MarkRewritten(idx);
    }
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
  RemoveLiteNamespaceQualifier(expr); // ISPC has no namespaces: 'std::pow' and 'LiteMath::dot' are 'pow' and 'dot'

  if(IsLocalOrParam(expr->getDecl()) && WasNotRewrittenYet(expr))
  {
    const std::string name = expr->getDecl()->getNameAsString();
    if(ISPCSafeName(name) != name)
      m_rewriter.ReplaceText(expr->getLocation(), name.size(), ISPCSafeName(name));
  }

  // C math functions: 'fabs(x)' ==> 'abs(x)', 'sqrtf(x)' ==> 'sqrt(x)', ...
  //
  static const kslicer::umap<std::string, std::string> cMathFunctions = { {"fabs", "abs"}, {"fabsf", "abs"}, {"fmin", "min"}, {"fminf", "min"},
    {"fmax", "max"}, {"fmaxf", "max"}, {"sqrtf", "sqrt"}, {"powf", "pow"}, {"expf", "exp"}, {"logf", "log"}, {"sinf", "sin"}, {"cosf", "cos"},
    {"tanf", "tan"}, {"asinf", "asin"}, {"acosf", "acos"}, {"atanf", "atan"}, {"atan2f", "atan2"}, {"floorf", "floor"}, {"ceilf", "ceil"} };
  if(auto pFunc = clang::dyn_cast<clang::FunctionDecl>(expr->getDecl()); pFunc != nullptr && !clang::isa<clang::CXXMethodDecl>(pFunc))
  {
    auto pReplace = cMathFunctions.find(pFunc->getNameAsString());
    if(pReplace != cMathFunctions.end() && !IsUserFunction(pFunc))
      m_rewriter.ReplaceText(clang::SourceRange(expr->getLocation(), expr->getLocation()), pReplace->second);
  }
  if(m_kernelMode)
  { 

  }

  return true;
}

std::string kslicer::ISPCRewriter::RecursiveRewrite(const clang::Stmt* expr)
{
  if(expr == nullptr)
    return "";
  if(clang::isa<clang::CXXDefaultArgExpr>(expr)) // default argument of function call: its value
    return RecursiveRewrite(clang::dyn_cast<clang::CXXDefaultArgExpr>(expr)->getExpr());
  
  std::string shallow;
  if(DetectAndRewriteShallowPattern(expr, shallow)) 
    return shallow;
    
  ISPCRewriter rvCopy = *this;
  rvCopy.TraverseStmt(const_cast<clang::Stmt*>(expr));

  return kslicer::GetRewrittenText(m_rewriter, expr->getSourceRange());
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


// block expansion: 'for(T i = a; i < b; i++) [[parallel]] body' ==> 'foreach(i = a ... b) body';
// other conditions with step 1 ==> 'for(T i = a + programIndex; cond; i += programCount) body'
//
std::string kslicer::ISPCCompiler::RewriteBEParallelFor(const clang::ForStmt* forExpr, std::shared_ptr<KernelRewriter> pRewriter, const KernelInfo& a_kernel)
{
  const clang::VarDecl* pVar  = nullptr;
  if(auto pDeclStmt = clang::dyn_cast_or_null<clang::DeclStmt>(forExpr->getInit()))
    if(pDeclStmt->isSingleDecl())
      pVar = clang::dyn_cast<clang::VarDecl>(pDeclStmt->getSingleDecl());

  auto IsLoopVar = [pVar](const clang::Expr* a_expr)
  {
    auto pRef = clang::dyn_cast_or_null<clang::DeclRefExpr>(a_expr == nullptr ? nullptr : a_expr->IgnoreParenImpCasts());
    return pRef != nullptr && pRef->getDecl() == pVar;
  };

  bool stepIsOne = false;
  const clang::Expr* inc = forExpr->getInc();
  if(auto pUnary = clang::dyn_cast_or_null<clang::UnaryOperator>(inc))
    stepIsOne = pUnary->isIncrementOp() && IsLoopVar(pUnary->getSubExpr());
  else if(auto pAssign = clang::dyn_cast_or_null<clang::CompoundAssignOperator>(inc))
  {
    auto pOne = clang::dyn_cast<clang::IntegerLiteral>(pAssign->getRHS()->IgnoreParenImpCasts());
    stepIsOne = pAssign->getOpcode() == clang::BO_AddAssign && IsLoopVar(pAssign->getLHS()) && pOne != nullptr && pOne->getValue() == 1;
  }

  const clang::CompilerInstance& compiler = pRewriter->GetCompiler();
  if(pVar == nullptr || pVar->getInit() == nullptr || forExpr->getCond() == nullptr || !stepIsOne)
  {
    kslicer::PrintError("parallel loop of block expansion kernel must have the form 'for(T i = start; cond; i++)'", forExpr->getSourceRange(), compiler.getSourceManager());
    return pRewriter->RecursiveRewrite(forExpr);
  }

  std::string typeName = pVar->getType().getUnqualifiedType().getAsString();
  ReplaceFirst(typeName, "LiteMath::", "");
  const std::string name  = pVar->getNameAsString();
  const std::string start = pRewriter->RecursiveRewrite(pVar->getInit());
  const clang::Stmt* body = forExpr->getBody();
  std::string bodyText    = pRewriter->RecursiveRewrite(body);
  const unsigned column   = compiler.getSourceManager().getSpellingColumnNumber(forExpr->getBeginLoc()) - 1;
  const std::string ind0(column, ' ');                   // indentation of the loop and of its body in the source
  const std::string ind1(column + 2, ' ');
  if(!clang::isa<clang::CompoundStmt>(body))             // range of a single statement does not include ';'
    bodyText = ind1 + bodyText + ";";

  auto pCond = clang::dyn_cast<clang::BinaryOperator>(forExpr->getCond()->IgnoreParenImpCasts());
  if(pCond != nullptr && pCond->getOpcode() == clang::BO_LT && IsLoopVar(pCond->getLHS()))
  {
    const std::string end = pRewriter->RecursiveRewrite(pCond->getRHS());
    if(typeName == "int")
      return "foreach(" + name + " = " + start + " ... " + end + ")\n" + bodyText;
    return "foreach(kgen_" + name + " = " + start + " ... " + end + ")\n" + ind0 + "{\n" + ind1 + "const " + typeName + " " + name + " = (" + typeName + ")kgen_" + name + ";\n" + 
           (clang::isa<clang::CompoundStmt>(body) ? ind1 : std::string("")) + bodyText + "\n" + ind0 + "}";
  }

  const std::string cond = pRewriter->RecursiveRewrite(forExpr->getCond());
  return "for(" + typeName + " " + name + " = " + start + " + programIndex; " + cond + "; " + name + " += programCount)\n" + bodyText;
}

// block expansion: code of a block is uniform, so functions called from it need uniform overloads:
// 'uint DivUp(uint a, uint b)' ==> 'static inline uniform uint DivUp(uniform uint a, uniform uint b)', local variables are uniform too
//
std::string kslicer::ISPCCompiler::BEUniformFunction(const clang::FunctionDecl* a_func, MainClassInfo* a_codeInfo, const clang::CompilerInstance& a_compiler)
{
  if(a_func == nullptr || !a_func->hasBody())
    return "";

  clang::Rewriter rewrite2;
  rewrite2.setSourceMgr(a_compiler.getSourceManager(), a_compiler.getLangOpts());
  auto pRewriter = std::make_shared<ISPCRewriter>(rewrite2, a_compiler, a_codeInfo);
  pRewriter->m_uniformFunction = true;

  auto TypeName = [&](clang::QualType a_type)
  {
    std::string name = pRewriter->RewriteStdVectorTypeStr(a_type.getAsString());
    while(ReplaceFirst(name, "LiteMath::", ""));
    return name;
  };

  const clang::QualType retType = a_func->getReturnType();
  std::string text = "static inline " + (retType->isVoidType() ? std::string("void") : "uniform " + TypeName(retType)) + " " + a_func->getNameAsString() + "(";
  for(unsigned i = 0; i < a_func->getNumParams(); i++)
  {
    const clang::ParmVarDecl* pParam = a_func->getParamDecl(i);
    const clang::QualType type = pParam->getType();
    if(type->isReferenceType())
      text += "uniform " + TypeName(type.getNonReferenceType()) + "& ";
    else if(type->isPointerType())
      text += "uniform " + TypeName(type->getPointeeType()) + "* uniform ";
    else
      text += "uniform " + TypeName(type) + " ";
    text += pParam->getNameAsString();
    if(i + 1 != a_func->getNumParams())
      text += ", ";
  }
  text += ") // uniform overload: called from the code of a block\n";
  return text + pRewriter->RecursiveRewrite(a_func->getBody());
}

// block expansion, 2D parallel loop: the outer loop is a uniform loop of the block, the inner one is 'foreach' over the gang
//
std::string kslicer::ISPCCompiler::RewriteBEParallelFor2D(const clang::ForStmt* a_outer, const clang::ForStmt* a_inner, std::shared_ptr<KernelRewriter> pRewriter,
                                                         const KernelInfo& a_kernel, const uint32_t a_dimOfLevel[2])
{
  const clang::VarDecl* pVar = nullptr;
  if(auto pDeclStmt = clang::dyn_cast_or_null<clang::DeclStmt>(a_outer->getInit()))
    if(pDeclStmt->isSingleDecl())
      pVar = clang::dyn_cast<clang::VarDecl>(pDeclStmt->getSingleDecl());

  bool stepIsOne = false;
  auto pUnary = clang::dyn_cast_or_null<clang::UnaryOperator>(a_outer->getInc());
  if(pUnary != nullptr && pUnary->isIncrementOp())
  {
    auto pRef = clang::dyn_cast<clang::DeclRefExpr>(pUnary->getSubExpr()->IgnoreParenImpCasts());
    stepIsOne = (pRef != nullptr && pRef->getDecl() == pVar);
  }

  const clang::CompilerInstance& compiler = pRewriter->GetCompiler();
  if(pVar == nullptr || pVar->getInit() == nullptr || a_outer->getCond() == nullptr || !stepIsOne)
  {
    kslicer::PrintError("parallel loop of block expansion kernel must have the form 'for(T i = start; cond; i++)'", a_outer->getSourceRange(), compiler.getSourceManager());
    return pRewriter->RecursiveRewrite(a_outer);
  }

  std::string typeName = pVar->getType().getUnqualifiedType().getAsString();
  ReplaceFirst(typeName, "LiteMath::", "");
  const std::string name        = pVar->getNameAsString();
  const unsigned    innerColumn = compiler.getSourceManager().getSpellingColumnNumber(a_inner->getBeginLoc()) - 1;
  return "for(uniform " + typeName + " " + name + " = " + pRewriter->RecursiveRewrite(pVar->getInit()) + "; " + pRewriter->RecursiveRewrite(a_outer->getCond()) + "; " + name + "++)\n" +
         std::string(innerColumn, ' ') + RewriteBEParallelFor(a_inner, pRewriter, a_kernel);
}

// block expansion: 'LiteNN::LinearForwardBlock(w, matOffs, biasOffs, in, IN, out, OUT, BATCH)' in uniform code of a block:
// outputs of the layer are computed by the gang ('foreach'), accumulation in fp32 in the same order as on CPU
//
std::string kslicer::ISPCCompiler::RewriteBEBlockFunction(const clang::CallExpr* a_call, std::shared_ptr<KernelRewriter> pRewriter, const KernelInfo& a_kernel)
{
  BELinearForwardArgs args;
  if(!ParseLinearForwardBlock(a_call, pRewriter, a_kernel, &args))
    return "// LiteNN::LinearForwardBlock: error";

  const std::string IN     = std::to_string(args.inDim);
  const std::string OUT    = std::to_string(args.outDim);
  const std::string STRIDE = std::to_string(args.stride);

  std::stringstream out;
  out << "{ // LiteNN::LinearForwardBlock: " << args.out << "[" << args.batch << " x " << OUT << "] = " << args.in << "[" << args.batch << " x " << IN << "] * W^T + bias\n";
  out << "  const uniform uint kgen_matOffs  = " << args.matOffs << ";\n";
  out << "  const uniform uint kgen_biasOffs = " << args.biasOffs << ";\n";
  out << "  for(uniform uint kgen_b = 0; kgen_b < " << args.batch << "; kgen_b++)\n";
  out << "  {\n";
  out << "    foreach(kgen_j = 0 ... " << OUT << ")\n";
  out << "    {\n";
  out << "      float kgen_acc = 0.0f;\n";
  out << "      for(uniform uint kgen_p = 0; kgen_p < " << IN << "; kgen_p++)\n";
  out << "        kgen_acc += (float)" << args.weights << "[kgen_matOffs + kgen_j*" << STRIDE << " + kgen_p]*(float)" << args.in << "[kgen_b*" << IN << " + kgen_p];\n";
  out << "      " << args.out << "[kgen_b*" << OUT << " + kgen_j] = kgen_acc + (float)" << args.weights << "[kgen_biasOffs + kgen_j];\n";
  out << "    }\n";
  out << "  }\n";
  out << "}\n";
  return out.str();
}

// RTV: payload is passed to the code of a thread by reference in both modes, for wavefront the kernel loads and stores
// elements of payload arrays of its chunk (see 'generated.ispc'), so payload is never accessed with an offset
//
std::string kslicer::ISPCCompiler::RTVGetFakeOffsetExpression(const kslicer::KernelInfo& a_funcInfo, const std::vector<kslicer::ArgFinal>& threadIds)
{
  if(a_funcInfo.pattern == kslicer::PATTERN_TP::PATTERN_RTV)
    return "";
  return ClspvCompiler::RTVGetFakeOffsetExpression(a_funcInfo, threadIds);
}

std::string kslicer::ISPCCompiler::BuildCommand(const std::string& a_inputFile) const
{
  return std::string("ispc ") + a_inputFile + " --target=" + m_target + " -O2 --pic "; // position independent code: executables are PIE by default
}

void kslicer::ISPCCompiler::GenerateShaders(nlohmann::json& a_kernelsJson, const MainClassInfo* a_codeInfo, const kslicer::TextGenSettings& a_settings)
{
  const auto& ignoreFolders     = a_codeInfo->ignoreFolders;

  // output names are based on original input file name (without extension), not on merged '*_temp.cpp' file
  //
  const std::filesystem::path rawName = a_kernelsJson["MainRawName"].get<std::string>();
  a_kernelsJson["MainISPCFile"] = rawName.filename().u8string() + "_kernels.h";

  // LiteMath::half is 'class half_float::half' in canonical types of arguments, in ISPC it is 'half' (typedef of 'float16')
  //
  std::set<std::string> paddedBuffers, texArrays;

  // structs with 2- and 3-component vectors have another layout in ISPC: member vectors and POD members of them are converted
  // field by field (see 'ISPCConvert' in templates_ispc/ispc_class.cpp); pointers to buffers of them are found by type name
  //
  std::set<std::string> convertedTypes, reportedArgs;
  std::map<std::string, std::string> convertedBuffers;   // member ==> struct
  std::map<std::string, std::string> typedefsOfConverted; // 'RandomGen' ==> 'RandomGenT'
  std::vector<kslicer::ISPCRecordInfo> convertedRecords; // structs for which converters are written, nested ones first
  auto useRecordsOf = [&](const kslicer::DataMemberInfo& a_member)
  {
    for(const auto& rec : a_member.ispcRecords)
      if(std::none_of(convertedRecords.begin(), convertedRecords.end(), [&](const auto& r) { return r.name == rec.name; }))
        convertedRecords.push_back(rec);
  };
  for(const auto& member : a_codeInfo->allDataMembers)
    for(const auto& rec : member.second.ispcRecords)
      convertedTypes.insert(rec.name);
  for(const auto& decl : a_kernelsJson["ClassDecls"])
    if(decl.contains("TypedefOfISPC"))
      typedefsOfConverted[decl["Type"].get<std::string>()] = decl["TypedefOfISPC"].get<std::string>();
  for(auto& kernel : a_kernelsJson["Kernels"])
  {
    // RTV wavefront: thread id is computed by kernel for its chunk of threads, thread flags are passed separately
    //
    const bool isWaveRTV = kernel["IsRTV"].get<bool>() && !kernel["IsMega"].get<bool>();
    for(auto& arg : kernel["Args"])
    {
      arg["SkipISPC"]       = arg["IsThreadFlags"].get<bool>() || (isWaveRTV && arg["IsThreadId"].get<bool>());
      arg["IsPayloadISPC"]  = isWaveRTV && arg["IsPayload"].get<bool>(); // loaded and stored by kernel, passed to code of a thread by reference
      std::string type = (arg.contains("Type") && arg["Type"].is_string()) ? arg["Type"].get<std::string>() : std::string("");
      ReplaceFirst(type, "const ", "");
      arg["TypeNoConstISPC"] = type;
      const std::string fullType = (arg.contains("Type") && arg["Type"].is_string()) ? arg["Type"].get<std::string>() : std::string("");
      arg["HostTypeISPC"]    = arg["IsPointer"].get<bool>() ? std::string("ISPCPtr") : fullType; // parameter of '<Kernel>_Wave' in host code
    }

    // arrays of textures ('std::vector<std::shared_ptr<ICombinedImageSampler>>'): array of LiteTextureDesc, see LiteMathAux/LiteImageISPC.h
    //
    for(auto& arg : kernel["Args"])
    {
      const bool texArray = arg.contains("IsImageArray") && arg["IsImageArray"].get<bool>() && arg["IsMember"].get<bool>();
      arg["IsTexArrayISPC"] = texArray;
      if(texArray)
      {
        std::string name = arg["Name"];
        ReplaceFirst(name, "[]", "");
        arg["Name"] = name;
        texArrays.insert(name);
      }
    }

    // member buffers of 'float2', 'uint3', ...: ISPC 2- and 3-component vectors take 16 bytes, the host passes padded copies
    //
    static const std::set<std::string> shortVectors = {"float2", "float3", "int2", "int3", "uint2", "uint3"};
    for(auto& arg : kernel["Args"])
    {
      std::string type = arg.contains("Type") && arg["Type"].is_string() ? arg["Type"].get<std::string>() : std::string("");
      ReplaceFirst(type, "const ", "");
      ReplaceFirst(type, "struct ", "");
      while(!type.empty() && type.back() == ' ')
        type.pop_back();
      if(typedefsOfConverted.count(type) != 0)
        type = typedefsOfConverted[type];
      const bool isMemberBuff = arg["IsMember"].get<bool>() && arg["IsPointer"].get<bool>() && !kernel["IsGraphics"].get<bool>();
      const bool padded       = isMemberBuff && shortVectors.count(type) != 0;
      auto pMember            = a_codeInfo->allDataMembers.find(arg["Name"].get<std::string>());
      const bool converted    = isMemberBuff && pMember != a_codeInfo->allDataMembers.end() && !pMember->second.ispcConvType.empty();
      arg["PaddedISPC"]    = padded;
      arg["ConvertedISPC"] = converted;
      if(padded)
        paddedBuffers.insert(arg["Name"].get<std::string>());
      if(converted)
      {
        convertedBuffers[arg["Name"].get<std::string>()] = pMember->second.ispcConvType;
        useRecordsOf(pMember->second);
      }
      const bool userBuffer = arg["IsPointer"].get<bool>() && !arg["IsMember"].get<bool>() && !kernel["IsGraphics"].get<bool>() && !kernel["IsRTV"].get<bool>(); // RTV: payload of generated code
      if(userBuffer && shortVectors.count(type) != 0 && reportedArgs.insert(type).second)
        kslicer::PrintError("ISPC: '" + type + "' takes 16 bytes in ISPC; buffers of it are padded only for members of the class (std::vector), "
                            "but a pointer is passed to kernel '" + kernel["Name"].get<std::string>() + "' (argument '" + arg["Name"].get<std::string>() + "'), use float4/uint4 or a member vector");
      if(userBuffer && convertedTypes.count(type) != 0 && reportedArgs.insert(type).second)
        kslicer::PrintError("ISPC: '" + type + "' has 2- or 3-component vectors which take 16 bytes in ISPC; buffers of it are converted only for members of the class (std::vector), "
                            "but a pointer is passed to kernel '" + kernel["Name"].get<std::string>() + "' (argument '" + arg["Name"].get<std::string>() + "')");
    }

    for(auto& arg : kernel["Args"])
    {
      if(!arg.contains("Type") || !arg["Type"].is_string())
        continue;
      std::string type = arg["Type"];
      while(ReplaceFirst(type, "class half_float::half", "half"));
      while(ReplaceFirst(type, "half_float::half", "half"));
      arg["Type"] = type;
    }
  }

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
      else if(arg["IsImage"].get<bool>() && arg.contains("ElemISPC"))  // Image2D: pointer to data and 'LiteImageInfo', see LiteMathAux/LiteImageISPC.h
      {
        nlohmann::json buff;
        buff["Name"]    = name;
        buff["Type"]    = arg["ElemISPC"];
        buff["IsImage"] = true;
        memberBuffers.push_back(buff);
      }
      else if(arg["IsPointer"].get<bool>())
      {
        nlohmann::json buff;
        buff["Name"]    = name;
        buff["Type"]    = arg["Type"];
        buff["IsImage"] = false;
        memberBuffers.push_back(buff);
      }
    }
  }
  std::set<std::string> samplers;
  for(const auto& kernel : a_kernelsJson["Kernels"])
    for(const auto& sampler : kernel["SamplersISPC"])
      samplers.insert(sampler.get<std::string>());
  a_kernelsJson["SamplersISPC"] = std::vector<std::string>(samplers.begin(), samplers.end());
  for(auto& mainFunc : a_kernelsJson["MainFunctions"]) // 'Test(args)' ==> 'Test_BE(args)', copy of control function with calls of block expansion kernels
  {
    const std::string name = mainFunc["Name"];
    const std::string decl = mainFunc["DeclOrig"];
    mainFunc["DeclBE"] = name + "_BE" + decl.substr(name.size());

    // RTV wavefront: payload (local variables of control function) is stored in arrays for a chunk of threads
    //
    mainFunc["IsWaveISPC"]    = false;
    mainFunc["PayloadISPC"]   = std::vector<nlohmann::json>();
    mainFunc["ThreadIdsISPC"] = std::vector<std::string>();
    for(const auto& cf : a_codeInfo->mainFunc)
    {
      if(cf.Name != name || cf.pattern != kslicer::PATTERN_TP::PATTERN_RTV)
        continue;
      for(const auto& var : cf.InOuts)
        if(var.isThreadId)
          mainFunc["ThreadIdsISPC"].push_back(var.name);
      if(a_codeInfo->megakernelRTV)
        break;
      mainFunc["IsWaveISPC"] = true;
      for(const auto& var : cf.Locals)
      {
        if(var.second.name == "threadFlags")
          continue;
        nlohmann::json local;
        local["Name"]     = var.second.name;
        local["TypeISPC"] = a_codeInfo->pShaderFuncRewriter->RewriteStdVectorTypeStr(kslicer::CleanTypeName(var.second.type));
        mainFunc["PayloadISPC"].push_back(local);
      }
    }
  }
  a_kernelsJson["PaddedBuffersISPC"] = std::vector<std::string>(paddedBuffers.begin(), paddedBuffers.end());
  a_kernelsJson["ConvertedBuffersISPC"] = std::vector<nlohmann::json>();
  for(const auto& buff : convertedBuffers)
    a_kernelsJson["ConvertedBuffersISPC"].push_back(nlohmann::json{{"Name", buff.first}, {"Type", buff.second}});
  for(auto& var : a_kernelsJson["ClassVars"]) // POD members of such structs in UBO
  {
    auto pMember = a_codeInfo->allDataMembers.find(var["Name"].get<std::string>());
    var["ConvertedISPC"] = pMember != a_codeInfo->allDataMembers.end() && !pMember->second.ispcConvType.empty() && !var["IsArray"].get<bool>();
    if(var["ConvertedISPC"].get<bool>())
      useRecordsOf(pMember->second);
  }
  a_kernelsJson["ConvertersISPC"] = std::vector<nlohmann::json>(); // only for converted structs: ISPC header declares only structs of 'export' functions
  for(const auto& rec : convertedRecords)
  {
    nlohmann::json recJson;
    recJson["Type"]       = rec.name;
    recJson["FieldsISPC"] = std::vector<nlohmann::json>();
    for(const auto& field : rec.fields)
      recJson["FieldsISPC"].push_back(nlohmann::json{{"Name", field.name}, {"ArraySize", field.arraySize}, {"Convert", field.convert}});
    a_kernelsJson["ConvertersISPC"].push_back(recJson);
  }
  a_kernelsJson["TexArraysISPC"]     = std::vector<std::string>(texArrays.begin(), texArrays.end());
  a_kernelsJson["WaveChunkISPC"] = m_waveChunk;
  a_kernelsJson["MegaChunkISPC"] = m_megaChunk;
  // RTV megakernels ('-megakernel 1'): kernels called by control functions become ISPC functions of one thread;
  // they are placed in the single ISPC file, so a kernel used by several control functions is written once
  //
  std::set<std::string> subkernelDecls;
  nlohmann::json subkernels = std::vector<nlohmann::json>();
  for(const auto& kernel : a_kernelsJson["Kernels"])
  {
    if(!kernel.contains("Subkernels"))
      continue;
    for(const auto& sub : kernel["Subkernels"])
    {
      const std::string decl = sub["Decl"];
      if(subkernelDecls.find(decl) != subkernelDecls.end())
        continue;
      subkernelDecls.insert(decl);
      subkernels.push_back(sub);
    }
  }
  a_kernelsJson["SubkernelsISPC"] = subkernels;

  // copies of functions which get buffers ('ShityFunctions' of kernels): declarations first, then definitions
  //
  std::set<std::string> globalPtrFuncsDone;
  std::vector<std::string> globalPtrFuncDecls, globalPtrFuncs;
  for(const auto& kernel : a_kernelsJson["Kernels"])
  {
    if(!kernel.contains("ShityFunctions"))
      continue;
    for(const auto& func : kernel["ShityFunctions"])
    {
      const std::string text = func;
      if(globalPtrFuncsDone.find(text) != globalPtrFuncsDone.end())
        continue;
      globalPtrFuncsDone.insert(text);
      const size_t bodyPos = text.find("{");
      if(bodyPos == std::string::npos)
        globalPtrFuncDecls.push_back("static inline " + text);
      else
      {
        std::string decl = text.substr(0, bodyPos); // a copy may call another copy which is written after it
        while(!decl.empty() && (decl.back() == ' ' || decl.back() == '\n'))
          decl.pop_back();
        if(globalPtrFuncsDone.insert(decl + ";").second)
          globalPtrFuncDecls.push_back("static inline " + decl + ";");
        globalPtrFuncs.push_back("static inline " + text);
      }
    }
  }
  globalPtrFuncDecls.insert(globalPtrFuncDecls.end(), globalPtrFuncs.begin(), globalPtrFuncs.end());
  a_kernelsJson["GlobalPtrFunctionsISPC"] = globalPtrFuncDecls;
  kslicer::AddGraphicsToISPCJson(a_kernelsJson, *a_codeInfo); // draws of graphics pipeline (LiteRS), see graphics_pipeline.cpp
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
  if(a_kernelsJson["HasGraphics"].get<bool>())
    build += " --opt=disable-fma"; // graphics pipeline: depth and coverage the same as in C++ (it is compiled without FMA contraction)
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
  const std::filesystem::path liteMathAux = std::filesystem::absolute("apps/LiteMathAux"); // ISPC headers of LiteMath for the generated code (current folder is kslicer root)
  if(std::filesystem::exists(liteMathAux / "LiteMathISPC.h"))
    build += " -I" + liteMathAux.u8string();
  buildSH << build.c_str() << " ";
  for(auto folder : ignoreFolders)
    buildSH << "-I" << folder.c_str() << " ";
  buildSH << std::endl;
  buildSH.close();
}

kslicer::umap<std::string, std::string> ListISPCVectorReplacements()
{
  kslicer::umap<std::string, std::string> m_vecReplacements;
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
  static const kslicer::uset<std::string> scalarTypes = {"float", "double", "int", "uint", "unsigned", "unsigned int", "bool", "_Bool", "char", "unsigned char", "short", "unsigned short",
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
    if(typeInCL.rfind("const ", 0) == 0)                 // without 'uniform' a global variable of ISPC is varying
      typeInCL = "const uniform " + typeInCL.substr(6);
    else
      typeInCL = "uniform " + typeInCL;
    result = typeInCL + " " + a_decl.name + " = " + kslicer::GetRangeSourceCode(a_decl.srcRange, a_compiler) + ";";
    break;
    case kslicer::DECL_IN_CLASS::DECL_TYPEDEF:
    ReplaceFirst(typeInCL, "LiteMath::", "");
    if(typeInCL != a_decl.name) // 'typedef LiteMath::float4 float4;' is not needed
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
