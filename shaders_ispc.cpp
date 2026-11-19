#include "kslicer.h"
#include "template_rendering.h"

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
    if(typeOfParam->isPointerType() && typeOfParam->getPointeeType()->isRecordType()) // 'Varyings* a_out' ==> 'Varyings& a_out': ISPC references are used as structures
      typeStr = typeOfParam->getPointeeType().getAsString() + "&";
    else if(typeOfParam->isPointerType())
      typeStr += "*";
    
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
  if(clang::isa<clang::CXXMethodDecl>(fDecl)) // methods are processed separately
    return true;

  if(fDecl->hasBody() && WasNotRewrittenYet(fDecl->getBody()))  // bodies of local functions are rewritten: 'float(x)' ==> '(float)(x)', 'std::pow' ==> 'pow', ...
  {
    RewrittenFunction done = RewriteFunction(fDecl);
    done.funDecl = kslicer::GetRangeSourceCode(fDecl->getSourceRange(), m_compiler); // declaration is kept as is: ISPC understands C pointers of parameters
    const auto posBrace = done.funDecl.find("{");
    if(posBrace != std::string::npos)
      done.funDecl = done.funDecl.substr(0, posBrace);
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
  if(m_kernelMode)
  {
    std::string originalText = kslicer::GetRangeSourceCode(expr->getSourceRange(), m_compiler);
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

bool kslicer::ISPCRewriter::VisitCXXMemberCallExpr_Impl(clang::CXXMemberCallExpr* f)  
{ 
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
bool kslicer::ISPCRewriter::VisitCXXConstructExpr_Impl(clang::CXXConstructExpr* call) 
{ 
  const clang::CXXConstructorDecl* ctorDecl = call->getConstructor();
  if(ctorDecl == nullptr || ctorDecl->isCopyOrMoveConstructor() || call->getNumArgs() == 0 || !WasNotRewrittenYet(call))
    return true;

  const std::string typeName = ctorDecl->getNameInfo().getName().getAsString();
  static const kslicer::uset<std::string> vecTypes = {"float2", "float3", "float4", "int2", "int3", "int4", "uint2", "uint3", "uint4"};
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

  ReplaceText(call->getSourceRange(), text);
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
  ReplaceText(cast->getSourceRange(), "(" + typeName + ")(" + RecursiveRewrite(cast->getSubExpr()) + ")");
  MarkRewritten(cast);
  return true;
}

bool kslicer::ISPCRewriter::VisitCallExpr_Impl(clang::CallExpr* call)                    
{ 
  if(m_kernelMode && WasNotRewrittenYet(call) && !clang::isa<clang::CXXOperatorCallExpr>(call)) // 'a*b' of LiteMath types is an operator call
  {
    clang::FunctionDecl* fDecl = call->getDirectCallee();
    if(fDecl == nullptr)
      return true;

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
  }
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
