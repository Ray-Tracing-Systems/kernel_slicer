#include "kslicer.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ExprCXX.h"

#include <iostream>

// std::unordered_map<Key,X> as class member becomes single GPU buffer of slots:
//
//   struct HashMapSlot_m_items { X val; Key key; uint _pad[N]; };
//
// 'val' is placed first, so it has offset 0 both in C++ and in std430;
// 'key' has offset sizeof(X); explicit '_pad' makes slot size the same in C++ and in std430.
// Open addressing with linear probing, capacity is power of 2, empty slot has 'key == sentinel'.

std::unordered_map<std::string, size_t> ListPredefinedAligmentTypes(); // extractor.cpp

bool kslicer::IsHashMapContainer(const std::string& a_typeName)
{
  return (a_typeName == "unordered_map") || (a_typeName == "std::unordered_map");
}

static const clang::ClassTemplateSpecializationDecl* GetHashMapDecl(const clang::DeclContext* a_ctx)
{
  if(a_ctx == nullptr)
    return nullptr;
  auto specDecl = clang::dyn_cast<clang::ClassTemplateSpecializationDecl>(a_ctx);
  if(specDecl == nullptr || !kslicer::IsHashMapContainer(specDecl->getNameAsString()))
    return nullptr;
  return specDecl;
}

bool kslicer::IsHashMapType(clang::QualType a_qt)
{
  const clang::QualType qt = a_qt.getNonReferenceType().getCanonicalType();
  return GetHashMapDecl(qt->getAsCXXRecordDecl()) != nullptr;
}

bool kslicer::IsHashMapIteratorType(clang::QualType a_qt)
{
  const clang::QualType qt = a_qt.getNonReferenceType().getCanonicalType();
  const clang::CXXRecordDecl* recordDecl = qt->getAsCXXRecordDecl();
  if(recordDecl == nullptr)
    return false;
  const std::string name = recordDecl->getNameAsString();
  if(name != "iterator" && name != "const_iterator")
    return false;
  return GetHashMapDecl(recordDecl->getParent()) != nullptr;
}

std::string kslicer::HashMapSlotTypeName(const std::string& a_mapName) { return "HashMapSlot_" + a_mapName; }
std::string kslicer::HashMapFuncName(const std::string& a_mapName, const std::string& a_func) { return "hmap_" + a_mapName + "_" + a_func; }

static const clang::Expr* RemoveTemporaries(const clang::Expr* a_expr)
{
  const clang::Expr* expr = a_expr;
  while(expr != nullptr)
  {
    const clang::Expr* next = nullptr;
    if(auto e = clang::dyn_cast<clang::ExprWithCleanups>(expr))
      next = e->getSubExpr();
    else if(auto e = clang::dyn_cast<clang::MaterializeTemporaryExpr>(expr))
      next = e->getSubExpr();
    else if(auto e = clang::dyn_cast<clang::CXXBindTemporaryExpr>(expr))
      next = e->getSubExpr();
    else if(auto e = clang::dyn_cast<clang::ImplicitCastExpr>(expr))
      next = e->getSubExpr();
    else if(auto e = clang::dyn_cast<clang::ParenExpr>(expr))
      next = e->getSubExpr();
    else if(auto e = clang::dyn_cast<clang::CXXConstructExpr>(expr)) // copy of iterator, conversion of iterator to const_iterator
    {
      if(e->getNumArgs() == 1)
        next = e->getArg(0);
    }
    if(next == nullptr)
      break;
    expr = next;
  }
  return expr;
}

std::string kslicer::GetHashMapNameFromExpr(const clang::Expr* a_expr)
{
  const clang::Expr* expr = RemoveTemporaries(a_expr);
  if(expr == nullptr)
    return "";

  if(auto memberExpr = clang::dyn_cast<clang::MemberExpr>(expr))  // 'm_items' (i.e. 'this->m_items')
  {
    if(IsHashMapType(memberExpr->getType()))
      return memberExpr->getMemberDecl()->getNameAsString();
  }
  else if(auto call = clang::dyn_cast<clang::CXXMemberCallExpr>(expr)) // 'm_items.find(key)' or 'm_items.end()'
  {
    if(IsHashMapIteratorType(call->getType()))
      return GetHashMapNameFromExpr(call->getImplicitObjectArgument());
  }
  else if(auto declRef = clang::dyn_cast<clang::DeclRefExpr>(expr)) // 'it' declared as 'auto it = m_items.find(key)'
  {
    auto varDecl = clang::dyn_cast<clang::VarDecl>(declRef->getDecl());
    if(varDecl != nullptr && IsHashMapIteratorType(varDecl->getType()) && varDecl->hasInit())
      return GetHashMapNameFromExpr(varDecl->getInit());
  }

  return "";
}

const clang::CXXOperatorCallExpr* kslicer::GetHashMapSubscript(const clang::Expr* a_expr)
{
  const clang::Expr* expr = a_expr;
  while(expr != nullptr)
  {
    expr = expr->IgnoreParenImpCasts();
    if(auto memberExpr = clang::dyn_cast<clang::MemberExpr>(expr)) // 'm_items[key].val.x'
    {
      expr = memberExpr->getBase();
      continue;
    }
    break;
  }

  auto opCall = clang::dyn_cast_or_null<clang::CXXOperatorCallExpr>(expr);
  if(opCall == nullptr || opCall->getOperator() != clang::OO_Subscript || opCall->getNumArgs() != 2)
    return nullptr;
  if(GetHashMapNameFromExpr(opCall->getArg(0)) == "")
    return nullptr;
  return opCall;
}

size_t kslicer::GetStd430Alignment(clang::QualType a_qt, const clang::ASTContext& a_astContext)
{
  static const auto predefined = ListPredefinedAligmentTypes();

  const clang::QualType qt = a_qt.getCanonicalType();
  const std::string typeName = kslicer::CleanTypeName(qt.getAsString());
  auto pFound = predefined.find(typeName);
  if(pFound != predefined.end())
    return pFound->second;

  if(auto arrayType = clang::dyn_cast<clang::ConstantArrayType>(qt.getTypePtr()))
    return GetStd430Alignment(arrayType->getElementType(), a_astContext);

  if(auto recordDecl = qt->getAsRecordDecl())
  {
    size_t maxAlign = 4;
    for(auto field : recordDecl->fields())
      maxAlign = std::max(maxAlign, GetStd430Alignment(field->getType(), a_astContext));
    return maxAlign;
  }

  return std::max<size_t>(4, a_astContext.getTypeSizeInChars(qt).getQuantity()); // scalars
}

void kslicer::MainClassInfo::ProcessHashMaps(const clang::ASTContext& a_astContext)
{
  hashMaps.clear();
  for(const auto& member : dataMembers)
  {
    if(!member.isContainer || !IsHashMapContainer(member.containerType))
      continue;

    HashMapInfo info;
    info.name       = member.name;
    info.valueType  = kslicer::CleanTypeName(member.containerDataType);
    info.slotType   = HashMapSlotTypeName(member.name);
    info.valueSize  = member.containerDataSize;
    info.valueAlign = std::max<size_t>(4, member.containerDataAlign);

    if(member.containerKeyType == "int")
    {
      info.keyType  = "int";
      info.sentinel = "int(0x80000000)";
    }
    else if(member.containerKeyType == "unsigned int")
    {
      info.keyType  = "uint";
      info.sentinel = "0xFFFFFFFFu";
    }
    else
    {
      std::cout << "  [kslicer]: error, std::unordered_map '" << member.name.c_str() << "' has key of type '" << member.containerKeyType.c_str()
                << "'; only 32 bit 'int' and 'uint' keys are supported" << std::endl;
      continue;
    }

    if(info.valueSize == 0 || info.valueSize % 4 != 0)
    {
      std::cout << "  [kslicer]: error, std::unordered_map '" << member.name.c_str() << "' has value of type '" << info.valueType.c_str()
                << "' with size " << info.valueSize << "; size of value must be multiple of 4 bytes" << std::endl;
      continue;
    }

    static const auto predefined = ListPredefinedAligmentTypes(); // float3 is fine: std430 places 'key' right after it
    const bool isUserStruct = (member.pContainerDataTypeDeclIfRecord != nullptr) && (predefined.find(info.valueType) == predefined.end());
    if(isUserStruct && info.valueSize % info.valueAlign != 0) // std430 rounds struct size up to its aligment, C++ may not
    {
      std::cout << "  [kslicer]: error, std::unordered_map '" << member.name.c_str() << "' has value of type '" << info.valueType.c_str()
                << "' with size " << info.valueSize << ", which is not multiple of its aligment in shaders (" << info.valueAlign
                << "); please add padding fields to '" << info.valueType.c_str() << "'" << std::endl;
      continue;
    }

    const size_t rawSize = info.valueSize + sizeof(uint32_t); // {X val; Key key;}
    info.slotSize = ((rawSize + info.valueAlign - 1) / info.valueAlign) * info.valueAlign;
    info.padWords = (info.slotSize - rawSize) / sizeof(uint32_t);

    hashMaps[member.name] = info;
    std::cout << "  hash map " << info.name.c_str() << ": std::unordered_map<" << info.keyType.c_str() << "," << info.valueType.c_str() << ">, slot size = "
              << info.slotSize << " bytes (" << info.padWords << " padding words)" << std::endl;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////// Slang rewriting ///////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool kslicer::SlangRewriter::RewriteHashMapSubscript(clang::CXXOperatorCallExpr* expr)
{
  if(!WasNotRewrittenYet(expr) || GetHashMapSubscript(expr) != expr)
    return false;

  const std::string mapName = GetHashMapNameFromExpr(expr->getArg(0));
  const std::string keyText = RecursiveRewrite(expr->getArg(1));
  ReplaceTextOrWorkAround(expr->getSourceRange(), mapName + "[" + HashMapFuncName(mapName, "insert") + "(" + keyText + ")].val");
  MarkRewritten(expr);
  return true;
}

bool kslicer::SlangRewriter::RewriteHashMapMemberCall(clang::CXXMemberCallExpr* call)
{
  if(!WasNotRewrittenYet(call) || !IsHashMapType(call->getObjectType()))
    return false;

  const std::string mapName = GetHashMapNameFromExpr(call->getImplicitObjectArgument());
  const std::string fname   = call->getMethodDecl()->getNameAsString();
  if(mapName == "")
  {
    kslicer::PrintError("std::unordered_map is supported in GPU code only as a member of main class", call->getSourceRange(), m_compiler.getSourceManager());
    return false;
  }

  const std::string capacity = m_codeInfo->pShaderCC->UBOAccess(mapName + "_capacity");
  const std::string size     = m_codeInfo->pShaderCC->UBOAccess(mapName + "_size");
  const std::string argText  = (call->getNumArgs() > 0) ? RecursiveRewrite(call->getArg(0)) : "";
  const std::string findText = HashMapFuncName(mapName, "find") + "(" + argText + ")";

  std::string result;
  if(fname == "find")
    result = findText;
  else if(fname == "end" || fname == "cend")
    result = capacity;
  else if(fname == "count")
    result = "(" + findText + " != " + capacity + " ? 1u : 0u)";
  else if(fname == "contains")
    result = "(" + findText + " != " + capacity + ")";
  else if(fname == "at")
    result = mapName + "[" + findText + "].val";
  else if(fname == "size")
    result = size;
  else if(fname == "empty")
    result = "(" + size + " == 0)";
  else
  {
    kslicer::PrintError("std::unordered_map::" + fname + " is not supported in GPU code", call->getSourceRange(), m_compiler.getSourceManager());
    return false;
  }

  ReplaceTextOrWorkAround(call->getSourceRange(), result);
  MarkRewritten(call);
  return true;
}

bool kslicer::SlangRewriter::RewriteHashMapIteratorAccess(clang::MemberExpr* expr)
{
  if(!WasNotRewrittenYet(expr))
    return false;

  auto arrow = clang::dyn_cast<clang::CXXOperatorCallExpr>(expr->getBase()->IgnoreImpCasts());
  if(arrow == nullptr || arrow->getOperator() != clang::OO_Arrow || arrow->getNumArgs() != 1 || !IsHashMapIteratorType(arrow->getArg(0)->getType()))
    return false;

  const std::string mapName = GetHashMapNameFromExpr(arrow->getArg(0));
  const std::string member  = expr->getMemberDecl()->getNameAsString();
  if(mapName == "")
  {
    kslicer::PrintError("can't find std::unordered_map for this iterator; declare it as 'auto it = m_map.find(key)'", expr->getSourceRange(), m_compiler.getSourceManager());
    return false;
  }
  if(member != "first" && member != "second")
    return false;

  const std::string itText = RecursiveRewrite(arrow->getArg(0));
  ReplaceTextOrWorkAround(expr->getSourceRange(), mapName + "[" + itText + "]." + (member == "first" ? "key" : "val"));
  MarkRewritten(expr);
  return true;
}

bool kslicer::SlangRewriter::RewriteHashMapAtomicAdd(const clang::Expr* a_wholeExpr, const clang::Expr* a_lhs, const std::string& a_value)
{
  if(!WasNotRewrittenYet(a_wholeExpr))
    return false;

  const std::string typeName = a_lhs->getType().getNonReferenceType().getCanonicalType().getUnqualifiedType().getAsString();
  std::string slangType;
  if(typeName == "int")
    slangType = "int";
  else if(typeName == "unsigned int")
    slangType = "uint";
  else if(typeName == "float")
    slangType = "float";
  else
  {
    kslicer::PrintError("atomic update of std::unordered_map value is supported only for 'int', 'uint' and 'float', but not for '" + typeName + "'", 
                        a_wholeExpr->getSourceRange(), m_compiler.getSourceManager());
    return false;
  }

  std::string func = "InterlockedAdd";
  if(slangType == "float")
  {
    if(m_codeInfo->atomicFloatEmul)
      func = "InterlockedAddEmul1f";
    else
      m_codeInfo->globalShaderFeatures.useFloatAtomicAdd = true;
  }

  if(m_codeInfo->hashMapSubgroups) // aggregate updates of the same key inside subgroup: 'hmap_m_hist_add(key, value)'
  {
    std::string path = "";            // 'm_items[key].val.x' ==> '.x'
    const clang::Expr* expr = a_lhs->IgnoreParenImpCasts();
    while(auto memberExpr = clang::dyn_cast<clang::MemberExpr>(expr))
    {
      const std::string name = memberExpr->getMemberDecl()->getNameAsString();
      if(name != "")                  // skip anonymous unions and structs, 'float4::x' in LiteMath for example
        path = "." + name + path;
      expr = memberExpr->getBase()->IgnoreParenImpCasts();
    }
    const clang::CXXOperatorCallExpr* subscript = GetHashMapSubscript(a_lhs);
    const std::string mapName = GetHashMapNameFromExpr(subscript->getArg(0));
    const std::string keyText = RecursiveRewrite(subscript->getArg(1));

    MainClassInfo::HashMapAddFunc addFunc;
    addFunc.mapName    = mapName;
    addFunc.path       = ".val" + path;
    addFunc.valueType  = slangType;
    addFunc.atomicFunc = func;
    addFunc.name       = HashMapFuncName(mapName, "add");
    for(char c : path)
      addFunc.name += (c == '.') ? '_' : c;
    m_codeInfo->hashMapAddFuncs[addFunc.name] = addFunc;

    ReplaceTextOrWorkAround(a_wholeExpr->getSourceRange(), addFunc.name + "(" + keyText + ", " + slangType + "(" + a_value + "))");
    MarkRewritten(a_wholeExpr);
    return true;
  }

  const std::string lhsText = RecursiveRewrite(a_lhs);
  ReplaceTextOrWorkAround(a_wholeExpr->getSourceRange(), func + "(" + lhsText + ", " + slangType + "(" + a_value + "))");
  MarkRewritten(a_wholeExpr);
  return true;
}
