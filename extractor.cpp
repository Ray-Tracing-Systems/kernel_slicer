#include "extractor.h"
#include "ast_matchers.h"

#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/DeclTemplate.h"

#include <queue>

class FuncExtractor : public clang::RecursiveASTVisitor<FuncExtractor>
{
public:
  
  FuncExtractor(const clang::CompilerInstance& a_compiler, kslicer::MainClassInfo& a_codeInfo) : m_compiler(a_compiler), m_sm(a_compiler.getSourceManager()), m_patternImpl(a_codeInfo)
  { 
    
  }

  kslicer::FuncData* pCurrProcessedFunc = nullptr; 

  bool VisitCallExpr(clang::CallExpr* call)
  {
    clang::FunctionDecl* f = call->getDirectCallee();
    if(f == nullptr)
      return true;

    if(f->isOverloadedOperator())
      return true;

    std::string fileName = std::string(m_sm.getFilename(f->getSourceRange().getBegin())); // check that we are in test_class.cpp or test_class.h or sms like that;                                                                             
    for(auto f : m_patternImpl.includeToShadersFolders)                                   // exclude everything from "shader" folders
    {
      if(fileName.find(f) != std::string::npos)
       return true;
    }

    if(fileName.find(".h") == std::string::npos && fileName.find(".cpp") == std::string::npos && fileName.find(".cxx") == std::string::npos)
      return true;

    kslicer::FuncData func;
    func.name     = f->getNameAsString();
    func.astNode  = f;
    func.srcRange = f->getSourceRange(); 
    func.srcHash  = kslicer::GetHashOfSourceRange(func.srcRange);
    func.isMember = clang::isa<clang::CXXMethodDecl>(func.astNode);
    func.isKernel = m_patternImpl.IsKernel(func.name);
    func.depthUse = 0;

    //pCurrProcessedFunc->calledMembers.insert(func.name);

    if(func.isKernel)
    {
      assert(pCurrProcessedFunc != nullptr);
      kslicer::PrintError(std::string("Calling kernel + '" + func.name + "' from a kernel is not allowed currently, ") + pCurrProcessedFunc->name, func.srcRange, m_sm);
      return true;
    }
    
    if(func.isMember) // currently we support export for members of current class only
    {
      clang::CXXRecordDecl* recordDecl = clang::dyn_cast<clang::CXXMemberCallExpr>(call)->getRecordDecl();
      const auto pType    = recordDecl->getTypeForDecl();     
      const auto qt       = pType->getLocallyUnqualifiedSingleStepDesugaredType();
      const auto typeName = qt.getAsString();
      if(typeName != std::string("class ") + m_patternImpl.mainClassName && 
         typeName != std::string("struct ") + m_patternImpl.mainClassName && 
         typeName != m_patternImpl.mainClassName)
        return true;
    }

    usedFunctions[func.srcHash] = func;

    return true;
  }

  std::unordered_map<uint64_t, kslicer::FuncData> usedFunctions;

private:
  const clang::CompilerInstance& m_compiler;
  const clang::SourceManager&    m_sm;
  kslicer::MainClassInfo&        m_patternImpl;

};


std::vector<kslicer::FuncData> kslicer::ExtractUsedFunctions(MainClassInfo& a_codeInfo, const clang::CompilerInstance& a_compiler)
{
  std::queue<FuncData> functionsToProcess; 

  for(const auto& k : a_codeInfo.kernels)
  {
    kslicer::FuncData func;
    func.name     = k.second.name;
    func.astNode  = k.second.astNode;
    func.srcRange = k.second.astNode->getSourceRange();
    func.srcHash  = GetHashOfSourceRange(func.srcRange);
    func.isMember = true; //isa<clang::CXXMethodDecl>(kernel.astNode);
    func.isKernel = true;
    func.depthUse = 0;
    functionsToProcess.push(func);
  }

  std::unordered_map<uint64_t, FuncData> usedFunctions;
  usedFunctions.reserve(functionsToProcess.size()*10);
  
  FuncExtractor visitor(a_compiler, a_codeInfo); // first traverse kernels to used functions, then repeat this recursivelly in a breadth first manner ... 
  while(!functionsToProcess.empty())
  {
    auto currFunc = functionsToProcess.front(); functionsToProcess.pop();
    
    visitor.pCurrProcessedFunc = &currFunc;
    visitor.TraverseDecl(const_cast<clang::FunctionDecl*>(currFunc.astNode));

    for(auto& f : visitor.usedFunctions)
    {
      auto nextFunc     = f.second;
      nextFunc.depthUse = currFunc.depthUse + 1;
      functionsToProcess.push(nextFunc);
      if(f.second.isMember)
        currFunc.calledMembers.insert(f.second.name);      
    }

    if(!currFunc.isKernel)
      usedFunctions[currFunc.srcHash] = currFunc;
    
    for(auto foundCall : visitor.usedFunctions)
    {
      auto p = usedFunctions.find(foundCall.first);
      if(p == usedFunctions.end())
        usedFunctions[foundCall.first] = foundCall.second;
      else
        p->second.depthUse++;
    }

    visitor.usedFunctions.clear();
  }

  std::vector<kslicer::FuncData> result; result.reserve(usedFunctions.size());
  for(const auto& f : usedFunctions)
    result.push_back(f.second);
  
  std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) { return a.depthUse > b.depthUse; });

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class DataExtractor : public clang::RecursiveASTVisitor<DataExtractor>
{
public:
  
  DataExtractor(const clang::CompilerInstance& a_compiler, kslicer::MainClassInfo& a_codeInfo, 
                std::unordered_map<std::string, kslicer::DataMemberInfo>& a_members, std::unordered_map<std::string, kslicer::UsedContainerInfo>& a_auxContainers) : 
                m_compiler(a_compiler), m_sm(a_compiler.getSourceManager()), m_patternImpl(a_codeInfo), m_usedMembers(a_members), m_auxContainers(a_auxContainers)
  { 
    
  }
  
  bool VisitMemberExpr(clang::MemberExpr* expr) 
  {
    //const std::string debugText1 = kslicer::GetRangeSourceCode(expr->getSourceRange(), m_compiler);
    if(false) // TODO: check if this is texture.sample, save texture name 
    {
      // clang::CXXMethodDecl* fDecl = call->getMethodDecl();  
      // if(fDecl == nullptr)  
      //   return;
      // 
      // //std::string debugText = GetRangeSourceCode(call->getSourceRange(), m_compiler); 
      // std::string fname     = fDecl->getNameInfo().getName().getAsString();
      // clang::Expr* pTexName =	call->getImplicitObjectArgument(); 
      // std::string objName   = GetRangeSourceCode(pTexName->getSourceRange(), m_compiler);     
      // 
      // if(fname == "sample" || fname == "Sample")
    }

    std::string setter, containerName;
    if(kslicer::CheckSettersAccess(expr, &m_patternImpl, m_compiler, &setter, &containerName))
    {
      clang::QualType qt = expr->getType(); // 
      kslicer::UsedContainerInfo container;
      container.type     = qt.getAsString();
      container.name     = setter + "_" + containerName;            
      container.kind     = kslicer::GetKindOfType(qt, false);
      container.isConst  = qt.isConstQualified();
      container.isSetter = true;
      container.setterPrefix = setter;
      container.setterSuffix = containerName;
      m_auxContainers[container.name] = container;
      return true;
    }

    clang::ValueDecl* pValueDecl = expr->getMemberDecl();
    if(!clang::isa<clang::FieldDecl>(pValueDecl))
      return true;

    clang::FieldDecl* pFieldDecl  = clang::dyn_cast<clang::FieldDecl>(pValueDecl);
    clang::RecordDecl* pRecodDecl = pFieldDecl->getParent();

    const std::string thisTypeName = pRecodDecl->getNameAsString();
    if(thisTypeName != m_patternImpl.mainClassName)
      return true;

    // process access to arguments payload->xxx
    //
    clang::Expr* baseExpr = expr->getBase(); 
    assert(baseExpr != nullptr);

    auto baseName = kslicer::GetRangeSourceCode(baseExpr->getSourceRange(), m_compiler);
    auto member   = kslicer::ExtractMemberInfo(pFieldDecl, m_compiler.getASTContext());
    m_usedMembers[member.name] = member;

    return true;
  }

private:
  const clang::CompilerInstance& m_compiler;
  const clang::SourceManager&    m_sm;
  kslicer::MainClassInfo&        m_patternImpl;
  std::unordered_map<std::string, kslicer::DataMemberInfo>& m_usedMembers; 
  std::unordered_map<std::string, kslicer::UsedContainerInfo>& m_auxContainers;

};

std::unordered_map<std::string, kslicer::DataMemberInfo> kslicer::ExtractUsedMemberData(kslicer::KernelInfo* pKernel, const kslicer::FuncData& a_funcData, const std::vector<kslicer::FuncData>& a_otherMembers,
                                                                                        std::unordered_map<std::string, kslicer::UsedContainerInfo>& a_auxContainers, MainClassInfo& a_codeInfo, const clang::CompilerInstance& a_compiler)
{
  std::unordered_map<std::string, kslicer::DataMemberInfo> result;
  std::unordered_map<std::string, kslicer::FuncData>       allMembers;

  for(auto f : a_otherMembers)
    allMembers[f.name] = f;
  
  //process a_funcData.astNode to get all accesed data, then recursivelly process calledMembers
  //
  std::queue<FuncData> functionsToProcess; 
  functionsToProcess.push(a_funcData);

  DataExtractor visitor(a_compiler, a_codeInfo, result, a_auxContainers);

  while(!functionsToProcess.empty())
  {
    auto currFunc = functionsToProcess.front(); functionsToProcess.pop();
    
    // (1) process curr function to get all accesed data
    //
    visitor.TraverseDecl(const_cast<clang::FunctionDecl*>(currFunc.astNode));

    // (2) then recursivelly process calledMembers
    //
    for(auto nextFuncName : currFunc.calledMembers)
    {
      auto pNext = allMembers.find(nextFuncName);
      if(pNext != allMembers.end())
        functionsToProcess.push(pNext->second);
    }
  }
  
  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<kslicer::ArgMatch> kslicer::MatchCallArgsForKernel(clang::CallExpr* call, const KernelInfo& k, const clang::CompilerInstance& a_compiler)
{
  std::vector<kslicer::ArgMatch> result;
  const clang::FunctionDecl* fDecl = call->getDirectCallee();  
  if(fDecl == nullptr || clang::isa<clang::CXXOperatorCallExpr>(call)) 
    return result;
  
  std::string debugText = kslicer::GetRangeSourceCode(call->getSourceRange(), a_compiler);

  for(size_t i=0;i<call->getNumArgs();i++)
  {
    const clang::ParmVarDecl* formalArg = fDecl->getParamDecl(i);
    const clang::Expr*        actualArg = call->getArg(i);
    const clang::QualType     qtFormal  = formalArg->getType();
    //const clang::QualType     qtActual  = actualArg->getType();
    std::string formalTypeName          = qtFormal.getAsString();

    std::string formalName = formalArg->getNameAsString();
    std::string actualText = kslicer::GetRangeSourceCode(actualArg->getSourceRange(), a_compiler);
    if(actualText.find(".data()") != std::string::npos) 
      actualText = actualText.substr(0, actualText.find(".data()"));
    
    for(const auto& argOfCurrKernel : k.args)
    {
      if(argOfCurrKernel.name == actualText)
      {
        ArgMatch arg;
        arg.formal    = formalName;
        arg.actual    = actualText;
        arg.type      = formalTypeName;
        arg.argId     = i;
        arg.isPointer = (qtFormal->isPointerType());
        result.push_back(arg);
      }
    }
    
    for(const auto& container : k.usedContainers)
    {
      if(container.second.name == actualText)
      {
        ArgMatch arg;
        arg.formal    = formalName;
        arg.actual    = actualText;
        arg.type      = formalTypeName;
        arg.argId     = i;
        arg.isPointer = (qtFormal->isPointerType());
        result.push_back(arg);
      }
    }
  }
  
  return result;
}


class ArgMatcher : public clang::RecursiveASTVisitor<ArgMatcher>
{
public:
  
  ArgMatcher(const clang::CompilerInstance& a_compiler, kslicer::MainClassInfo& a_codeInfo, std::vector< std::unordered_map<std::string, std::string> >& a_match, std::string a_funcName) : 
             m_compiler(a_compiler), m_sm(a_compiler.getSourceManager()), m_patternImpl(a_codeInfo), m_argMatch(a_match), m_currFuncName(a_funcName)
  { 
        
  }

  bool VisitCallExpr(clang::CallExpr* call)
  {
    const clang::FunctionDecl* fDecl = call->getDirectCallee();  
    if(fDecl == nullptr)             // definitely can't process nullpointer 
      return true;
    
    std::string fname = fDecl->getNameInfo().getName().getAsString();
    if(fname != m_currFuncName)
      return true;
    
    std::unordered_map<std::string, std::string> agrMap;
    for(size_t i=0;i<call->getNumArgs();i++)
    {
      const clang::ParmVarDecl* formalArg = fDecl->getParamDecl(i);
      const clang::Expr*        actualArg = call->getArg(i);
      const clang::QualType     qtFormal  = formalArg->getType();
      //const clang::QualType     qtActual  = actualArg->getType();
      std::string formalTypeName          = qtFormal.getAsString();

      bool supportedContainerType = (formalTypeName.find("Texture") != std::string::npos) || 
                                    (formalTypeName.find("Image") != std::string::npos)   || 
                                    (formalTypeName.find("vector") != std::string::npos)  || 
                                    (formalTypeName.find("Sampler") != std::string::npos) || 
                                    (formalTypeName.find("sampler") != std::string::npos);

      if(qtFormal->isPointerType() || (qtFormal->isReferenceType() && supportedContainerType))
      {
        std::string formalName = formalArg->getNameAsString();
        std::string actualText = kslicer::GetRangeSourceCode(actualArg->getSourceRange(), m_compiler);
        if(actualText.find(".data()") != std::string::npos) 
          actualText = actualText.substr(0, actualText.find(".data()"));
        //TODO: for pointers we should support pointrer arithmatic here ... or not?
        agrMap[formalName] = actualText;
      }
    }
    
    if(!agrMap.empty())
      m_argMatch.push_back(agrMap);

    return true;
  }

private:
  const clang::CompilerInstance& m_compiler;
  const clang::SourceManager&    m_sm;
  kslicer::MainClassInfo&        m_patternImpl;
  std::vector< std::unordered_map<std::string, std::string> >& m_argMatch; 
  std::string                    m_currFuncName;
};


std::vector< std::unordered_map<std::string, std::string> > kslicer::ArgMatchTraversal(kslicer::KernelInfo* pKernel, const kslicer::FuncData& a_funcData, const std::vector<kslicer::FuncData>& a_otherMambers,
                                                                                       MainClassInfo& a_codeInfo, const clang::CompilerInstance& a_compiler)
{
  std::vector< std::unordered_map<std::string, std::string> > result;
  ArgMatcher visitor(a_compiler, a_codeInfo, result, a_funcData.name);
  visitor.TraverseDecl(const_cast<clang::CXXMethodDecl*>(pKernel->astNode));
  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class DeclExtractor : public clang::RecursiveASTVisitor<DeclExtractor>
{
public:
  
  DeclExtractor(const clang::CompilerInstance& a_compiler, const std::vector<kslicer::DeclInClass>& a_listedNames) : m_compiler(a_compiler), m_sm(a_compiler.getSourceManager())
  { 
    for(const auto& decl : a_listedNames)
      usedDecls[decl.name] = decl;
  }

  std::unordered_map<std::string, kslicer::DeclInClass> usedDecls;

  bool VisitCXXRecordDecl(clang::CXXRecordDecl* record)
  {
    const auto pType = record->getTypeForDecl(); 
    const auto qt    = pType->getLocallyUnqualifiedSingleStepDesugaredType();
    const std::string typeName = qt.getAsString();

    const auto ddPos = typeName.find("::");
    if(ddPos == std::string::npos)
      return true;

    const std::string key = typeName.substr(ddPos+2);
    auto p = usedDecls.find(key);
    if(p != usedDecls.end())
    {
      p->second.srcRange  = record->getSourceRange();
      p->second.srcHash   = kslicer::GetHashOfSourceRange( p->second.srcRange);
      p->second.extracted = true;
    }

    return true;
  } 

  bool VisitVarDecl(clang::VarDecl* var)
  {
    if(var->isImplicit() || !var->isConstexpr())
      return true;

    const clang::QualType qt = var->getType();
    const auto typePtr = qt.getTypePtr(); 
    if(typePtr->isPointerType())
      return true;
    
    auto p = usedDecls.find(var->getNameAsString());
    if(p != usedDecls.end())
    {
      const clang::Expr* initExpr = var->getAnyInitializer();
      if(initExpr != nullptr)
      {
        p->second.srcRange  = initExpr->getSourceRange();
        p->second.srcHash   = kslicer::GetHashOfSourceRange(p->second.srcRange);
        p->second.extracted = true;
      }
    }

    return true;
  }

  bool VisitTypedefDecl(clang::TypedefDecl* tDecl)
  { 
    auto p = usedDecls.find(tDecl->getNameAsString());
    if(p != usedDecls.end())
    {
      //const std::string typeName = qt2.getAsString(); 
      //const auto qt2 = tDecl->getUnderlyingType();
      //const std::string typeName1 = tDecl->getNameAsString();
      p->second.srcRange  = tDecl->getSourceRange();
      p->second.srcHash   = kslicer::GetHashOfSourceRange(p->second.srcRange);
      p->second.extracted = true;
    }
    return true;
  }

private:
  const clang::CompilerInstance& m_compiler;
  const clang::SourceManager&    m_sm;
};


std::vector<kslicer::DeclInClass> kslicer::ExtractUsedTC(const std::vector<kslicer::DeclInClass>& a_listedNames, const clang::CXXRecordDecl* classAstNode, const clang::CompilerInstance& a_compiler)
{
  DeclExtractor visitor(a_compiler, a_listedNames);
  visitor.TraverseDecl(const_cast<clang::CXXRecordDecl*>(classAstNode));

  std::vector<kslicer::DeclInClass> result(a_listedNames.size());
  for(const auto& decl : visitor.usedDecls)
    result[decl.second.order] = decl.second;

  return result;
}

const char* GetClangToolingErrorCodeMessage(int code);

std::vector<kslicer::DeclInClass> kslicer::ExtractTCFromClass(const std::string& a_className, const clang::CXXRecordDecl* classAstNode,
                                                              const clang::CompilerInstance& compiler, clang::tooling::ClangTool& Tool)
{
  auto structMatcher = kslicer::MakeMatch_StructDeclInsideClass(a_className);
  auto varMatcher    = kslicer::MakeMatch_VarDeclInsideClass(a_className);
  auto tpdefMatcher  = kslicer::MakeMatch_TypedefInsideClass(a_className);
    
  clang::ast_matchers::MatchFinder finder;
  kslicer::TC_Extractor typeAndConstantsHandler(compiler);
  finder.addMatcher(clang::ast_matchers::traverse(clang::TK_IgnoreUnlessSpelledInSource, structMatcher), &typeAndConstantsHandler);
  finder.addMatcher(clang::ast_matchers::traverse(clang::TK_IgnoreUnlessSpelledInSource, varMatcher),    &typeAndConstantsHandler);
  finder.addMatcher(clang::ast_matchers::traverse(clang::TK_IgnoreUnlessSpelledInSource, tpdefMatcher),  &typeAndConstantsHandler);

  auto res = Tool.run(clang::tooling::newFrontendActionFactory(&finder).get());
  std::cout << "  [TC_Extractor]: end process constants and structs:\t" << GetClangToolingErrorCodeMessage(res) << std::endl;
  
  std::vector<kslicer::DeclInClass> usedDecls;
  usedDecls.reserve(typeAndConstantsHandler.foundDecl.size());
  for(const auto decl : typeAndConstantsHandler.foundDecl)
    usedDecls.push_back(decl.second);
  
  std::sort(usedDecls.begin(), usedDecls.end(), [](const auto& a, const auto& b) { return a.order < b.order; } );
  return kslicer::ExtractUsedTC(usedDecls, classAstNode, compiler);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<const kslicer::KernelInfo*> kslicer::extractUsedKernelsByName(const std::unordered_set<std::string>& a_usedNames, const std::unordered_map<std::string, KernelInfo>& a_kernels)
{
  std::vector<const kslicer::KernelInfo*> result;
  result.reserve(16);
  for(const auto& name : a_usedNames)
  {
    auto p = a_kernels.find(name);
    if(p != a_kernels.end())
      result.push_back(&p->second);
  }
  return result;
}

kslicer::DATA_KIND kslicer::GetKindOfType(const clang::QualType qt, bool isContainer)
{
  DATA_KIND kind = DATA_KIND::KIND_UNKNOWN;
  if(kslicer::IsTexture(qt))           // TODO: detect other cases
    kind = kslicer::DATA_KIND::KIND_TEXTURE;
  else if(kslicer::IsAccelStruct(qt))
    kind = kslicer::DATA_KIND::KIND_ACCEL_STRUCT;
  else if(qt->isPointerType())
    kind = kslicer::DATA_KIND::KIND_POINTER;
  //else if(qt->isPODType())
  //  kind = kslicer::DATA_KIND::KIND_POD;
  else if(isContainer)
    kind = kslicer::DATA_KIND::KIND_VECTOR; 
  else 
    kind = kslicer::DATA_KIND::KIND_POD; 
  return kind;
}
