#include "extractor.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Decl.h"

#include <queue>

class FuncExtractor : public clang::RecursiveASTVisitor<FuncExtractor>
{
public:
  
  FuncExtractor(const clang::CompilerInstance& a_compiler, kslicer::MainClassInfo& a_codeInfo) : m_compiler(a_compiler), m_sm(a_compiler.getSourceManager()), m_patternImpl(a_codeInfo)
  { 
    
  }
  
  std::string currProcessedFuncName;

  bool VisitCallExpr(clang::CallExpr* call)
  {
    clang::FunctionDecl* f = call->getDirectCallee();
    if(f == nullptr)
      return true;

    if(f->isOverloadedOperator())
      return true;

    std::string fileName = std::string(m_sm.getFilename(f->getSourceRange().getBegin())); // check that we are in test_class.cpp or test_class.h or sms like that;                                                                             
    if(fileName.find("include/") != std::string::npos)                       // definitely exclude everything from 'include/' folder
      return true;

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
    usedFunctions[func.srcHash] = func;

    if(func.isKernel)
    {
      auto beginLoc = func.srcRange.getBegin();
      std::string fileName = std::string(m_sm.getFilename(beginLoc));
      std::cout << "[FuncExtractor] ERROR! " << currProcessedFuncName.c_str() << " --> " << func.name.c_str() << std::endl; 
      std::cout << "[FuncExtractor] file:  " << fileName.c_str() << ", line: " << m_sm.getPresumedLoc(beginLoc).getLine() << std::endl;
      std::cout << "[FuncExtractor] calling kernel from a kernel is not allowed currently!" << std::endl;
    }

    return true;
  }

  std::unordered_map<uint64_t, kslicer::FuncData> usedFunctions;

private:
  const clang::SourceManager&    m_sm;
  const clang::CompilerInstance& m_compiler;
  kslicer::MainClassInfo&        m_patternImpl;

};


std::vector<kslicer::FuncData> kslicer::ExtractUsedFunctions(MainClassInfo& a_codeInfo, const clang::CompilerInstance& a_compiler)
{
  std::queue<FuncData> functionsToProcess; 

  for(const auto& kernel : a_codeInfo.kernels)
  {
    kslicer::FuncData func;
    func.name     = kernel.name;
    func.astNode  = kernel.astNode;
    func.srcRange = kernel.astNode->getSourceRange();
    func.srcHash  = GetHashOfSourceRange(func.srcRange);
    func.isMember = true; //isa<clang::CXXMethodDecl>(kernel.astNode);
    func.isKernel = true;
    func.depthUse = 0;
    functionsToProcess.push(func);
  }

  std::unordered_map<uint64_t, FuncData> usedFunctions;
  std::unordered_map<uint64_t, FuncData> usedMembers;
  usedFunctions.reserve(functionsToProcess.size()*10);
  usedMembers.reserve(functionsToProcess.size()*2);
  
  FuncExtractor visitor(a_compiler, a_codeInfo); // first traverse kernels to get first level of used functions
  while(!functionsToProcess.empty())
  {
    auto currFunc = functionsToProcess.front(); functionsToProcess.pop();
    if(!currFunc.isKernel && !currFunc.isMember)
      usedFunctions[currFunc.srcHash] = currFunc;
    else if(!currFunc.isKernel && currFunc.isMember)
      usedMembers[currFunc.srcHash] = currFunc; 
    
    visitor.currProcessedFuncName = currFunc.name;
    visitor.TraverseDecl(const_cast<clang::FunctionDecl*>(currFunc.astNode));

    for(auto& f : visitor.usedFunctions)
    {
      auto nextFunc     = f.second;
      nextFunc.depthUse = currFunc.depthUse + 1;
      functionsToProcess.push(nextFunc);      
    }
    
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

  const clang::SourceManager&    m_sm;
  const clang::CompilerInstance& m_compiler;

};


std::vector<kslicer::DeclInClass> kslicer::ExtractUsedTC(const std::vector<kslicer::DeclInClass>& a_listedNames, MainClassInfo& a_codeInfo, const clang::CompilerInstance& a_compiler)
{
  DeclExtractor visitor(a_compiler, a_listedNames);
  visitor.TraverseDecl(const_cast<clang::CXXRecordDecl*>(a_codeInfo.mainClassASTNode));

  std::vector<kslicer::DeclInClass> result(a_listedNames.size());
  for(const auto& decl : visitor.usedDecls)
    result[decl.second.order] = decl.second;

  return result;
}