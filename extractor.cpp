#include "extractor.h"
#include "clang/AST/RecursiveASTVisitor.h"

#include <queue>

struct FuncData
{
  const clang::FunctionDecl* astNode;
  std::string        name;
  clang::SourceRange srcRange;
  uint64_t           srcHash;
  bool               isMember = false;
  bool               isKernel = false;
};

class FuncExtractor : public clang::RecursiveASTVisitor<FuncExtractor>
{
public:
  
  FuncExtractor(const clang::CompilerInstance& a_compiler) : m_compiler(a_compiler), m_sm(a_compiler.getSourceManager())
  { 
    
  }
  
  bool VisitCallExpr(clang::CallExpr* call)
  {
    clang::FunctionDecl* f = call->getDirectCallee();
    if(f == nullptr)
      return true;

    if(f->isOverloadedOperator())
      return true;

    FuncData func;
    func.name     = f->getNameAsString();
    func.astNode  = f;
    func.srcRange = f->getSourceRange(); 
    func.srcHash  = kslicer::GetHashOfSourceRange(func.srcRange);
    func.isMember = clang::isa<clang::CXXMethodDecl>(func.astNode);
    func.isKernel = false;                                           // TODO: add check here with pattern implementation functions IsKernel
    usedFunctions[func.srcHash] = func;
    return true;
  }

  std::unordered_map<uint64_t, FuncData> usedFunctions;

private:
  const clang::SourceManager&    m_sm;
  const clang::CompilerInstance& m_compiler;

};


void kslicer::ExtractUsedCode(MainClassInfo& a_codeInfo, std::unordered_map<std::string, clang::SourceRange>& a_usedFunctions, const clang::CompilerInstance& a_compiler)
{
  std::queue<FuncData> functionsToProcess; 

  for(const auto& kernel : a_codeInfo.kernels)
  {
    FuncData func;
    func.name     = kernel.name;
    func.astNode  = kernel.astNode;
    func.srcRange = kernel.astNode->getSourceRange();
    func.srcHash  = GetHashOfSourceRange(func.srcRange);
    func.isMember = true; //isa<clang::CXXMethodDecl>(kernel.astNode);
    func.isKernel = true;
    functionsToProcess.push(func);
  }

  std::unordered_map<uint64_t, FuncData> usedFunctions;
  std::unordered_map<uint64_t, FuncData> usedMembers;
  usedFunctions.reserve(functionsToProcess.size()*10);
  usedMembers.reserve(functionsToProcess.size()*2);
  
  FuncExtractor visitor(a_compiler); // first traverse kernels to get first level of used functions
  while(!functionsToProcess.empty())
  {
    auto currFunc = functionsToProcess.front(); functionsToProcess.pop();
    if(!currFunc.isKernel && !currFunc.isMember)
      usedFunctions[currFunc.srcHash] = currFunc;
    else if(!currFunc.isKernel && currFunc.isMember)
      usedMembers[currFunc.srcHash] = currFunc; 

    visitor.TraverseDecl(const_cast<clang::FunctionDecl*>(currFunc.astNode));

    usedFunctions.merge(visitor.usedFunctions);
    visitor.usedFunctions.clear();
  }

  int a = 2;

}