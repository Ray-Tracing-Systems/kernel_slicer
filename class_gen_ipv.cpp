#include "kslicer.h"
#include "class_gen.h"
#include "ast_matchers.h"

#include <sstream>
#include <algorithm>

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::string kslicer::IPV_Pattern::RemoveKernelPrefix(const std::string& a_funcName) const ///<! "kernel2D_XXX" --> "XXX"; 
{
  std::string name = a_funcName;
  if(ReplaceFirst(name, "kernel1D_", "") || ReplaceFirst(name, "kernel2D_", "") || ReplaceFirst(name, "kernel3D_", ""))
    return name;
  else
    return a_funcName;
}

bool kslicer::IPV_Pattern::IsKernel(const std::string& a_funcName) const ///<! return true if function is a kernel
{
  auto pos1 = a_funcName.find("kernel1D_");
  auto pos2 = a_funcName.find("kernel2D_");
  auto pos3 = a_funcName.find("kernel3D_");
  return (pos1 != std::string::npos) || (pos2 != std::string::npos) || (pos3 != std::string::npos); 
}

uint32_t kslicer::IPV_Pattern::GetKernelDim(const kslicer::KernelInfo& a_kernel) const
{
  const std::string& a_funcName = a_kernel.name;
  auto pos1 = a_funcName.find("kernel1D_");
  auto pos2 = a_funcName.find("kernel2D_");
  auto pos3 = a_funcName.find("kernel3D_");

  if(pos1 != std::string::npos)
    return 1;
  else if(pos2 != std::string::npos) 
    return 2;
  else if(pos3 != std::string::npos)
    return 3;
  else
    return 0;
} 

void kslicer::IPV_Pattern::ProcessKernelArg(KernelInfo::Arg& arg, const KernelInfo& a_kernel) const 
{
  auto found = std::find_if(a_kernel.loopIters.begin(), a_kernel.loopIters.end(), 
                           [&](const auto& val){ return arg.name == val.sizeExpr; });
  arg.isLoopSize = (found != a_kernel.loopIters.end());
}

std::vector<kslicer::MainClassInfo::ArgTypeAndNamePair> kslicer::IPV_Pattern::GetKernelTIDArgs(const KernelInfo& a_kernel) const 
{
  std::vector<kslicer::MainClassInfo::ArgTypeAndNamePair> args;
  for (const auto& arg : a_kernel.args) 
  {    
    if(arg.isLoopSize)
    { 
      auto found = std::find_if(a_kernel.loopIters.begin(), a_kernel.loopIters.end(), 
                                [&](const auto& val){ return arg.name == val.sizeExpr; });

      ArgTypeAndNamePair arg2;
      arg2.argName  = found->name;
      arg2.sizeName = found->sizeExpr;
      arg2.typeName = RemoveTypeNamespaces(found->type);
      arg2.id       = found - a_kernel.loopIters.begin();
      args.push_back(arg2);
    }
  }

  std::sort(args.begin(), args.end(), [](const auto& a, const auto & b) { return a.id < b.id; });

  return args;
}

std::string kslicer::IPV_Pattern::VisitAndRewrite_CF(MainFuncInfo& a_mainFunc, clang::CompilerInstance& compiler)
{
  //const std::string&   a_mainClassName = this->mainClassName;
  //const CXXMethodDecl* a_node          = a_mainFunc.Node;
  //const std::string&   a_mainFuncName  = a_mainFunc.Name;

  std::string sourceCode   = GetCFSourceCodeCmd(a_mainFunc, compiler); // ==> write this->allDescriptorSetsInfo // TODO: may simplify impl for image processing
  a_mainFunc.GeneratedDecl = GetCFDeclFromSource(sourceCode); 
  a_mainFunc.endDSNumber   = allDescriptorSetsInfo.size();

  return sourceCode;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

kslicer::IPV_Pattern::MList kslicer::IPV_Pattern::ListMatchers_CF(const std::string& mainFuncName)
{
  std::vector<clang::ast_matchers::StatementMatcher> list;
  list.push_back(kslicer::MakeMatch_LocalVarOfMethod(mainFuncName));
  list.push_back(kslicer::MakeMatch_MethodCallFromMethod(mainFuncName));
  list.push_back(kslicer::MakeMatch_SingleForLoopInsideFunction(mainFuncName));
  list.push_back(kslicer::MakeMatch_IfInsideForLoopInsideFunction(mainFuncName));
  list.push_back(kslicer::MakeMatch_FunctionCallInsideForLoopInsideFunction(mainFuncName));
  list.push_back(kslicer::MakeMatch_IfReturnFromFunction(mainFuncName));
  return list;
}

kslicer::IPV_Pattern::MHandlerCFPtr kslicer::IPV_Pattern::MatcherHandler_CF(kslicer::MainFuncInfo& a_mainFuncRef, const clang::CompilerInstance& a_compiler)
{
  return std::move(std::make_unique<kslicer::MainFuncAnalyzerRT>(std::cout, *this, a_compiler.getASTContext(), a_mainFuncRef));
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

kslicer::IPV_Pattern::MList kslicer::IPV_Pattern::ListMatchers_KF(const std::string& a_kernelName)
{
  std::vector<clang::ast_matchers::StatementMatcher> list;
  list.push_back(kslicer::MakeMatch_MemberVarOfMethod(a_kernelName));
  list.push_back(kslicer::MakeMatch_FunctionCallFromFunction(a_kernelName));
  list.push_back(kslicer::MakeMatch_ForLoopInsideFunction(a_kernelName));
  return list;
}

static bool areSameVariable(const clang::ValueDecl *First, const clang::ValueDecl *Second) 
{
  return First && Second && First->getCanonicalDecl() == Second->getCanonicalDecl();
}


class LoopHandlerInsideKernelsIPV : public kslicer::UsedCodeFilter
{
public:
  explicit LoopHandlerInsideKernelsIPV(std::ostream& s, kslicer::MainClassInfo& a_allInfo, kslicer::KernelInfo* a_currKernel, const clang::CompilerInstance& a_compiler) : 
                                       UsedCodeFilter(s, a_allInfo, a_currKernel, a_compiler)
  {
    m_maxNesting = a_allInfo.GetKernelDim(*a_currKernel);
    a_currKernel->loopIters.clear(); 
  } 

  void run(clang::ast_matchers::MatchFinder::MatchResult const & result) override
  {
    using namespace clang;
    const FunctionDecl* func_decl = result.Nodes.getNodeAs<FunctionDecl>("targetFunction");
    const ForStmt* forLoop        = result.Nodes.getNodeAs<ForStmt>("loop");

    clang::SourceManager& srcMgr(const_cast<clang::SourceManager &>(result.Context->getSourceManager()));

    if(forLoop && func_decl)
    {
      const VarDecl* initVar = result.Nodes.getNodeAs<clang::VarDecl>("initVar");
      const VarDecl* condVar = result.Nodes.getNodeAs<clang::VarDecl>("condVar");
      const VarDecl* incVar  = result.Nodes.getNodeAs<clang::VarDecl>("incVar");
      const Expr*    loopSZ  = result.Nodes.getNodeAs<clang::Expr>   ("loopSize");
      
      if(areSameVariable(initVar,condVar) && areSameVariable(initVar, incVar) && loopSZ)
      {
        std::string name = initVar->getNameAsString();
        //std::cout << "  [LoopHandlerIPV]: Variable name is: " << name.c_str() << std::endl;
        if(currKernel->loopIters.size() < m_maxNesting)
        {
          const clang::QualType qt = initVar->getType();
          kslicer::KernelInfo::LoopIter tidArg;
          tidArg.name        = initVar->getNameAsString();
          tidArg.type        = qt.getAsString();
          tidArg.sizeExpr    = kslicer::GetRangeSourceCode(loopSZ->getSourceRange(), m_compiler);
          tidArg.loopNesting = uint32_t(currKernel->loopIters.size());
          currKernel->loopIters.push_back(tidArg);
          currKernel->loopInsides = forLoop->getBody()->getSourceRange();
        }
      }
    }
    else
      UsedCodeFilter::run(result);

    return;
  }  // run


  uint32_t m_maxNesting = 0;
};


kslicer::IPV_Pattern::MHandlerKFPtr kslicer::IPV_Pattern::MatcherHandler_KF(KernelInfo& kernel, const clang::CompilerInstance& a_compile)
{
  return std::move(std::make_unique<LoopHandlerInsideKernelsIPV>(std::cout, *this, &kernel, a_compile));
}

std::string kslicer::IPV_Pattern::VisitAndRewrite_KF(KernelInfo& a_funcInfo, const clang::CompilerInstance& compiler)
{
  Rewriter rewrite2;
  rewrite2.setSourceMgr(compiler.getSourceManager(), compiler.getLangOpts());

  kslicer::KernelReplacerASTVisitor rv(rewrite2, compiler, this, a_funcInfo, "");
  rv.TraverseDecl(const_cast<clang::CXXMethodDecl*>(a_funcInfo.astNode));
  
  return rewrite2.getRewrittenText(a_funcInfo.loopInsides);
}
