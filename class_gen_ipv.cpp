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

void kslicer::IPV_Pattern::ProcessKernelArg(KernelInfo::ArgInfo& arg, const KernelInfo& a_kernel) const 
{
  auto found = std::find_if(a_kernel.loopIters.begin(), a_kernel.loopIters.end(), 
                           [&](const auto& val){ return arg.name == val.sizeText; });
  arg.isLoopSize = (found != a_kernel.loopIters.end());
}

std::vector<kslicer::ArgFinal> kslicer::IPV_Pattern::GetKernelTIDArgs(const KernelInfo& a_kernel) const 
{
  std::vector<kslicer::ArgFinal> args;
  for (uint32_t i = 0; i < a_kernel.loopIters.size(); i++) 
  {    
    const auto& arg = a_kernel.loopIters[i];
    ArgFinal arg2;
    arg2.name        = arg.name;
    arg2.type        = pShaderFuncRewriter->RewriteStdVectorTypeStr(arg.type);
    arg2.loopIter    = arg;
    arg2.loopIter.id = i;
    args.push_back(arg2);
  }

  std::sort(args.begin(), args.end(), [](const auto& a, const auto & b) { return a.loopIter.id < b.loopIter.id; });

  return args;
}

void kslicer::IPV_Pattern::VisitAndRewrite_CF(MainFuncInfo& a_mainFunc, clang::CompilerInstance& compiler)
{
  //const std::string&   a_mainClassName = this->mainClassName;
  //const CXXMethodDecl* a_node          = a_mainFunc.Node;
  //const std::string&   a_mainFuncName  = a_mainFunc.Name;
  GetCFSourceCodeCmd(a_mainFunc, compiler, false); // ==> write this->allDescriptorSetsInfo, a_mainFunc // TODO: may simplify impl for image processing 
  a_mainFunc.endDSNumber   = allDescriptorSetsInfo.size();
  a_mainFunc.InOuts        = kslicer::ListParamsOfMainFunc(a_mainFunc.Node, compiler);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

kslicer::IPV_Pattern::MList kslicer::IPV_Pattern::ListMatchers_CF(const std::string& mainFuncName)
{
  std::vector<clang::ast_matchers::StatementMatcher> list;
  list.push_back(kslicer::MakeMatch_LocalVarOfMethod(mainFuncName));
  list.push_back(kslicer::MakeMatch_MemberVarOfMethod(mainFuncName));
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
  list.push_back(kslicer::MakeMatch_BeforeForLoopInsideFunction(a_kernelName));
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

  std::string GetStrideText(const clang::Expr* a_expr)
  {
    //a_expr->dump();
    if(clang::isa<clang::UnaryOperator>(a_expr))
    {  
      auto uop = clang::dyn_cast<clang::UnaryOperator>(a_expr);
      std::string opStr = std::string(uop->getOpcodeStr(uop->getOpcode()));
      if(opStr == "--")
        return "(-1)";
      else
        return "1";
    }

    //#TODO: process += and -= expressinons, but first we must fix ast matchers for that 

    return "1";
  }

  void run(clang::ast_matchers::MatchFinder::MatchResult const & result) override
  {
    using namespace clang;
    const FunctionDecl* func_decl = result.Nodes.getNodeAs<FunctionDecl>("targetFunction");
    const ForStmt*      forLoop   = result.Nodes.getNodeAs<ForStmt>("loop");
    const CompoundStmt* loopOutsidesInit  = result.Nodes.getNodeAs<CompoundStmt>("loopInitCode");

    //clang::SourceManager& srcMgr(const_cast<clang::SourceManager &>(result.Context->getSourceManager()));

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
          
          tidArg.sizeText    = kslicer::GetRangeSourceCode(loopSZ->getSourceRange(), m_compiler);
          tidArg.startText   = kslicer::GetRangeSourceCode(initVar->getAnyInitializer()->getSourceRange(), m_compiler); 
          tidArg.strideText  = GetStrideText(forLoop->getInc()); //kslicer::GetRangeSourceCode(forLoop->getInc()->getSourceRange(), m_compiler);
          tidArg.condTextOriginal = kslicer::GetRangeSourceCode(forLoop->getCond()->getSourceRange(), m_compiler);
          tidArg.iterTextOriginal = kslicer::GetRangeSourceCode(forLoop->getInc()->getSourceRange(), m_compiler);
          //tidArg.startRange  = initVar->getAnyInitializer()->getSourceRange();
          //tidArg.sizeRange   = loopSZ->getSourceRange();
          //tidArg.strideRange = forLoop->getInc()->getSourceRange();
          //tidArg.startNode   = initVar->getAnyInitializer();
          //tidArg.sizeNode    = loopSZ;
          //tidArg.strideNode  = forLoop->getInc();

          tidArg.loopNesting = uint32_t(currKernel->loopIters.size());
          currKernel->loopIters.push_back(tidArg);
          currKernel->loopInsides = forLoop->getBody()->getSourceRange();
        }
      }
    }
    else if(loopOutsidesInit && func_decl)
    {
      currKernel->loopOutsidesInit    = loopOutsidesInit->getSourceRange();
      //auto debugMe = kslicer::GetRangeSourceCode(loopOutsidesInit->getSourceRange(), m_compiler);
      //std::cout << "debugMe = " << debugMe.c_str() << std::endl;
      clang::SourceLocation endOfInit = currKernel->loopOutsidesInit.getEnd();
      auto p = loopOutsidesInit->body_begin();
      for(; p != loopOutsidesInit->body_end(); ++p)
      {
        const Stmt* expr = *p; 
        if(isa<ForStmt>(expr))                       // TODO: CHECK THIS IS EXACTLY THE 'for' WE NEED! HOW?
          break;                                     // TODO: REMEMBER 'for' LOCATION IN SOME VARIABLE AND IGNORE OTHER 'fors' INSIDE 'if(forLoop && func_decl)'        
        endOfInit = expr->getSourceRange().getEnd(); // TODO: WHICH ARE NOT INSIDE THIS FOR AND NOT THIS FOR ITSELF.
      }

      currKernel->hasInitPass = (currKernel->loopOutsidesInit.getEnd() != endOfInit);
      currKernel->loopOutsidesInit.setEnd(endOfInit);
      
      ++p;
      if(p != loopOutsidesInit->body_end())
      {
        const Stmt* beginOfEnd = *p;
        currKernel->loopOutsidesFinish.setBegin(beginOfEnd->getSourceRange().getBegin());
        p = loopOutsidesInit->body_end(); --p;
        const Stmt* endOfFinish = *p;
        currKernel->loopOutsidesFinish.setEnd(endOfFinish->getSourceRange().getEnd());
        currKernel->hasFinishPassSelf = (currKernel->loopOutsidesFinish.getBegin() != currKernel->loopOutsidesFinish.getEnd());
        //auto debugMeTail = kslicer::GetRangeSourceCode(currKernel->loopOutsidesFinish, m_compiler);
        //std::cout << "debugMeTail = " << debugMeTail.c_str() << std::endl;
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

std::string kslicer::IPV_Pattern::VisitAndRewrite_KF(KernelInfo& a_funcInfo, const clang::CompilerInstance& compiler, 
                                                     std::string& a_outLoopInitCode, std::string& a_outLoopFinishCode)
{
  //a_funcInfo.astNode->dump();
  
  Rewriter rewrite2;
  rewrite2.setSourceMgr(compiler.getSourceManager(), compiler.getLangOpts());
  
  auto pVisitor = pShaderCC->MakeKernRewriter(rewrite2, compiler, this, a_funcInfo, "", false);
  pVisitor->TraverseDecl(const_cast<clang::CXXMethodDecl*>(a_funcInfo.astNode));
  
  a_funcInfo.shaderFeatures = a_funcInfo.shaderFeatures || pVisitor->GetKernelShaderFeatures(); // TODO: dont work !!!
  
  if(a_funcInfo.loopOutsidesInit.isValid())
    a_outLoopInitCode   = rewrite2.getRewrittenText(a_funcInfo.loopOutsidesInit)   + ";";

  if(a_funcInfo.loopOutsidesFinish.isValid())  
    a_outLoopFinishCode = rewrite2.getRewrittenText(a_funcInfo.loopOutsidesFinish) + ";";

  return rewrite2.getRewrittenText(a_funcInfo.loopInsides) + ";";
}

void kslicer::IPV_Pattern::VisitAndPrepare_KF(KernelInfo& a_funcInfo, const clang::CompilerInstance& compiler)
{
  //a_funcInfo.astNode->dump();
  Rewriter rewrite2;
  rewrite2.setSourceMgr(compiler.getSourceManager(), compiler.getLangOpts());

  auto pVisitor = pShaderCC->MakeKernRewriter(rewrite2, compiler, this, a_funcInfo, "", true);
  pVisitor->TraverseDecl(const_cast<clang::CXXMethodDecl*>(a_funcInfo.astNode));

  // check kernel
  //
  if(a_funcInfo.loopIters.size() == 1)
  {
    auto pos = a_funcInfo.name.find("kernel1D_");
    if(pos == std::string::npos)
      std::cout << "  [KF, ERROR]: wrong naming for 1D kernels, 'kernel1D_' should be used for kernel " << a_funcInfo.name.c_str() << std::endl;
  }
  else if (a_funcInfo.loopIters.size() == 2)
  {
    auto pos = a_funcInfo.name.find("kernel2D_");
    if(pos == std::string::npos)
      std::cout << "  [KF, ERROR]: wrong naming for 2D kernels, 'kernel2D_' should be used for kernel " << a_funcInfo.name.c_str() << std::endl;
  }
  else if (a_funcInfo.loopIters.size() == 3)
  {
    auto pos = a_funcInfo.name.find("kernel3D_");
    if(pos == std::string::npos)
      std::cout << "  [KF, ERROR]: wrong naming for 3D kernels, 'kernel3D_' should be used for kernel " << a_funcInfo.name.c_str() << std::endl;
  }
  else
  {
    std::cout << "  [KF, ERROR]: wrong loop nesting " << a_funcInfo.loopIters.size() << " for kernel " << a_funcInfo.name.c_str() << std::endl;
  }
  

}

