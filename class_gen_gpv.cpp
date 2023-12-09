#include "kslicer.h"
#include "class_gen.h"
#include "ast_matchers.h"

#include <sstream>
#include <algorithm>

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::string kslicer::GPV_Pattern::RemoveKernelPrefix(const std::string& a_funcName) const { return a_funcName; }
uint32_t kslicer::GPV_Pattern::GetKernelDim(const kslicer::KernelInfo& a_kernel) const { return 1; } 

bool kslicer::GPV_Pattern::IsKernel(const std::string& a_funcName) const ///<! return true if function is a kernel
{
  auto pos1 = a_funcName.find("vertex_shader");
  auto pos2 = a_funcName.find("pixel_shader");
  return (pos1 != std::string::npos) || (pos2 != std::string::npos); 
}

void kslicer::GPV_Pattern::ProcessKernelArg(KernelInfo::ArgInfo& arg, const KernelInfo& a_kernel) const 
{

}

std::vector<kslicer::ArgFinal> kslicer::GPV_Pattern::GetKernelTIDArgs(const KernelInfo& a_kernel) const 
{
  std::vector<kslicer::ArgFinal> args;
  return args;
}

void kslicer::GPV_Pattern::VisitAndRewrite_CF(MainFuncInfo& a_mainFunc, clang::CompilerInstance& compiler)
{
  //GetCFSourceCodeCmd(a_mainFunc, compiler, false); // ==> write this->allDescriptorSetsInfo, a_mainFunc // TODO: may simplify impl for image processing 
  //a_mainFunc.endDSNumber   = allDescriptorSetsInfo.size();
  //a_mainFunc.InOuts        = kslicer::ListParamsOfMainFunc(a_mainFunc.Node, compiler);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

kslicer::GPV_Pattern::MList kslicer::GPV_Pattern::ListMatchers_CF(const std::string& mainFuncName)
{
  std::vector<clang::ast_matchers::StatementMatcher> list;
  //list.push_back(kslicer::MakeMatch_LocalVarOfMethod(mainFuncName));
  //list.push_back(kslicer::MakeMatch_MemberVarOfMethod(mainFuncName));
  //list.push_back(kslicer::MakeMatch_MethodCallFromMethod(mainFuncName));
  //list.push_back(kslicer::MakeMatch_SingleForLoopInsideFunction(mainFuncName));
  //list.push_back(kslicer::MakeMatch_IfInsideForLoopInsideFunction(mainFuncName));
  //list.push_back(kslicer::MakeMatch_FunctionCallInsideForLoopInsideFunction(mainFuncName));
  //list.push_back(kslicer::MakeMatch_IfReturnFromFunction(mainFuncName));
  return list;
}

kslicer::GPV_Pattern::MHandlerCFPtr kslicer::GPV_Pattern::MatcherHandler_CF(kslicer::MainFuncInfo& a_mainFuncRef, const clang::CompilerInstance& a_compiler)
{
  return std::move(std::make_unique<kslicer::MainFuncAnalyzerRT>(std::cout, *this, a_compiler.getASTContext(), a_mainFuncRef)); // ?
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

kslicer::GPV_Pattern::MList kslicer::GPV_Pattern::ListMatchers_KF(const std::string& a_kernelName)
{
  std::vector<clang::ast_matchers::StatementMatcher> list;
  //list.push_back(kslicer::MakeMatch_MemberVarOfMethod(a_kernelName));
  //list.push_back(kslicer::MakeMatch_FunctionCallFromFunction(a_kernelName));
  //list.push_back(kslicer::MakeMatch_ForLoopInsideFunction(a_kernelName));
  //list.push_back(kslicer::MakeMatch_BeforeForLoopInsideFunction(a_kernelName));
  return list;
}

static bool areSameVariable(const clang::ValueDecl *First, const clang::ValueDecl *Second) 
{
  return First && Second && First->getCanonicalDecl() == Second->getCanonicalDecl();
}


class ShaderHandlerInsideIPV : public kslicer::UsedCodeFilter
{
public:
  explicit ShaderHandlerInsideIPV(std::ostream& s, kslicer::MainClassInfo& a_allInfo, kslicer::KernelInfo* a_currKernel, const clang::CompilerInstance& a_compiler) : 
                                  UsedCodeFilter(s, a_allInfo, a_currKernel, a_compiler)
  {
    m_maxNesting = a_allInfo.GetKernelDim(*a_currKernel);
    a_currKernel->loopIters.clear(); 
  } 

  std::string GetStrideText(const clang::Expr* a_expr)
  {
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


kslicer::GPV_Pattern::MHandlerKFPtr kslicer::GPV_Pattern::MatcherHandler_KF(KernelInfo& kernel, const clang::CompilerInstance& a_compile)
{
  return std::move(std::make_unique<ShaderHandlerInsideIPV>(std::cout, *this, &kernel, a_compile));
}

void kslicer::GPV_Pattern::VisitAndPrepare_KF(KernelInfo& a_funcInfo, const clang::CompilerInstance& compiler)
{
  //a_funcInfo.astNode->dump(); //print AST tree. Uses this during debugging!
  Rewriter rewrite2;
  rewrite2.setSourceMgr(compiler.getSourceManager(), compiler.getLangOpts());

  auto pVisitor = pShaderCC->MakeKernRewriter(rewrite2, compiler, this, a_funcInfo, "", true);
  pVisitor->TraverseDecl(const_cast<clang::CXXMethodDecl*>(a_funcInfo.astNode));  
}

std::string kslicer::GPV_Pattern::VisitAndRewrite_KF(KernelInfo& a_funcInfo, const clang::CompilerInstance& compiler, 
                                                     std::string& a_outLoopInitCode, std::string& a_outLoopFinishCode)
{
  //if(a_funcInfo.name == "kernel1D_ArraySumm")
  //  a_funcInfo.astNode->dump();
  
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

