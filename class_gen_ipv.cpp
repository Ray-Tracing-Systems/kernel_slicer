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
      arg2.typeName = found->type;
      arg2.id       = found - a_kernel.loopIters.begin();
      args.push_back(arg2);
    }
  }

  std::sort(args.begin(), args.end(), [](const auto& a, const auto & b) { return a.id < b.id; });

  return args;
}

std::string kslicer::IPV_Pattern::VisitAndRewrite_CF(MainFuncInfo& a_mainFunc, clang::CompilerInstance& compiler)
{
  const std::string&   a_mainClassName = this->mainClassName;
  auto&                a_outDsInfo     = this->allDescriptorSetsInfo;

  const CXXMethodDecl* a_node          = a_mainFunc.Node;
  const std::string&   a_mainFuncName  = a_mainFunc.Name;
  std::string&         a_outFuncDecl   = a_mainFunc.GeneratedDecl;

  const auto  inOutParamList = kslicer::ListPointerParamsOfMainFunc(a_node);

  Rewriter rewrite2;
  rewrite2.setSourceMgr(compiler.getSourceManager(), compiler.getLangOpts());

  a_mainFunc.startDSNumber = a_outDsInfo.size();

  kslicer::MainFuncASTVisitor rv(rewrite2, compiler, a_mainFunc, inOutParamList, this);

  rv.m_kernCallTypes = a_outDsInfo;
  rv.TraverseDecl(const_cast<clang::CXXMethodDecl*>(a_node));
  
  clang::SourceLocation b(a_node->getBeginLoc()), _e(a_node->getEndLoc());
  clang::SourceLocation e(clang::Lexer::getLocForEndOfToken(_e, 0, compiler.getSourceManager(), compiler.getLangOpts()));
  
  std::string sourceCode = rewrite2.getRewrittenText(clang::SourceRange(b,e));
  
  // (1) TestClass::MainFuncCmd --> TestClass_Generated::MainFuncCmd and add input command Buffer as first argument
  // 
  const std::string replaceFrom = a_mainClassName + "::" + rv.mainFuncCmdName;
  const std::string replaceTo   = a_mainClassName + "_Generated" + "::" + rv.mainFuncCmdName;

  assert(ReplaceFirst(sourceCode, replaceFrom, replaceTo));
  assert(ReplaceFirst(sourceCode, "(", "(VkCommandBuffer a_commandBuffer, "));

  // (3) set m_currCmdBuffer with input command bufer and add other prolog to MainFunCmd
  //
  {
    std::stringstream strOut;
    strOut << "{" << std::endl;
    strOut << "  m_currCmdBuffer = a_commandBuffer;" << std::endl;
    strOut << "  const uint32_t outOfForFlags  = KGEN_FLAG_RETURN;" << std::endl;
    strOut << "  const uint32_t inForFlags     = KGEN_FLAG_RETURN | KGEN_FLAG_BREAK;" << std::endl;
    if(a_mainFunc.needToAddThreadFlags)
    {
      strOut << "  const uint32_t outOfForFlagsN = KGEN_FLAG_RETURN | KGEN_FLAG_SET_EXIT_NEGATIVE;" << std::endl;
      strOut << "  const uint32_t inForFlagsN    = KGEN_FLAG_RETURN | KGEN_FLAG_BREAK | KGEN_FLAG_SET_EXIT_NEGATIVE;" << std::endl;
      strOut << "  const uint32_t outOfForFlagsD = KGEN_FLAG_RETURN | KGEN_FLAG_DONT_SET_EXIT;" << std::endl;
      strOut << "  const uint32_t inForFlagsD    = KGEN_FLAG_RETURN | KGEN_FLAG_BREAK | KGEN_FLAG_DONT_SET_EXIT;" << std::endl;
      strOut << "  vkCmdFillBuffer(a_commandBuffer, " << a_mainFunc.Name.c_str() << "_local.threadFlagsBuffer , 0, VK_WHOLE_SIZE, 0); // zero thread flags, mark all threads to be active" << std::endl;
      strOut << "  VkMemoryBarrier fillBarrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT }; " << std::endl;
      strOut << "  vkCmdPipelineBarrier(a_commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_DEPENDENCY_BY_REGION_BIT, 1, &fillBarrier, 0, nullptr, 0, nullptr); " << std::endl;
    }
    strOut << std::endl;

    size_t bracePos = sourceCode.find("{");
    sourceCode = (sourceCode.substr(0, bracePos) + strOut.str() + sourceCode.substr(bracePos+2)); 
  }

  // (4) get function decl from full function code
  //
  std::string mainFuncDecl = sourceCode.substr(0, sourceCode.find(")")+1) + ";";
  assert(ReplaceFirst(mainFuncDecl, a_mainClassName + "_Generated" + "::", ""));

  a_outFuncDecl = "virtual " + mainFuncDecl;
  a_outDsInfo   = rv.m_kernCallTypes;

  a_mainFunc.endDSNumber = a_outDsInfo.size();

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


class LoopHandlerInsideKernelsIPV : public kslicer::VariableAndFunctionFilter
{
public:
  explicit LoopHandlerInsideKernelsIPV(std::ostream& s, kslicer::MainClassInfo& a_allInfo, kslicer::KernelInfo* a_currKernel, const clang::CompilerInstance& a_compiler) : 
                                       VariableAndFunctionFilter(s, a_allInfo, a_currKernel, a_compiler)
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
          tidArg.name       = initVar->getNameAsString();
          tidArg.type       = qt.getAsString();
          tidArg.sizeExpr   = kslicer::GetRangeSourceCode(loopSZ->getSourceRange(), m_compiler);
          currKernel->loopIters.push_back(tidArg);
          currKernel->loopInsides = forLoop->getBody()->getSourceRange();
        }
      }
    }
    else
      VariableAndFunctionFilter::run(result);

    return;
  }  // run


  uint32_t m_maxNesting = 0;
};


kslicer::IPV_Pattern::MHandlerKFPtr kslicer::IPV_Pattern::MatcherHandler_KF(KernelInfo& kernel, const clang::CompilerInstance& a_compile)
{
  return std::move(std::make_unique<LoopHandlerInsideKernelsIPV>(std::cout, *this, &kernel, a_compile));
}

std::string kslicer::IPV_Pattern::VisitAndRewrite_KF(const KernelInfo& a_funcInfo, const clang::CompilerInstance& compiler)
{
  Rewriter rewrite2;
  rewrite2.setSourceMgr(compiler.getSourceManager(), compiler.getLangOpts());

  kslicer::KernelReplacerASTVisitor rv(rewrite2, compiler, this->mainClassName, this->dataMembers, a_funcInfo.args, "", a_funcInfo.isBoolTyped);
  rv.TraverseDecl(const_cast<clang::CXXMethodDecl*>(a_funcInfo.astNode));
  
  return rewrite2.getRewrittenText(a_funcInfo.loopInsides);
}
