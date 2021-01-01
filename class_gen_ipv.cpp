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

  kslicer::MainFuncASTVisitor rv(rewrite2, compiler, a_mainFunc, inOutParamList, this->allDataMembers, this->kernels);

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

kslicer::IPV_Pattern::MHandlerPtr kslicer::IPV_Pattern::MatcherHandler_CF(kslicer::MainFuncInfo& a_mainFuncRef, const clang::ASTContext& a_astContext)
{
  return std::move(std::make_unique<kslicer::MainFuncAnalyzerRT>(std::cout, *this, a_astContext, a_mainFuncRef));
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
  return list;
}
