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

void kslicer::GPV_Pattern::ElableSpecialKernels()
{
  kernels["vertex_shader"] = allKernels["vertex_shader"];
  kernels["pixel_shader"]  = allKernels["pixel_shader"];
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

kslicer::GPV_Pattern::MList kslicer::GPV_Pattern::ListMatchers_CF(const std::string& mainFuncName) // we don't have control functions in GP currently
{
  std::vector<clang::ast_matchers::StatementMatcher> list;
  return list;
}

kslicer::GPV_Pattern::MHandlerCFPtr kslicer::GPV_Pattern::MatcherHandler_CF(kslicer::MainFuncInfo& a_mainFuncRef, const clang::CompilerInstance& a_compiler)
{
  return std::move(std::make_unique<kslicer::MainFuncAnalyzerRT>(std::cout, *this, a_compiler.getASTContext(), a_mainFuncRef)); // ?
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

kslicer::GPV_Pattern::MList kslicer::GPV_Pattern::ListMatchers_KF(const std::string& a_kernelName) // we don't AST matchers for GP, but we need sume matchers for 'UsedCodeFilter' class
{
  std::vector<clang::ast_matchers::StatementMatcher> list;
  list.push_back(kslicer::MakeMatch_MemberVarOfMethod(a_kernelName));                              // detect class members
  list.push_back(kslicer::MakeMatch_FunctionCallFromFunction(a_kernelName));                       // detect function calls
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
    a_currKernel->loopIters.clear(); 
  } 

  void run(clang::ast_matchers::MatchFinder::MatchResult const & result) override
  {
    UsedCodeFilter::run(result);                                                                  // we don't AST matchers for GP, but we need sume matchers for 'UsedCodeFilter' class
  } 
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

kslicer::TemplatesPaths kslicer::GPV_Pattern::WhereIsMyTemplates() const
{
  #ifdef WIN32
  const std::string slash = "\\";
  #else
  const std::string slash = "/";
  #endif

  TemplatesPaths res;
  res.classHeader  = "templates_gp" + slash + "gp_class.h";
  res.classCppInit = "templates_gp" + slash + "gp_class_init.cpp";
  res.classCppMain = "templates_gp" + slash + "gp_class.cpp";
  res.classCppDS   = "templates_gp" + slash + "gp_class_ds.cpp";
  res.shaderMain   = "templates_gp" + slash + "gp_shaders.glsl";
  return res;
}
