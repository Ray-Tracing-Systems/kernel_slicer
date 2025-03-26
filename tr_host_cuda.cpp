#include "kslicer.h"
#include "template_rendering.h"
#include "class_gen.h"

void kslicer::CudaCodeGen::GenerateHost(std::string fullSuffix, nlohmann::json jsonHost, kslicer::MainClassInfo& a_mainClass, const kslicer::TextGenSettings& a_settings)
{
  kslicer::ApplyJsonToTemplate("templates_cuda/gen_class.cu", fullSuffix + ".cu", jsonHost);
}

bool kslicer::MainFunctionRewriterCUDA::VisitCXXMemberCallExpr(clang::CXXMemberCallExpr* f)
{
  const DeclarationNameInfo dni = f->getMethodDecl()->getNameInfo();
  const DeclarationName dn      = dni.getName();
  const std::string fname       = dn.getAsString();

  if(fname == "data")
  {
    const std::string exprContent = GetRangeSourceCode(f->getSourceRange(), m_compiler);
    const auto posOfPoint         = exprContent.find(".");
    std::string memberNameA       = exprContent.substr(0, posOfPoint);
    m_rewriter.ReplaceText(f->getSourceRange(), memberNameA + "_dev.data()");
  }

  return true;
}

bool kslicer::MainFunctionRewriterCUDA::VisitCallExpr(clang::CallExpr* f)
{
  const FunctionDecl* fDecl = f->getDirectCallee();
  if(fDecl == nullptr)             // definitely can't process nullpointer
    return true;

  // Get name of function
  const DeclarationNameInfo dni = fDecl->getNameInfo();
  const DeclarationName dn      = dni.getName();
  const std::string fname       = dn.getAsString();

  if(fname == "memcpy")
  {
    std::stringstream strOut;
    strOut << "cudaMemcpy(";
    for (unsigned i = 0; i < f->getNumArgs(); ++i) {
      const clang::Expr* arg = f->getArg(i);
      const std::string exprContent = GetRangeSourceCode(arg->getSourceRange(), m_compiler);
      strOut << exprContent.c_str() << ", ";
    }
    strOut << "cudaMemcpyDeviceToDevice)";
    m_rewriter.ReplaceText(f->getSourceRange(), strOut.str());
  }

  return true;
}