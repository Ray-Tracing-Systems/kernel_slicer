#include "kslicer.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"

// LiteNN (apps/LiteMathAux/neural.h): small neural networks evaluated per thread with cooperative vectors.
//
// 'LinearForward(m_weights.data(), m_layers[i], in, inDim, out, outDim)' binds weights vector 'm_weights' to table of layers 'm_layers'.
// Generated 'CommitDeviceData()' repacks matrices of such weights to 'InferencingOptimal' layout and uploads 'm_layers' with GPU offsets.
// Weights used with explicit offsets 'LinearForward(m_weights.data(), matOffs, biasOffs, ...)' stay row-major on GPU.

namespace
{
  // 'm_weights.data()' or 'm_layers[i]' ==> 'm_weights' or 'm_layers'; "" if expression is not a member of main class
  //
  std::string MemberVectorName(const clang::Expr* a_expr)
  {
    a_expr = a_expr->IgnoreParenImpCasts();
    if(auto pCall = clang::dyn_cast<clang::CXXMemberCallExpr>(a_expr))
      a_expr = pCall->getImplicitObjectArgument()->IgnoreParenImpCasts();
    else if(auto pOp = clang::dyn_cast<clang::CXXOperatorCallExpr>(a_expr))
    {
      if(pOp->getOperator() == clang::OO_Subscript)
        a_expr = pOp->getArg(0)->IgnoreParenImpCasts();
    }
    auto pMember = clang::dyn_cast<clang::MemberExpr>(a_expr);
    if(pMember == nullptr || !clang::isa<clang::CXXThisExpr>(pMember->getBase()->IgnoreParenImpCasts()))
      return "";
    return pMember->getMemberDecl()->getNameAsString();
  }

  class LiteNNCallsVisitor : public clang::RecursiveASTVisitor<LiteNNCallsVisitor>
  {
  public:
    LiteNNCallsVisitor(const clang::SourceManager& a_sm) : m_sm(a_sm) {}

    bool TraverseNamespaceDecl(clang::NamespaceDecl* a_decl) // implementation of LiteNN itself calls its functions with pointer arguments
    {
      if(a_decl->getName() == "LiteNN")
        return true;
      return clang::RecursiveASTVisitor<LiteNNCallsVisitor>::TraverseNamespaceDecl(a_decl);
    }

    bool VisitCallExpr(clang::CallExpr* call)
    {
      const clang::FunctionDecl* fDecl = call->getDirectCallee();
      if(fDecl == nullptr || fDecl->getQualifiedNameAsString().rfind("LiteNN::", 0) != 0)
        return true;
      const std::string fname = fDecl->getNameAsString();
      if(fname != "LinearForward" && fname != "LinearForward_ReLU")
        return true;

      const std::string weights = MemberVectorName(call->getArg(0));
      if(weights == "")
      {
        kslicer::PrintError("LiteNN::" + fname + ": weights must be a member vector of main class, 'm_weights.data()'", call->getSourceRange(), m_sm);
        return true;
      }

      if(call->getNumArgs() == 7) // explicit offsets
      {
        if(layersOfWeights.find(weights) != layersOfWeights.end())
          kslicer::PrintError("LiteNN::" + fname + ": weights '" + weights + "' are used both with table of layers and with explicit offsets", call->getSourceRange(), m_sm);
        rowMajorWeights.insert(weights);
        return true;
      }

      const std::string layers = MemberVectorName(call->getArg(1));
      if(layers == "")
      {
        kslicer::PrintError("LiteNN::" + fname + ": layer must be an element of member vector, 'm_layers[i]'", call->getSourceRange(), m_sm);
        return true;
      }
      if(rowMajorWeights.count(weights) != 0)
        kslicer::PrintError("LiteNN::" + fname + ": weights '" + weights + "' are used both with table of layers and with explicit offsets", call->getSourceRange(), m_sm);

      auto pLayers  = layersOfWeights.find(weights);
      auto pWeights = weightsOfLayers.find(layers);
      if(pLayers != layersOfWeights.end() && pLayers->second != layers)
        kslicer::PrintError("LiteNN::" + fname + ": weights '" + weights + "' are used with two tables of layers, '" + pLayers->second + "' and '" + layers + "'", call->getSourceRange(), m_sm);
      else if(pWeights != weightsOfLayers.end() && pWeights->second != weights)
        kslicer::PrintError("LiteNN::" + fname + ": table of layers '" + layers + "' is used with two weights vectors, '" + pWeights->second + "' and '" + weights + "'", call->getSourceRange(), m_sm);
      else
      {
        layersOfWeights[weights] = layers;
        weightsOfLayers[layers]  = weights;
      }
      return true;
    }

    kslicer::umap<std::string, std::string> layersOfWeights;
    kslicer::umap<std::string, std::string> weightsOfLayers;
    kslicer::uset<std::string>              rowMajorWeights;

  private:
    const clang::SourceManager& m_sm;
  };
};

void kslicer::MainClassInfo::ProcessNeuralNetworks(clang::ASTContext& a_astContext)
{
  LiteNNCallsVisitor visitor(a_astContext.getSourceManager());
  visitor.TraverseDecl(a_astContext.getTranslationUnitDecl());

  // only members which are used on GPU: calls in code of CPU only should not cause repacking on host
  nnLayersOfWeights.clear();
  for(const auto& nn : visitor.layersOfWeights)
  {
    bool usedOnGPU = false;
    for(const auto& member : dataMembers)
      if(member.name == nn.first && member.isContainer && member.usedInKernel)
        usedOnGPU = true;
    if(usedOnGPU)
      nnLayersOfWeights[nn.first] = nn.second;
  }
}
