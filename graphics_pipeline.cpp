#include "kslicer.h"
#include "graphics_pipeline.h"
#include "template_rendering.h"
#include "class_gen.h"

#include <sstream>

#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/ExprCXX.h"

#include <iostream>
#include <algorithm>

static const std::string VS_PREFIX = "kernelVS_";
static const std::string PS_PREFIX = "kernelPS_";

kslicer::GP_STAGE kslicer::GraphicsStageByKernelName(const std::string& a_kernelName)
{
  if(a_kernelName.find(VS_PREFIX) == 0)
    return GP_STAGE::VERTEX;
  else if(a_kernelName.find(PS_PREFIX) == 0)
    return GP_STAGE::PIXEL;
  return GP_STAGE::NONE;
}

std::string kslicer::GraphicsShortKernelName(const std::string& a_kernelName)
{
  if(a_kernelName.find(VS_PREFIX) == 0)
    return "VS_" + a_kernelName.substr(VS_PREFIX.size());
  else if(a_kernelName.find(PS_PREFIX) == 0)
    return "PS_" + a_kernelName.substr(PS_PREFIX.size());
  return "";
}

bool kslicer::IsHostOnlyMemberType(const std::string& a_typeName)
{
  return a_typeName.find("LiteRS::") != std::string::npos;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// name of class member in expressions 'm_x', 'this->m_x', 'm_vec.data()'; empty string otherwise
static std::string MemberNameOf(const clang::Expr* a_expr)
{
  if(a_expr == nullptr)
    return "";
  const clang::Expr* expr = a_expr->IgnoreParenImpCasts();
  if(auto pCall = clang::dyn_cast<clang::CXXMemberCallExpr>(expr)) // m_vec.data()
    return MemberNameOf(pCall->getImplicitObjectArgument());
  if(auto pMember = clang::dyn_cast<clang::MemberExpr>(expr))
    return pMember->getMemberDecl()->getNameAsString();
  return "";
}

/// 'this->m_x' or 'm_x' where m_x is a field of the class
static const clang::FieldDecl* ThisFieldOf(const clang::Expr* a_expr)
{
  auto pMember = clang::dyn_cast<clang::MemberExpr>(a_expr->IgnoreParenImpCasts());
  if(pMember == nullptr || !clang::isa<clang::CXXThisExpr>(pMember->getBase()->IgnoreParenImpCasts()))
    return nullptr;
  return clang::dyn_cast<clang::FieldDecl>(pMember->getMemberDecl());
}

/// true if expression reads data of the class: fields or sizes of member vectors
static bool ReadsClassData(const clang::Stmt* a_stmt)
{
  if(a_stmt == nullptr)
    return false;
  if(auto pExpr = clang::dyn_cast<clang::Expr>(a_stmt))
    if(ThisFieldOf(pExpr) != nullptr)
      return true;
  for(const clang::Stmt* child : a_stmt->children())
    if(ReadsClassData(child))
      return true;
  return false;
}

/**
\brief Shader code of count of draw command which reads data of the class: fields (and elements of array fields) are read from
       UBO, 'm_vec.size()' is 'm_vec_size' of UBO, constant subexpressions are evaluated. Returns false for other expressions
       (arguments and local variables of control function, function calls).
*/
static bool CountExprForShader(const clang::Expr* a_expr, const kslicer::MainClassInfo& a_info, const clang::ASTContext& a_ctx, std::string& a_out)
{
  const clang::Expr* expr = a_expr->IgnoreParenImpCasts();
  clang::Expr::EvalResult result;
  if(!expr->isValueDependent() && !ReadsClassData(expr) && expr->EvaluateAsInt(result, a_ctx))
  {
    a_out = std::to_string(result.Val.getInt().getZExtValue()) + "u";
    return true;
  }
  if(auto pCast = clang::dyn_cast<clang::ExplicitCastExpr>(expr)) // 'uint32_t(x)', '(uint32_t)x', 'static_cast<uint32_t>(x)'
  {
    std::string sub;
    if(!CountExprForShader(pCast->getSubExpr(), a_info, a_ctx, sub))
      return false;
    a_out = "uint(" + sub + ")";
    return true;
  }
  if(const clang::FieldDecl* pField = ThisFieldOf(expr))
  {
    if(a_info.allDataMembers.find(pField->getNameAsString()) == a_info.allDataMembers.end())
      return false;
    a_out = a_info.pShaderCC->UBOAccess(pField->getNameAsString());
    return true;
  }
  if(auto pCall = clang::dyn_cast<clang::CXXMemberCallExpr>(expr)) // 'm_vec.size()'
  {
    const clang::FieldDecl* pField = ThisFieldOf(pCall->getImplicitObjectArgument());
    if(pField == nullptr || pCall->getMethodDecl() == nullptr || pCall->getMethodDecl()->getNameAsString() != "size")
      return false;
    a_out = a_info.pShaderCC->UBOAccess(pField->getNameAsString() + "_size");
    return true;
  }
  if(auto pIndex = clang::dyn_cast<clang::ArraySubscriptExpr>(expr)) // 'm_counts[MESH_ID]'
  {
    std::string base, index;
    if(!CountExprForShader(pIndex->getBase(), a_info, a_ctx, base) || !CountExprForShader(pIndex->getIdx(), a_info, a_ctx, index))
      return false;
    a_out = base + "[" + index + "]";
    return true;
  }
  if(auto pBin = clang::dyn_cast<clang::BinaryOperator>(expr))
  {
    std::string lhs, rhs;
    if(pBin->isAssignmentOp() || pBin->isCommaOp() || !CountExprForShader(pBin->getLHS(), a_info, a_ctx, lhs) || !CountExprForShader(pBin->getRHS(), a_info, a_ctx, rhs))
      return false;
    a_out = "(" + lhs + " " + pBin->getOpcodeStr().str() + " " + rhs + ")";
    return true;
  }
  if(auto pCond = clang::dyn_cast<clang::ConditionalOperator>(expr))
  {
    std::string cond, lhs, rhs;
    if(!CountExprForShader(pCond->getCond(), a_info, a_ctx, cond) || !CountExprForShader(pCond->getTrueExpr(), a_info, a_ctx, lhs) ||
       !CountExprForShader(pCond->getFalseExpr(), a_info, a_ctx, rhs))
      return false;
    a_out = "(" + cond + " ? " + lhs + " : " + rhs + ")";
    return true;
  }
  return false;
}

std::string kslicer::GraphicsCallKey(const std::string& a_controlFunc, const clang::CallExpr* a_call, const clang::SourceManager& a_sm)
{
  const clang::SourceLocation loc = a_call->getBeginLoc();
  return a_controlFunc + ":" + std::to_string(a_sm.getSpellingLineNumber(loc)) + ":" + std::to_string(a_sm.getSpellingColumnNumber(loc));
}

/// name of member function in expression '&Class::func'; empty string otherwise
static std::string MethodNameOf(const clang::Expr* a_expr)
{
  const clang::Expr* expr = a_expr->IgnoreParenImpCasts();
  auto pUnary = clang::dyn_cast<clang::UnaryOperator>(expr);
  if(pUnary == nullptr || pUnary->getOpcode() != clang::UO_AddrOf)
    return "";
  auto pRef = clang::dyn_cast<clang::DeclRefExpr>(pUnary->getSubExpr()->IgnoreParenImpCasts());
  if(pRef == nullptr || !clang::isa<clang::CXXMethodDecl>(pRef->getDecl()))
    return "";
  return pRef->getDecl()->getNameAsString();
}

bool kslicer::IsLiteRSCall(const clang::CallExpr* a_call)
{
  const clang::FunctionDecl* pFunc = a_call->getDirectCallee();
  return pFunc != nullptr && pFunc->getQualifiedNameAsString().find("LiteRS::") == 0;
}

bool kslicer::ParseGraphicsCommand(const clang::CallExpr* a_call, GPCommand& a_cmd)
{
  if(!IsLiteRSCall(a_call))
    return false;
  const clang::FunctionDecl* pFunc = a_call->getDirectCallee();

  const std::string fname = pFunc->getNameAsString();
  a_cmd = GPCommand();
  if(a_call->getNumArgs() == 0)
    return false;
  a_cmd.pipeline = MemberNameOf(a_call->getArg(0));

  if(fname == "Clear")
  {
    a_cmd.kind = GPCommand::KIND::CLEAR;
    return true;
  }
  else if(fname != "DrawTriangles" && fname != "DrawIndexedTriangles" && fname != "DrawInstancedTriangles" && fname != "DrawIndexedInstancedTriangles")
    return false;

  // DrawTriangles(pipe, this, &C::kernelVS_X, [&C::kernelPS_X,] count, args...)
  // DrawIndexedTriangles(pipe, this, &C::kernelVS_X, [&C::kernelPS_X,] m_indices.data(), count, args...)
  // DrawInstancedTriangles(pipe, this, &C::kernelVS_X, [&C::kernelPS_X,] count, instCount, args...)
  // DrawIndexedInstancedTriangles(pipe, this, &C::kernelVS_X, [&C::kernelPS_X,] m_indices.data(), count, instCount, args...)
  //
  a_cmd.kind      = GPCommand::KIND::DRAW;
  a_cmd.indexed   = (fname == "DrawIndexedTriangles"   || fname == "DrawIndexedInstancedTriangles");
  a_cmd.instanced = (fname == "DrawInstancedTriangles" || fname == "DrawIndexedInstancedTriangles");
  if(a_call->getNumArgs() < 4)
    return true;
  a_cmd.vsKernel = MethodNameOf(a_call->getArg(2));
  a_cmd.psKernel = MethodNameOf(a_call->getArg(3));
  uint32_t argId = (a_cmd.psKernel != "") ? 4 : 3;
  if(a_cmd.indexed && argId < a_call->getNumArgs())
  {
    a_cmd.indexBuffer = MemberNameOf(a_call->getArg(argId));
    argId++;
  }
  for(; argId < a_call->getNumArgs(); argId++)
    a_cmd.countAndArgs.push_back(a_call->getArg(argId));
  return true;
}

std::string kslicer::GraphicsDrawName(const GPCommand& a_cmd)
{
  const std::string psName = (a_cmd.psKernel != "") ? a_cmd.psKernel.substr(PS_PREFIX.size()) : std::string("DepthOnly");
  return "Draw_" + a_cmd.vsKernel.substr(VS_PREFIX.size()) + "_" + psName + "_" + a_cmd.pipeline + (a_cmd.indexed ? "_Indexed" : "") + (a_cmd.instanced ? "_Instanced" : "");
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace
{
  /// finds 'm_pipe.SetColorTarget(m_image, index)' and 'm_pipe.SetDepthTarget(m_image)' in the whole translation unit
  class PipelineSetupVisitor : public clang::RecursiveASTVisitor<PipelineSetupVisitor>
  {
  public:
    PipelineSetupVisitor(kslicer::GraphicsInfo& a_info, const clang::CompilerInstance& a_compiler) : m_info(a_info), m_compiler(a_compiler) {}

    bool VisitCXXMemberCallExpr(clang::CXXMemberCallExpr* a_call)
    {
      const clang::CXXMethodDecl* pMethod = a_call->getMethodDecl();
      if(pMethod == nullptr || pMethod->getParent()->getQualifiedNameAsString() != "LiteRS::Pipeline")
        return true;

      const std::string fname = pMethod->getNameAsString();
      if(fname != "SetColorTarget" && fname != "SetDepthTarget")
        return true;

      const std::string pipeName   = MemberNameOf(a_call->getImplicitObjectArgument());
      const std::string targetName = (a_call->getNumArgs() > 0) ? MemberNameOf(a_call->getArg(0)) : "";
      if(pipeName == "" || targetName == "")
      {
        kslicer::PrintError("LiteRS::Pipeline::" + fname + ": pipeline and image must be members of the main class", a_call->getSourceRange(), m_compiler.getSourceManager());
        return true;
      }

      auto& pipe = m_info.pipelines[pipeName];
      pipe.name  = pipeName;
      if(fname == "SetDepthTarget")
      {
        if(pipe.depthTarget != "" && pipe.depthTarget != targetName)
          kslicer::PrintError("LiteRS::Pipeline '" + pipeName + "': depth target is set to different images, this is not supported", a_call->getSourceRange(), m_compiler.getSourceManager());
        pipe.depthTarget = targetName;
        return true;
      }

      uint32_t index = 0;
      if(a_call->getNumArgs() > 1 && !clang::isa<clang::CXXDefaultArgExpr>(a_call->getArg(1)))
      {
        clang::Expr::EvalResult result;
        if(a_call->getArg(1)->EvaluateAsInt(result, m_compiler.getASTContext()))
          index = uint32_t(result.Val.getInt().getZExtValue());
        else
          kslicer::PrintError("LiteRS::Pipeline::SetColorTarget: index of color target must be a constant", a_call->getSourceRange(), m_compiler.getSourceManager());
      }

      for(const auto& target : pipe.colorTargets)
      {
        if(target.index == index)
        {
          if(target.member != targetName)
            kslicer::PrintError("LiteRS::Pipeline '" + pipeName + "': color target " + std::to_string(index) + " is set to different images, this is not supported", a_call->getSourceRange(), m_compiler.getSourceManager());
          return true;
        }
      }
      pipe.colorTargets.push_back(kslicer::GPColorTarget{targetName, index});
      return true;
    }

  private:
    kslicer::GraphicsInfo&          m_info;
    const clang::CompilerInstance&  m_compiler;
  };

  /// finds LiteRS commands in the body of control function
  class CommandsVisitor : public clang::RecursiveASTVisitor<CommandsVisitor>
  {
  public:
    CommandsVisitor(std::vector<std::pair<kslicer::GPCommand, const clang::CallExpr*> >& a_out, const clang::SourceManager& a_sm) : m_out(a_out), m_sm(a_sm) {}

    bool VisitCallExpr(clang::CallExpr* a_call)
    {
      kslicer::GPCommand cmd;
      if(kslicer::ParseGraphicsCommand(a_call, cmd))
        m_out.push_back(std::make_pair(cmd, a_call));
      else if(kslicer::IsLiteRSCall(a_call)) // otherwise the call would silently remain in generated host code
        kslicer::PrintError("LiteRS: unknown command '" + a_call->getDirectCallee()->getNameAsString() + "' in control function", a_call->getSourceRange(), m_sm);
      return true;
    }

  private:
    std::vector<std::pair<kslicer::GPCommand, const clang::CallExpr*> >& m_out;
    const clang::SourceManager& m_sm;
  };
};

void kslicer::ProcessGraphicsPipelines(MainClassInfo& a_codeInfo, const clang::CompilerInstance& a_compiler)
{
  PipelineSetupVisitor visitor(a_codeInfo.graphics, a_compiler);
  visitor.TraverseDecl(a_compiler.getASTContext().getTranslationUnitDecl());

  for(auto& pipe : a_codeInfo.graphics.pipelines)
  {
    std::sort(pipe.second.colorTargets.begin(), pipe.second.colorTargets.end(), [](const auto& a, const auto& b) { return a.index < b.index; });
    for(uint32_t i = 0; i < uint32_t(pipe.second.colorTargets.size()); i++)
    {
      if(pipe.second.colorTargets[i].index != i)
        std::cout << "  [kslicer]: ERROR, LiteRS::Pipeline '" << pipe.first.c_str() << "': color targets must have indices 0, 1, ... without gaps" << std::endl;
    }
  }
}

void kslicer::ProcessGraphicsCommands(MainFuncInfo& a_mainFunc, MainClassInfo& a_codeInfo, const clang::CompilerInstance& a_compiler)
{
  const clang::FunctionDecl* pDef = nullptr;
  if(a_mainFunc.Node == nullptr || !a_mainFunc.Node->hasBody(pDef))
    return;

  std::vector<std::pair<GPCommand, const clang::CallExpr*> > commands;
  CommandsVisitor visitor(commands, a_compiler.getSourceManager());
  visitor.TraverseStmt(const_cast<clang::Stmt*>(pDef->getBody()));

  const auto& sm = a_compiler.getSourceManager();
  for(const auto& cmdAndCall : commands)
  {
    const GPCommand&       cmd   = cmdAndCall.first;
    const clang::CallExpr* call  = cmdAndCall.second;
    const auto             pPipe = a_codeInfo.graphics.pipelines.find(cmd.pipeline);
    if(pPipe == a_codeInfo.graphics.pipelines.end())
    {
      kslicer::PrintError("LiteRS: pipeline '" + cmd.pipeline + "' must be a member of the main class with render targets set by SetColorTarget/SetDepthTarget", call->getSourceRange(), sm);
      continue;
    }
    // render targets and index buffers are used by commands, so GPU objects are created for them even if no kernel reads them
    //
    std::vector<std::string> usedMembers;
    for(const auto& target : pPipe->second.colorTargets)
      usedMembers.push_back(target.member);
    if(pPipe->second.depthTarget != "")
      usedMembers.push_back(pPipe->second.depthTarget);
    if(cmd.indexBuffer != "")
      usedMembers.push_back(cmd.indexBuffer);
    for(const auto& name : usedMembers)
    {
      auto pMember = a_codeInfo.allDataMembers.find(name);
      if(pMember != a_codeInfo.allDataMembers.end())
        pMember->second.usedInMainFn = true;
    }

    if(cmd.kind != GPCommand::KIND::DRAW)
      continue;

    if(GraphicsStageByKernelName(cmd.vsKernel) != GP_STAGE::VERTEX || (cmd.psKernel != "" && GraphicsStageByKernelName(cmd.psKernel) != GP_STAGE::PIXEL))
    {
      kslicer::PrintError("LiteRS: shaders must be member functions with names 'kernelVS_*' and 'kernelPS_*' passed as '&Class::kernelVS_Name'", call->getSourceRange(), sm);
      continue;
    }
    if(cmd.indexed && cmd.indexBuffer == "")
    {
      kslicer::PrintError("LiteRS::DrawIndexedTriangles: index buffer must be a member vector, 'm_indices.data()'", call->getSourceRange(), sm);
      continue;
    }

    for(const auto& shaderName : {cmd.vsKernel, cmd.psKernel})
    {
      if(shaderName == "")
        continue;
      auto pKernel = a_codeInfo.allKernels.find(shaderName);
      if(pKernel == a_codeInfo.allKernels.end())
      {
        kslicer::PrintError("LiteRS: shader '" + shaderName + "' is not found among member functions of the main class", call->getSourceRange(), sm);
        continue;
      }
      // instance id is the second argument of vertex shader of instanced draws, so one vertex shader can not be used in both kinds of draws
      //
      KernelInfo& kernel = pKernel->second;
      if(kernel.gpStage == GP_STAGE::VERTEX && kernel.gpInstanced != cmd.instanced)
        kslicer::PrintError("LiteRS: vertex shader '" + shaderName + "' is used both in instanced and in not instanced draws, this is not supported", call->getSourceRange(), sm);
      if(GraphicsStageByKernelName(shaderName) == GP_STAGE::VERTEX && cmd.instanced)
      {
        const clang::CXXMethodDecl* pDecl = kernel.astNode;
        const bool idIsOk = (pDecl != nullptr && pDecl->getNumParams() >= 2 && pDecl->getParamDecl(1)->getType()->isUnsignedIntegerType() &&
                             !pDecl->getParamDecl(1)->getType()->isPointerType());
        if(!idIsOk)
          kslicer::PrintError("LiteRS: vertex shader '" + shaderName + "' of instanced draw must be 'float4 " + shaderName + "(uint32_t a_vertexId, uint32_t a_instanceId, ...) const'", call->getSourceRange(), sm);
      }
      kernel.usedInMainFunc = true;
      kernel.gpStage        = GraphicsStageByKernelName(shaderName);
      kernel.gpInstanced    = cmd.instanced && (kernel.gpStage == GP_STAGE::VERTEX);
      a_mainFunc.UsedKernels.insert(shaderName);
    }

    GPDrawInfo draw;
    draw.pipeline    = cmd.pipeline;
    draw.vsKernel    = cmd.vsKernel;
    draw.psKernel    = cmd.psKernel;
    draw.indexed     = cmd.indexed;
    draw.instanced   = cmd.instanced;
    draw.name        = GraphicsDrawName(cmd);
    if(cmd.indexBuffer != "")
      a_codeInfo.graphics.indexBuffers.insert(cmd.indexBuffer);

    // counts which read data of the class (members, 'm_vec.size()') are known only on GPU: the draw is indirect;
    // counts from arguments, local variables and constants are computed on host as usual
    //
    const std::string callKey = GraphicsCallKey(a_mainFunc.Name, call, sm);
    const size_t      countsNum = std::min<size_t>(cmd.instanced ? 2 : 1, cmd.countAndArgs.size());
    std::string       counts[2];
    bool              indirect = false, countsAreOk = true;
    for(size_t i = 0; i < countsNum; i++)
    {
      const auto pCount = static_cast<const clang::Expr*>(cmd.countAndArgs[i]);
      if(!ReadsClassData(pCount))
        continue;
      indirect = true;
      if(!CountExprForShader(pCount, a_codeInfo, a_compiler.getASTContext(), counts[i]))
      {
        kslicer::PrintError("LiteRS: count of draw reads data of the class, so it is computed on GPU (indirect draw), but it also uses "
                            "variables or functions of the control function; use only members, 'm_vec.size()' and constants, "
                            "or read the value into a local variable to make an ordinary draw", call->getSourceRange(), sm);
        countsAreOk = false;
      }
    }
    if(indirect && countsAreOk && a_codeInfo.graphics.indirectDrawByCall.find(callKey) == a_codeInfo.graphics.indirectDrawByCall.end())
    {
      GPIndirectDraw indirectDraw;
      indirectDraw.id        = uint32_t(a_codeInfo.graphics.indirectDraws.size());
      indirectDraw.indexed   = cmd.indexed;
      indirectDraw.countExpr = counts[0];
      indirectDraw.instExpr  = counts[1];
      a_codeInfo.graphics.indirectDraws.push_back(indirectDraw);
      a_codeInfo.graphics.indirectDrawByCall[callKey] = indirectDraw.id;
    }
    for(size_t i = 0; i < countsNum; i++) // members of indirect counts are read from UBO
    {
      std::vector<const clang::Stmt*> stack = { static_cast<const clang::Expr*>(cmd.countAndArgs[i]) };
      while(!stack.empty())
      {
        const clang::Stmt* pStmt = stack.back();
        stack.pop_back();
        if(pStmt == nullptr)
          continue;
        if(auto pExpr = clang::dyn_cast<clang::Expr>(pStmt))
          if(const clang::FieldDecl* pField = ThisFieldOf(pExpr))
          {
            auto pMember = a_codeInfo.allDataMembers.find(pField->getNameAsString());
            if(pMember != a_codeInfo.allDataMembers.end())
              pMember->second.usedInMainFn = true;
          }
        for(const clang::Stmt* child : pStmt->children())
          stack.push_back(child);
      }
    }
    a_codeInfo.graphics.draws[draw.name] = draw;
  }
}

void kslicer::PrepareGraphicsKernel(KernelInfo& a_kernel)
{
  if(a_kernel.gpStage == GP_STAGE::PIXEL) // 'PS(const V& a_in)': input comes from rasterizer
  {
    a_kernel.args.clear();
    return;
  }

  // 'VS(uint32_t a_vertexId, [uint32_t a_instanceId,] Args..., V* a_out)': vertex and instance ids come from rasterizer,
  // varyings go to rasterizer; other args are push constants
  //
  std::vector<KernelInfo::ArgInfo> args;
  for(size_t i = a_kernel.gpInstanced ? 2 : 1; i < a_kernel.args.size(); i++)
  {
    const bool isLast = (i + 1 == a_kernel.args.size());
    if(isLast && a_kernel.args[i].IsPointer())
      continue;
    args.push_back(a_kernel.args[i]);
  }
  a_kernel.args = args;
}

void kslicer::AddGraphicsShadersAsMemberFunctions(MainClassInfo& a_codeInfo)
{
  for(auto& k : a_codeInfo.kernels)
  {
    if(k.second.gpStage == GP_STAGE::NONE)
      continue;
    FuncData func = FuncDataFromKernel(k.second);
    func.isKernel = false;   // the shader is emitted as an ordinary function, entry point calls it
    func.depthUse = 0;
    k.second.usedMemberFunctions[func.srcHash] = func; // inserted after the functions it calls, so it is emitted after them
  }
}

bool kslicer::IsGraphicsRenderTarget(const MainClassInfo& a_codeInfo, const std::string& a_member, bool* a_isDepth)
{
  for(const auto& pipe : a_codeInfo.graphics.pipelines)
  {
    if(pipe.second.depthTarget == a_member)
    {
      if(a_isDepth != nullptr)
        (*a_isDepth) = true;
      return true;
    }
    for(const auto& target : pipe.second.colorTargets)
    {
      if(target.member == a_member)
      {
        if(a_isDepth != nullptr)
          (*a_isDepth) = false;
        return true;
      }
    }
  }
  return false;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////// json for templates ////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// type of shader parameter or result in shader language: references, pointers and qualifiers are removed
static std::string ShaderTypeName(clang::QualType a_type, const kslicer::MainClassInfo& a_classInfo)
{
  a_type = a_type.getNonReferenceType();
  if(a_type->isPointerType())
    a_type = a_type->getPointeeType();
  std::string name = a_type.getUnqualifiedType().getAsString();
  for(const std::string& prefix : {"const ", "struct ", "class "})
    while(ReplaceFirst(name, prefix, ""))
      ;
  return a_classInfo.pShaderFuncRewriter->RewriteStdVectorTypeStr(name);
}

/// Vulkan format of render target; RGBA8 target 'Image2D<uint32_t>' is R32_UINT on GPU (as for kernels), pixel shader packs color to uint
static std::string TargetFormat(const kslicer::MainClassInfo& a_classInfo, const std::string& a_member)
{
  const auto pMember = a_classInfo.allDataMembers.find(a_member);
  if(pMember == a_classInfo.allDataMembers.end())
    return "VK_FORMAT_UNDEFINED";
  const std::string dataType = a_classInfo.pShaderFuncRewriter->RewriteStdVectorTypeStr(pMember->second.containerDataType);
  return kslicer::InferenceVulkanTextureFormatFromTypeName(dataType, false);
}

void kslicer::AddGraphicsToKernelJson(nlohmann::json& a_kernelJson, const KernelInfo& a_kernel, const MainClassInfo& a_classInfo)
{
  a_kernelJson["IsGraphics"]       = (a_kernel.gpStage != GP_STAGE::NONE);
  a_kernelJson["GPStage"]          = int(a_kernel.gpStage);
  a_kernelJson["GPInstanced"]      = a_kernel.gpInstanced;
  a_kernelJson["DSetId"]           = (a_kernel.gpStage == GP_STAGE::PIXEL) ? 1 : 0; // vertex shader uses set 0, pixel shader uses set 1
  a_kernelJson["GPFuncName"]       = a_kernel.name;
  a_kernelJson["GPHasVaryings"]    = false;
  a_kernelJson["GPVaryings"]       = "";
  a_kernelJson["GPOutputIsStruct"] = false;
  a_kernelJson["GPOutputType"]     = "float4";
  a_kernelJson["GPOutputs"]        = std::vector<nlohmann::json>();
  if(a_kernel.gpStage == GP_STAGE::NONE || a_kernel.astNode == nullptr)
    return;

  const clang::CXXMethodDecl* pFunc = a_kernel.astNode;
  if(a_kernel.gpStage == GP_STAGE::VERTEX)
  {
    const unsigned numParams = pFunc->getNumParams();
    if(numParams >= 2 && pFunc->getParamDecl(numParams-1)->getType()->isPointerType())
    {
      a_kernelJson["GPHasVaryings"] = true;
      a_kernelJson["GPVaryings"]    = ShaderTypeName(pFunc->getParamDecl(numParams-1)->getType(), a_classInfo);
    }
    return;
  }

  // pixel shader: 'float4 PS(const V& a_in)' writes color target 0, 'Struct PS(const V& a_in)' writes one target for each float4 field
  //
  if(pFunc->getNumParams() == 1)
  {
    a_kernelJson["GPHasVaryings"] = true;
    a_kernelJson["GPVaryings"]    = ShaderTypeName(pFunc->getParamDecl(0)->getType(), a_classInfo);
  }

  std::vector<std::string> targetFormats; // RGBA8 is packed to R32_UINT; raw uint4 keeps all bits
  for(const auto& draw : a_classInfo.graphics.draws)
  {
    if(draw.second.psKernel != a_kernel.name)
      continue;
    const auto pPipe = a_classInfo.graphics.pipelines.find(draw.second.pipeline);
    if(pPipe != a_classInfo.graphics.pipelines.end())
      for(const auto& target : pPipe->second.colorTargets)
        targetFormats.push_back(TargetFormat(a_classInfo, target.member));
    break;
  }

  const clang::QualType retType = pFunc->getReturnType();
  const std::string     retName = ShaderTypeName(retType, a_classInfo);
  a_kernelJson["GPOutputType"]  = retName;
  const clang::CXXRecordDecl* pRecord = retType->getAsCXXRecordDecl();
  if(retName == "float4" || retName == "vec4" || retName == "uint4" || retName == "uvec4" || pRecord == nullptr)
  {
    nlohmann::json out;
    out["Name"]   = "color0";
    const std::string format = targetFormats.empty() ? "" : targetFormats[0];
    out["PackRGBA8"] = (format == "VK_FORMAT_R32_UINT");
    out["Type"] = a_classInfo.pShaderFuncRewriter->RewriteStdVectorTypeStr(format == "VK_FORMAT_R32G32B32A32_UINT" ? "uint4" : (format == "VK_FORMAT_R32_UINT" ? "uint" : "float4"));
    a_kernelJson["GPOutputs"].push_back(out);
    return;
  }

  a_kernelJson["GPOutputIsStruct"] = true;
  for(const clang::FieldDecl* pField : pRecord->fields())
  {
    const size_t index = a_kernelJson["GPOutputs"].size();
    nlohmann::json out;
    out["Name"]   = pField->getNameAsString();
    const std::string format = index < targetFormats.size() ? targetFormats[index] : "";
    out["PackRGBA8"] = (format == "VK_FORMAT_R32_UINT");
    out["Type"] = a_classInfo.pShaderFuncRewriter->RewriteStdVectorTypeStr(format == "VK_FORMAT_R32G32B32A32_UINT" ? "uint4" : (format == "VK_FORMAT_R32_UINT" ? "uint" : "float4"));
    a_kernelJson["GPOutputs"].push_back(out);
  }
}

nlohmann::json kslicer::GraphicsIndirectDrawsJson(const MainClassInfo& a_classInfo)
{
  nlohmann::json draws = std::vector<nlohmann::json>();
  for(const auto& draw : a_classInfo.graphics.indirectDraws)
  {
    nlohmann::json drawJson;
    drawJson["Id"]        = draw.id;
    drawJson["Indexed"]   = draw.indexed;
    drawJson["CountExpr"] = draw.countExpr;
    drawJson["InstExpr"]  = draw.instExpr;
    draws.push_back(drawJson);
  }
  return draws;
}

void kslicer::AddGraphicsToHostJson(nlohmann::json& a_hostJson, const MainClassInfo& a_classInfo)
{
  a_hostJson["HasGraphics"] = a_classInfo.graphics.Used();
  a_hostJson["GPPipelines"] = std::vector<nlohmann::json>();
  a_hostJson["GPDraws"]     = std::vector<nlohmann::json>();
  a_hostJson["GPRenderTargets"] = std::vector<nlohmann::json>();
  a_hostJson["GPIndirectDraws"] = GraphicsIndirectDrawsJson(a_classInfo);
  a_hostJson["HasGPIndirect"]   = !a_classInfo.graphics.indirectDraws.empty();

  // render targets are created with attachment usage, stay in GENERAL layout and are not copied from CPU
  //
  for(auto& var : a_hostJson["ClassTextureVars"])
  {
    var["Aspect"] = "VK_IMAGE_ASPECT_COLOR_BIT";
    var["IsRenderTarget"] = false;
    bool isDepth = false;
    if(!a_classInfo.graphics.Used() || !IsGraphicsRenderTarget(a_classInfo, var["Name"].get<std::string>(), &isDepth))
      continue;
    var["NeedUpdate"] = false;
    var["IsRenderTarget"] = true;
    if(isDepth)
    {
      var["Format"] = "VK_FORMAT_D32_SFLOAT";
      var["Usage"]  = "VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT";
      var["Aspect"] = "VK_IMAGE_ASPECT_DEPTH_BIT";
    }
    else
      var["Usage"]  = var["Usage"].get<std::string>() + " | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT";
    a_hostJson["GPRenderTargets"].push_back(var);
  }

  for(auto& var : a_hostJson["ClassVectorVars"]) // index buffers of draws need VK_BUFFER_USAGE_INDEX_BUFFER_BIT
    var["IsGPIndexBuffer"] = (a_classInfo.graphics.indexBuffers.find(var["Name"].get<std::string>()) != a_classInfo.graphics.indexBuffers.end());

  for(auto& mainFunc : a_hostJson["MainFunctions"])
  {
    for(auto& ds : mainFunc["DescriptorSets"])
    {
      for(auto& arg : ds["Args"])
      {
        if(arg["IsTexture"].get<bool>() && IsGraphicsRenderTarget(a_classInfo, arg["NameOriginal"].get<std::string>()))
          arg["AccessLayout"] = "VK_IMAGE_LAYOUT_GENERAL";
      }
    }
  }

  if(!a_classInfo.graphics.Used())
    return;

  for(const auto& pipe : a_classInfo.graphics.pipelines)
  {
    nlohmann::json pipeJson;
    pipeJson["Name"]         = pipe.second.name;
    pipeJson["ColorTargets"] = std::vector<nlohmann::json>();
    for(const auto& target : pipe.second.colorTargets)
    {
      nlohmann::json targetJson;
      targetJson["Name"]   = target.member;
      targetJson["Format"] = TargetFormat(a_classInfo, target.member);
      targetJson["IsRawUint4"] = (targetJson["Format"] == "VK_FORMAT_R32G32B32A32_UINT");
      targetJson["IsUint"] = (targetJson["Format"] == "VK_FORMAT_R32_UINT" || targetJson["IsRawUint4"].get<bool>());
      pipeJson["ColorTargets"].push_back(targetJson);
    }
    pipeJson["HasDepth"]    = (pipe.second.depthTarget != "");
    pipeJson["DepthTarget"] = pipe.second.depthTarget;
    pipeJson["SizeFrom"]    = pipe.second.colorTargets.empty() ? pipe.second.depthTarget : pipe.second.colorTargets[0].member;
    a_hostJson["GPPipelines"].push_back(pipeJson);
  }

  for(const auto& drawPair : a_classInfo.graphics.draws)
  {
    const GPDrawInfo& draw = drawPair.second;
    const auto pPipe       = a_classInfo.graphics.pipelines.find(draw.pipeline);
    if(pPipe == a_classInfo.graphics.pipelines.end())
      continue;

    nlohmann::json drawJson;
    drawJson["Name"]        = draw.name;
    drawJson["Pipeline"]    = draw.pipeline;
    drawJson["VS"]          = GraphicsShortKernelName(draw.vsKernel);
    drawJson["VSOriginal"]  = draw.vsKernel;
    drawJson["VSFile"]      = draw.vsKernel + ".vert.spv";
    drawJson["HasPS"]       = (draw.psKernel != "");
    drawJson["PS"]          = GraphicsShortKernelName(draw.psKernel);
    drawJson["PSFile"]      = draw.psKernel + ".frag.spv";
    drawJson["Indexed"]     = draw.indexed;
    drawJson["Instanced"]   = draw.instanced;
    drawJson["HasDepth"]    = (pPipe->second.depthTarget != "");

    drawJson["PassTargets"] = std::vector<nlohmann::json>(); // all color targets of pipeline object: attachments of dynamic rendering
    for(const auto& target : pPipe->second.colorTargets)
    {
      nlohmann::json targetJson;
      targetJson["Name"]   = target.member;
      targetJson["Format"] = TargetFormat(a_classInfo, target.member);
      targetJson["IsRawUint4"] = (targetJson["Format"] == "VK_FORMAT_R32G32B32A32_UINT");
      targetJson["IsUint"] = (targetJson["Format"] == "VK_FORMAT_R32_UINT" || targetJson["IsRawUint4"].get<bool>());
      drawJson["PassTargets"].push_back(targetJson);
    }
    drawJson["DepthTarget"] = pPipe->second.depthTarget;

    drawJson["ColorTargets"] = std::vector<nlohmann::json>();
    if(draw.psKernel != "")
    {
      for(const auto& target : pPipe->second.colorTargets)
      {
        nlohmann::json targetJson;
        const std::string format = TargetFormat(a_classInfo, target.member);
        targetJson["IsUint"] = (format == "VK_FORMAT_R32_UINT" || format == "VK_FORMAT_R32G32B32A32_UINT");
        drawJson["ColorTargets"].push_back(targetJson);
      }
    }

    drawJson["UserArgs"] = std::vector<nlohmann::json>();
    const auto pVS = a_classInfo.kernels.find(draw.vsKernel);
    if(pVS != a_classInfo.kernels.end())
    {
      for(const auto& arg : pVS->second.args)
      {
        nlohmann::json argJson;
        std::string typeName = arg.type;
        ReplaceFirst(typeName, "const ", "");
        argJson["Type"] = typeName;
        argJson["Name"] = arg.name;
        drawJson["UserArgs"].push_back(argJson);
      }
    }
    a_hostJson["GPDraws"].push_back(drawJson);
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////// Vulkan: commands in control function /////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::string kslicer::MainFunctionRewriterVulkan::MakeGraphicsCommandCmdString(clang::CallExpr* call, const GPCommand& a_cmd)
{
  std::stringstream strOut;
  if(a_cmd.kind == GPCommand::KIND::CLEAR)
  {
    strOut << "{ vkCmdPipelineBarrier(a_commandBuffer, prevStageBits, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &m_gpBarrier, 0, nullptr, 0, nullptr);" << std::endl;
    strOut << "  Clear_" << a_cmd.pipeline << "Cmd(a_commandBuffer);" << std::endl;
    strOut << "  vkCmdPipelineBarrier(a_commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &m_gpBarrier, 0, nullptr, 0, nullptr);" << std::endl;
    strOut << "  prevStageBits = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT; }";
    return strOut.str();
  }

  // each shader has its own descriptor set: vertex shader uses set 0, pixel shader uses set 1
  //
  auto descriptorSetOf = [this](const std::string& a_shaderName) -> size_t
  {
    const auto pKernel = m_kernels.find(a_shaderName);
    assert(pKernel != m_kernels.end());
    std::vector<ArgReferenceOnCall> noArgs; // shaders have no buffer arguments, only class members
    const std::string callSign = a_shaderName + ":" + MakeKernellCallSignature(m_mainFuncName, noArgs, pKernel->second.usedContainers);
    auto pFound = dsIdBySignature.find(callSign);
    if(pFound == dsIdBySignature.end())
    {
      dsIdBySignature[callSign] = uint32_t(allDescriptorSetsInfo.size());
      KernelCallInfo dsCall;
      dsCall.kernelName       = m_pCodeInfo->RemoveKernelPrefix(a_shaderName);
      dsCall.originKernelName = a_shaderName;
      dsCall.callerName       = m_mainFuncName;
      allDescriptorSetsInfo.push_back(dsCall);
      pFound = dsIdBySignature.find(callSign);
    }
    return pFound->second;
  };

  // counts which read data of the class are computed on GPU (indirect draw, see ProcessGraphicsCommands), other counts on host
  //
  const auto& indirectDraws = m_pCodeInfo->graphics.indirectDrawByCall;
  const auto  pIndirect     = indirectDraws.find(GraphicsCallKey(m_mainFuncName, call, m_compiler.getSourceManager()));
  const bool  indirect      = (pIndirect != indirectDraws.end());
  const size_t countsNum    = std::min<size_t>(a_cmd.instanced ? 2 : 1, a_cmd.countAndArgs.size());
  std::string hostCounts[2] = {"0u", "1u"};
  for(size_t i = 0; i < countsNum; i++)
  {
    const auto pCount = static_cast<const clang::Expr*>(a_cmd.countAndArgs[i]);
    if(!indirect || !ReadsClassData(pCount))
      hostCounts[i] = "uint32_t(" + RecursiveRewrite(pCount) + ")";
  }

  const size_t dsVS = descriptorSetOf(a_cmd.vsKernel);
  if(indirect)
  {
    strOut << "{ vkCmdPipelineBarrier(a_commandBuffer, prevStageBits, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, 0, 1, &m_gpBarrier, 0, nullptr, 0, nullptr);" << std::endl;
    strOut << "  GPUpdateIndirectCmd(a_commandBuffer, " << pIndirect->second << ", " << hostCounts[0] << ", " << hostCounts[1] << "); // counts are read from class data on GPU" << std::endl;
    strOut << "  vkCmdPipelineBarrier(a_commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1, &m_gpBarrier, 0, nullptr, 0, nullptr);" << std::endl;
  }
  else
    strOut << "{ vkCmdPipelineBarrier(a_commandBuffer, prevStageBits, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, 0, 1, &m_gpBarrier, 0, nullptr, 0, nullptr);" << std::endl;
  strOut << "  " << GraphicsDrawName(a_cmd) << "Cmd(a_commandBuffer, m_allGeneratedDS[" << dsVS << "], ";
  if(a_cmd.psKernel != "")
    strOut << "m_allGeneratedDS[" << descriptorSetOf(a_cmd.psKernel) << "]";
  else
    strOut << "VK_NULL_HANDLE";
  if(a_cmd.indexed)
    strOut << ", m_vdata." << a_cmd.indexBuffer << "Buffer";
  for(size_t i = 0; i < countsNum; i++)
    strOut << ", " << (indirect ? std::string("0u") : hostCounts[i]);
  strOut << ", " << (indirect ? std::to_string(pIndirect->second) : std::string("GP_DIRECT_DRAW"));
  for(size_t i = countsNum; i < a_cmd.countAndArgs.size(); i++) // additional arguments of vertex shader
    strOut << ", " << RecursiveRewrite(static_cast<const clang::Expr*>(a_cmd.countAndArgs[i]));
  strOut << ");" << std::endl;
  strOut << "  vkCmdPipelineBarrier(a_commandBuffer, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &m_gpBarrier, 0, nullptr, 0, nullptr);" << std::endl;
  strOut << "  prevStageBits = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT; }";
  return strOut.str();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////// ISPC: draws executed by LiteRS::DrawBackend ///////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// number of float components of varyings field type: float, float2, float3, float4
static int FloatComponents(clang::QualType a_type)
{
  std::string name = a_type.getCanonicalType().getUnqualifiedType().getAsString();
  for(const std::string& prefix : {"struct ", "class ", "LiteMath::"})
    ReplaceFirst(name, prefix, "");
  if(name == "float")  return 1;
  if(name == "float2") return 2;
  if(name == "float3") return 3;
  if(name == "float4") return 4;
  return 0;
}

/// C++ type of vertex shader argument as in detail::UserArgsOf: without reference and const
static std::string DecayedTypeName(clang::QualType a_type)
{
  std::string name = a_type.getNonReferenceType().getCanonicalType().getUnqualifiedType().getAsString();
  for(const std::string& prefix : {"struct ", "class "})
    ReplaceFirst(name, prefix, "");
  return name;
}

void kslicer::AddGraphicsToISPCJson(nlohmann::json& a_kernelsJson, const MainClassInfo& a_classInfo)
{
  a_kernelsJson["GPDrawsISPC"]     = std::vector<nlohmann::json>();
  a_kernelsJson["GPPipelinesISPC"] = std::vector<std::string>();
  a_kernelsJson["HasGraphics"]     = a_classInfo.graphics.Used();
  if(!a_classInfo.graphics.Used())
    return;

  for(const auto& pipe : a_classInfo.graphics.pipelines)
    a_kernelsJson["GPPipelinesISPC"].push_back(pipe.second.name);

  for(const auto& drawPair : a_classInfo.graphics.draws)
  {
    const GPDrawInfo& draw = drawPair.second;
    const auto pPipe = a_classInfo.graphics.pipelines.find(draw.pipeline);
    const auto pVS   = a_classInfo.kernels.find(draw.vsKernel);
    if(pPipe == a_classInfo.graphics.pipelines.end() || pVS == a_classInfo.kernels.end() || pVS->second.astNode == nullptr)
      continue;
    if(draw.instanced) // LiteRS::DrawBackend does not support instancing yet, LiteRS executes such draws by C++ code
      continue;
    const clang::CXXMethodDecl* pVSDecl = pVS->second.astNode;
    const bool hasPS = (draw.psKernel != "");

    nlohmann::json drawJson;
    drawJson["Name"]        = draw.name;
    drawJson["Pipeline"]    = draw.pipeline;
    drawJson["VSOriginal"]  = draw.vsKernel;
    drawJson["PSOriginal"]  = draw.psKernel;
    drawJson["HasPS"]       = hasPS;
    drawJson["Indexed"]     = draw.indexed;
    drawJson["HasDepth"]    = (pPipe->second.depthTarget != "");

    // additional arguments of vertex shader: between vertex id and varyings
    //
    const unsigned numParams = pVSDecl->getNumParams();
    const unsigned lastArg   = hasPS ? numParams - 1 : numParams;
    drawJson["UserArgs"] = std::vector<nlohmann::json>();
    for(unsigned i = 1; i < lastArg; i++)
    {
      const clang::ParmVarDecl* pParam = pVSDecl->getParamDecl(i);
      nlohmann::json argJson;
      argJson["Name"]     = pParam->getNameAsString();
      argJson["Type"]     = DecayedTypeName(pParam->getType());
      argJson["TypeISPC"] = a_classInfo.pShaderFuncRewriter->RewriteStdVectorTypeStr(DecayedTypeName(pParam->getType()));
      drawJson["UserArgs"].push_back(argJson);
    }

    // varyings as an array of floats: every component of every field
    //
    int numFloats = 0;
    drawJson["Varyings"]     = "";
    drawJson["VaryingComps"] = std::vector<nlohmann::json>();
    if(hasPS && numParams >= 2 && pVSDecl->getParamDecl(numParams-1)->getType()->isPointerType())
    {
      const clang::QualType varType = pVSDecl->getParamDecl(numParams-1)->getType()->getPointeeType();
      drawJson["Varyings"] = ShaderTypeName(varType, a_classInfo);
      if(const clang::RecordDecl* pRecord = varType->getAsRecordDecl())
      {
        for(const clang::FieldDecl* pField : pRecord->fields())
        {
          const int comps = FloatComponents(pField->getType());
          const char* names[4] = {"x", "y", "z", "w"};
          for(int c = 0; c < comps; c++)
          {
            nlohmann::json compJson;
            compJson["Access"] = pField->getNameAsString() + ((comps == 1) ? std::string("") : std::string(".") + names[c]);
            compJson["Offset"] = numFloats;
            drawJson["VaryingComps"].push_back(compJson);
            numFloats++;
          }
        }
      }
    }
    const int arrayFloats = std::max(numFloats, 1); // detail::FloatArray<N> needs N >= 1
    for(auto& comp : drawJson["VaryingComps"])
    {
      comp["Offset1"] = comp["Offset"].get<int>() + arrayFloats;
      comp["Offset2"] = comp["Offset"].get<int>() + 2*arrayFloats;
    }
    drawJson["NumFloats"]  = arrayFloats;
    drawJson["NumFloats3"] = 3*arrayFloats;

    // pixel shader outputs and color targets
    //
    drawJson["Outputs"]        = std::vector<nlohmann::json>();
    drawJson["OutputIsStruct"] = false;
    drawJson["OutputType"]     = "float4";
    const auto pPS = hasPS ? a_classInfo.kernels.find(draw.psKernel) : a_classInfo.kernels.end();
    if(pPS != a_classInfo.kernels.end() && pPS->second.astNode != nullptr)
    {
      const clang::QualType retType = pPS->second.astNode->getReturnType();
      const std::string     retName = ShaderTypeName(retType, a_classInfo);
      const clang::CXXRecordDecl* pRecord = retType->getAsCXXRecordDecl();
      std::vector<std::string> outNames;
      if(retName == "float4" || pRecord == nullptr)
        outNames.push_back("");
      else
      {
        drawJson["OutputIsStruct"] = true;
        drawJson["OutputType"]     = retName;
        for(const clang::FieldDecl* pField : pRecord->fields())
          outNames.push_back(pField->getNameAsString());
      }
      for(size_t i = 0; i < outNames.size() && i < pPipe->second.colorTargets.size(); i++)
      {
        nlohmann::json outJson;
        outJson["Field"]  = outNames[i];
        outJson["IsUint"] = (TargetFormat(a_classInfo, pPipe->second.colorTargets[i].member) == "VK_FORMAT_R32_UINT");
        drawJson["Outputs"].push_back(outJson);
      }
    }
    a_kernelsJson["GPDrawsISPC"].push_back(drawJson);
  }
}
