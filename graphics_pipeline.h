#pragma once
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Graphics pipeline (LiteRS, apps/LiteMathAux/raster.h): vertex and pixel shaders of the main class and LiteRS
// commands in control functions. Most of the logic is in graphics_pipeline.cpp; the rest of the translator calls
// the functions declared here at a few points.
//
// Vertex and pixel shaders are kernels with names 'kernelVS_*' and 'kernelPS_*' (KernelInfo::gpStage != NONE).
// They are not called from control functions directly: LiteRS::DrawTriangles(...) names them. Their bodies are
// emitted to shaders as ordinary member functions, generated entry points call them.
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <string>
#include <vector>
#include <cstdint>
#include "kslicer_containers.h"

namespace clang
{
  class CompilerInstance;
  class CallExpr;
};

namespace kslicer
{
  struct KernelInfo;
  struct MainClassInfo;
  struct MainFuncInfo;

  enum class GP_STAGE { NONE = 0, VERTEX = 1, PIXEL = 2 };

  struct GPColorTarget
  {
    std::string member;      ///<! name of Image2D member
    uint32_t    index = 0;   ///<! color attachment index
  };

  /**
  \brief LiteRS::Pipeline member of the main class and its render targets found in 'SetColorTarget' / 'SetDepthTarget' calls
  */
  struct GPPipelineInfo
  {
    std::string                name;
    std::vector<GPColorTarget> colorTargets; ///<! sorted by index
    std::string                depthTarget;  ///<! empty if there is no depth target
  };

  /**
  \brief Unique draw: a pair of shaders with a pipeline object; several calls in control functions may share it
  */
  struct GPDrawInfo
  {
    std::string name;          ///<! name of generated function 'Draw_<VS>_<PS>_<pipeline>'
    std::string pipeline;      ///<! LiteRS::Pipeline member
    std::string vsKernel;      ///<! 'kernelVS_XXX'
    std::string psKernel;      ///<! 'kernelPS_XXX', empty for depth only pass
    bool        indexed = false;
    std::string indexBuffer;   ///<! member vector with indices for indexed draw
  };

  /**
  \brief LiteRS command in control function, parsed from AST
  */
  struct GPCommand
  {
    enum class KIND { UNKNOWN = 0, CLEAR = 1, DRAW = 2 };
    KIND                     kind = KIND::UNKNOWN;
    std::string              pipeline;
    std::string              vsKernel;
    std::string              psKernel;
    bool                     indexed = false;
    std::string              indexBuffer;
    std::vector<const void*> countAndArgs;  ///<! clang::Expr* of vertex/index count and of additional vertex shader arguments
  };

  struct GraphicsInfo
  {
    kslicer::umap<std::string, GPPipelineInfo> pipelines; ///<! by member name
    kslicer::umap<std::string, GPDrawInfo>     draws;     ///<! by GPDrawInfo::name
    bool Used() const { return !draws.empty(); }
  };

  GP_STAGE    GraphicsStageByKernelName(const std::string& a_kernelName);
  std::string GraphicsShortKernelName(const std::string& a_kernelName);   ///<! 'kernelVS_XXX' --> 'VS_XXX', otherwise empty string
  bool        IsHostOnlyMemberType(const std::string& a_typeName);         ///<! LiteRS objects are used on host only, they are not placed to UBO

  /// parses LiteRS command, returns false if a_call is not a LiteRS command
  bool        ParseGraphicsCommand(const clang::CallExpr* a_call, GPCommand& a_cmd);

  /// name of generated function for draw command 'Draw_<VS>_<PS>_<pipeline>'
  std::string GraphicsDrawName(const GPCommand& a_cmd);

  /// finds render targets of all LiteRS::Pipeline members in all functions of the main class
  void        ProcessGraphicsPipelines(MainClassInfo& a_codeInfo, const clang::CompilerInstance& a_compiler);

  /// finds LiteRS commands in control function, marks used shaders and registers draws
  void        ProcessGraphicsCommands(MainFuncInfo& a_mainFunc, MainClassInfo& a_codeInfo, const clang::CompilerInstance& a_compiler);

  /// removes arguments of shaders which are not passed by host (vertex id, varyings) after arguments are processed
  void        PrepareGraphicsKernel(KernelInfo& a_kernel);

  /// adds shader functions to the list of member functions of their kernels, so they are emitted to shader code
  void        AddGraphicsShadersAsMemberFunctions(MainClassInfo& a_codeInfo);

  /// returns true if a_member is a render target; a_isDepth is set for depth target
  bool        IsGraphicsRenderTarget(const MainClassInfo& a_codeInfo, const std::string& a_member, bool* a_isDepth = nullptr);
};
