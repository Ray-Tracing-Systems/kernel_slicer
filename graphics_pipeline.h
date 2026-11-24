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
  class SourceManager;
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
    bool        indexed   = false;
    bool        instanced = false; ///<! DrawInstancedTriangles, DrawIndexedInstancedTriangles
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
    bool                     indexed   = false;
    bool                     instanced = false;
    std::string              indexBuffer;
    const void*              indexOffset = nullptr; ///<! clang::Expr* of offset in index buffer: 'm_indices.data() + offset', '&m_indices[offset]'
    std::vector<const void*> countAndArgs;  ///<! clang::Expr* of vertex/index count, [instance count,] additional vertex shader arguments
  };

  /**
  \brief Call of a draw command with counts which are read from data of the class (members, 'm_vec.size()'): the counts are
         known only on GPU, so they are written to indirect buffer by a compute shader and the draw is vkCmdDraw[Indexed]Indirect.
         Counts from arguments and local variables of control function are passed to that shader as push constants.
  */
  struct GPIndirectDraw
  {
    uint32_t    id      = 0;     ///<! index of the command in indirect buffer
    bool        indexed = false;
    std::string countExpr;       ///<! shader expression of vertex (index) count, empty if it is computed on host
    std::string instExpr;        ///<! shader expression of instance count, empty if it is computed on host (or is 1)
  };

  struct GraphicsInfo
  {
    kslicer::umap<std::string, GPPipelineInfo> pipelines; ///<! by member name
    kslicer::umap<std::string, GPDrawInfo>     draws;     ///<! by GPDrawInfo::name
    kslicer::uset<std::string>                 indexBuffers; ///<! member vectors used as index buffers; calls of one draw may use different index buffers
    std::vector<GPIndirectDraw>                indirectDraws;
    kslicer::umap<std::string, uint32_t>       indirectDrawByCall; ///<! 'ControlFunc:line:column' of command call --> GPIndirectDraw::id
    bool Used() const { return !draws.empty(); }
  };

  GP_STAGE    GraphicsStageByKernelName(const std::string& a_kernelName);
  std::string GraphicsShortKernelName(const std::string& a_kernelName);   ///<! 'kernelVS_XXX' --> 'VS_XXX', otherwise empty string
  bool        IsHostOnlyMemberType(const std::string& a_typeName);         ///<! LiteRS objects are used on host only, they are not placed to UBO

  /// parses LiteRS command, returns false if a_call is not a LiteRS command (or is an unknown LiteRS function)
  bool        ParseGraphicsCommand(const clang::CallExpr* a_call, GPCommand& a_cmd);

  /// returns true if a_call is a call of a function from namespace LiteRS
  bool        IsLiteRSCall(const clang::CallExpr* a_call);

  /// key of a command call for GraphicsInfo::indirectDrawByCall
  std::string GraphicsCallKey(const std::string& a_controlFunc, const clang::CallExpr* a_call, const clang::SourceManager& a_sm);

  /// name of generated function for draw command 'Draw_<VS>_<PS>_<pipeline>'
  std::string GraphicsDrawName(const GPCommand& a_cmd);

  /// finds render targets of all LiteRS::Pipeline members in all functions of the main class
  void        ProcessGraphicsPipelines(MainClassInfo& a_codeInfo, const clang::CompilerInstance& a_compiler);

  /// finds LiteRS commands in control function, marks used shaders and registers draws; returns false if some command is invalid
  bool        ProcessGraphicsCommands(MainFuncInfo& a_mainFunc, MainClassInfo& a_codeInfo, const clang::CompilerInstance& a_compiler);

  /// removes arguments of shaders which are not passed by host (vertex id, varyings) after arguments are processed
  void        PrepareGraphicsKernel(KernelInfo& a_kernel);

  /// adds shader functions to the list of member functions of their kernels, so they are emitted to shader code
  void        AddGraphicsShadersAsMemberFunctions(MainClassInfo& a_codeInfo);

  /// returns true if a_member is a render target; a_isDepth is set for depth target
  bool        IsGraphicsRenderTarget(const MainClassInfo& a_codeInfo, const std::string& a_member, bool* a_isDepth = nullptr);
};
