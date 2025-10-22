#pragma once

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <array>
#include "wk_context.h"

{% if length(TextureMembers) > 0 or length(ClassTexArrayVars) > 0 %}
#include "Image2d.h"
using LiteImage::Image2D;
using LiteImage::Sampler;
using namespace LiteMath;
{% endif %}

#include "{{MainInclude}}"
{% for Include in AdditionalIncludes %}
#include "{{Include}}"
{% endfor %}

## for Decl in ClassDecls
{% if Decl.InClass and Decl.IsType %}
using {{Decl.Type}} = {{MainClassName}}::{{Decl.Type}}; // for passing this data type to UBO
{% endif %}
## endfor

/////////////////////////////////////////////////////////////////////////////////////////// UBO

#include "LiteMath.h"
#ifndef CUDA_MATH
using   LiteMath::uint;
typedef LiteMath::float4x4 mat4;
typedef LiteMath::float2   vec2;
typedef LiteMath::float3   vec3;
typedef LiteMath::float4   vec4;
typedef LiteMath::int2     ivec2;
typedef LiteMath::int3     ivec3;
typedef LiteMath::int4     ivec4;
typedef LiteMath::uint2    uvec2;
typedef LiteMath::uint3    uvec3;
typedef LiteMath::uint4    uvec4;
#else
//typedef float4x4 mat4;
typedef float2   vec2;
typedef float3   vec3;
typedef float4   vec4;
typedef int2     ivec2;
typedef int3     ivec3;
typedef int4     ivec4;
typedef uint2    uvec2;
typedef uint3    uvec3;
typedef uint4    uvec4;
#endif

struct {{MainClassName}}{{MainClassSuffix}}_UBO_Data
{
  {% for Field in UBO.UBOStructFields %}
  {% if Field.IsDummy %} 
  uint {{Field.Name}}; 
  {% else %}
  {{Field.Type}} {{Field.Name}}{% if Field.IsArray %}[{{Field.ArraySize}}]{% endif %};
  {% endif %}
  {% endfor %}
  uint dummy_last;
};

class {{MainClassName}}{{MainClassSuffix}} : public {{MainClassName}}
{
public:

  {% for ctorDecl in Constructors %}
  {% if ctorDecl.NumParams == 0 %}
  {{ctorDecl.ClassName}}{{MainClassSuffix}}()
  {
    {% if HasPrefixData %}
    if({{PrefixDataName}} == nullptr)
      {{PrefixDataName}} = std::make_shared<{{PrefixDataClass}}>();
    {% endif %}
  }
  {% else %}
  {{ctorDecl.ClassName}}{{MainClassSuffix}}({{ctorDecl.Params}}) : {{ctorDecl.ClassName}}({{ctorDecl.PrevCall}})
  {
    {% if HasPrefixData %}
    if({{PrefixDataName}} == nullptr)
      {{PrefixDataName}} = std::make_shared<{{PrefixDataClass}}>();
    {% endif %}
  }
  {% endif %}
  {% endfor %}
  virtual ~{{MainClassName}}{{MainClassSuffix}}();
  
  virtual void InitWulkanObjects(WGPUDevice a_device, WGPUAdapter a_physicalDevice, size_t a_maxThreads);

  {% for MainFunc in MainFunctions %}
  virtual void SetWulkanInOutFor_{{MainFunc.Name}}(
    {% for Arg in MainFunc.InOutVars %}
    {% if Arg.IsTexture %}
    WGPUImage     {{Arg.Name}}Text,
    WGPUImageView {{Arg.Name}}View,
    {% else %}
    WGPUBuffer {{Arg.Name}}Buffer,
    size_t     {{Arg.Name}}Offset,
    size_t     {{Arg.Name}}Size,
    {% endif %}
    {% endfor %}
    uint32_t dummyArgument = 0)
  {
    {% for Arg in MainFunc.InOutVars %}
    {% if Arg.IsTexture %}
    {{MainFunc.Name}}_local.{{Arg.Name}}Text   = {{Arg.Name}}Text;
    {{MainFunc.Name}}_local.{{Arg.Name}}View   = {{Arg.Name}}View;
    {% else %}
    {{MainFunc.Name}}_local.{{Arg.Name}}Buffer = {{Arg.Name}}Buffer;
    {{MainFunc.Name}}_local.{{Arg.Name}}Offset = {{Arg.Name}}Offset;
    {{MainFunc.Name}}_local.{{Arg.Name}}Size   = {{Arg.Name}}Size;
    {% endif %}
    {% endfor %}
    UpdateAllBindingGroup_{{MainFunc.Name}}();
  }

  {% endfor %}
  
protected:

  WGPUAdapter  physicalDevice = nullptr;
  WGPUDevice   device         = nullptr;
  WGPUQueue    queue          = nullptr;

  {% for Kernel in Kernels %}
  WGPUComputePipeline {{Kernel.Name}}Pipeline = nullptr;
  WGPUBindGroupLayout {{Kernel.Name}}DSLayout = nullptr;
  {% if Kernel.HasLoopInit %}
  WGPUComputePipeline {{Kernel.Name}}InitPipeline = nullptr;
  {% endif %}
  {% if Kernel.HasLoopFinish %}
  WGPUComputePipeline {{Kernel.Name}}FinishPipeline = nullptr;
  {% endif %}
  {% if Kernel.FinishRed %}
  WGPUComputePipeline {{Kernel.Name}}ReductionPipeline = nullptr;
  {% endif %}
  virtual void InitKernel_{{Kernel.Name}}(const char* a_filePath);
  {% if Kernel.IsIndirect %}
  virtual void {{Kernel.Name}}_UpdateIndirect();
  {% endif %}
  {% endfor %}

  virtual void InitKernels(const char* a_path);
  virtual void InitDeviceData();

  virtual void UpdatePlainMembers();
  virtual void UpdateVectorMembers();
  {% for UpdateFun in UpdateVectorFun %}
  {% if UpdateFun.NumParams == 2 %}
  void {{UpdateFun.Name}}(size_t a_first, size_t a_size) override;
  {% else %}
  void {{UpdateFun.Name}}() override;
  {% endif %}
  {% endfor %}

  void CommitDeviceData() override 
  {
    InitDeviceData();
    UpdatePlainMembers();
    UpdateVectorMembers();
  }

  void GetExecutionTime(const char* a_funcName, float a_out[4]) override; 

  {% for KernelDecl in KernelsDecls %}
  {{KernelDecl}}
  {% endfor %}
  
  {% for MainFunc in MainFunctions %}
  {% if MainFunc.OverrideMe %}
  {{MainFunc.ReturnType}} {{MainFunc.DeclOrig}} override;
  {% endif %}
  {% endfor %}

  {% for MainFunc in MainFunctions %}
  struct {{MainFunc.Name}}_Data
  {
    {% if MainFunc.IsRTV and not MainFunc.IsMega %}
    {% for Buffer in MainFunc.LocalVarsBuffersDecl %}
    WGPUBuffer {{Buffer.Name}}Buffer = nullptr;
    size_t     {{Buffer.Name}}Offset = 0;
    {% endfor %}
    {% endif %}
    {% for Arg in MainFunc.InOutVars %}
    {% if Arg.IsTexture %}
    WGPUImage     {{Arg.Name}}Text = nullptr;
    WGPUImageView {{Arg.Name}}View = nullptr;
    {% else %}
    WGPUBuffer {{Arg.Name}}Buffer = nullptr;
    size_t     {{Arg.Name}}Offset = 0;
    size_t     {{Arg.Name}}Size   = 0;
    {% endif %}
    {% endfor %}
    bool needToClearOutput = {% if MainFunc.IsRTV %}true{% else %}false{% endif %};
  } {{MainFunc.Name}}_local;

  void UpdateAllBindingGroup_{{MainFunc.Name}}();
  virtual {{MainFunc.ReturnType}} {{MainFunc.Decl}};
  {% endfor %}

  struct MembersDataGPU
  {
    {% for Vector in VectorMembers %}
    WGPUBuffer {{Vector.Name}}Buffer = nullptr;
    size_t     {{Vector.Name}}Offset = 0;
    size_t     {{Vector.Name}}Size   = 0;
    {% endfor %}
  } m_vdata;

  WGPUBindGroup m_allGeneratedDS[{{TotalDSNumber}}];
  const size_t  m_totalDSNumber = {{TotalDSNumber}};

  WGPUBuffer    m_classDataBuffer = nullptr;
  size_t        m_classDataSize   = 0;

  WGPUBuffer    m_readBackBuffer = nullptr;
  size_t        m_readBackSize   = 0;

  WGPUBuffer    m_pushConstantBuffer = nullptr; // use this buffer to emulate push constants
  size_t        m_pushConstantStride = 256;     // each binding point have it's own offset inside this buffer
  
  WGPUCommandEncoder     m_currEncoder  = nullptr;
  WGPUComputePassEncoder m_currPassCS   = nullptr;
  size_t                 m_currPCOffset = 0;

  {{MainClassName}}{{MainClassSuffix}}_UBO_Data m_uboData;

  virtual void ReadBufferBack(WGPUBuffer a_buffer, size_t a_size, void* a_data);

  {% if HasGetResDirFunc %}
  virtual std::string AlterShaderPath(const char* in_shaderPath) { return GetResourcesRootDir() + "/" + std::string("{{ShaderFolderPrefix}}") + std::string(in_shaderPath); }
  {% else %}
  virtual std::string AlterShaderPath(const char* in_shaderPath) { return std::string("{{ShaderFolderPrefix}}") + std::string(in_shaderPath); }
  {% endif %}
}; 

