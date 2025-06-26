#pragma once

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <array>

#include <webgpu/webgpu.h>
#include <webgpu/wgpu.h>

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

  virtual void InitWulkanObjects(WGPUDevice a_device, WGPUAdapter a_physicalDevice, size_t a_maxThreads);
  
protected:

  WGPUAdapter  physicalDevice = nullptr;
  WGPUDevice   device         = nullptr;

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
};

