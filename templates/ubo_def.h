#ifndef {{MainClassName}}_UBO_H
#define {{MainClassName}}_UBO_H

{% if ShaderGLSL %}
#ifndef GLSL
#define LAYOUT_STD140
#include "LiteMath.h"
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
#define MAXFLOAT 1e37f
#define M_PI          3.14159265358979323846f
#define M_TWOPI       6.28318530717958647692f
#define INV_PI        0.31830988618379067154f
#define INV_TWOPI     0.15915494309189533577f
#endif
{% else %}
#ifndef GLSL
  #ifdef ISPC
  typedef float<2> float2;
  typedef float<3> float3;
  typedef float<4> float4;

  typedef int<2>   int2;
  typedef int<3>   int3;
  typedef int<4>   int4;

  typedef uint<2>  uint2;
  typedef uint<3>  uint3;
  typedef uint<4>  uint4;

  typedef uint8    uint8_t;
  typedef uint16   uint16_t;
  typedef uint32   uint32_t;
  #define __global 
  #include "LiteMathISPC.h"
  #else
  #include "LiteMath.h"
  #endif
#else
#define float4x4 mat4
#define float2   vec2
#define float3   vec3
#define float4   vec4
#define int2     ivec2
#define int3     ivec3
#define int4     ivec4
#define uint2    uvec2
#define uint3    uvec3
#define uint4    uvec4
#define uint32_t uint
#define int32_t  int
#endif
{% endif %}

struct {{MainClassName}}_UBO_Data
{
## for Field in UBOStructFields  
  {% if Field.IsDummy %} 
  #ifndef GLSL 
  uint {{Field.Name}}; 
  #endif 
  {% else %}
  {{Field.Type}} {{Field.Name}}{% if Field.IsArray %}[{{Field.ArraySize}}]{% endif %}; 
  {% if Field.IsVec3 %}
  #ifdef GLSL 
  uint {{Field.Name}}Dummy; 
  #endif 
  {% endif %}
  {% endif %}
## endfor
  uint dummy_last;
{% for hierarchy in Hierarchies %}
{% if hierarchy.IndirectDispatch %}

  uint objNum_{{hierarchy.Name}}Src[{{hierarchy.ImplAlignedSize}}];  
  uint objNum_{{hierarchy.Name}}Acc[{{hierarchy.ImplAlignedSize}}];
  uint objNum_{{hierarchy.Name}}Off[{{hierarchy.ImplAlignedSize}}];
{% endif %}  
{% endfor %}
};

#endif
