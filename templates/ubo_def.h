#ifndef {{MainClassName}}_UBO_H
#define {{MainClassName}}_UBO_H

#ifndef GLSL
#include "OpenCLMath.h"
#else
#define MAXFLOAT 1e37f
//#define float4x4 mat4
//#define float3   vec3
//#define float4   vec4
//#define uint32_t uint
#endif

struct {{MainClassName}}_UBO_Data
{
## for Field in UBOStructFields  
  {{Field.Type}} {{Field.Name}}{% if Field.IsArray %}[{{Field.ArraySize}}]{% endif %};
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
