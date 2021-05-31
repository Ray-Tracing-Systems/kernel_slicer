#ifndef {{MainClassName}}_UBO_H
#define {{MainClassName}}_UBO_H

{% if ShaderGLSL %}
#ifndef GLSL
#include "OpenCLMath.h"
typedef LiteMath::float4x4 mat4;
typedef LiteMath::float3   vec3;
typedef LiteMath::float4   vec4;
#else
#define MAXFLOAT 1e37f
#endif
{% else %}
#include "OpenCLMath.h"
{% endif %}

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
