#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_buffer_reference     : require
#extension GL_EXT_ray_tracing          : require

#include "common{{MainClassSuffixLowerCase}}.h"

{% for Hierarchy in Kernel.Hierarchies %} 
{% if Hierarchy.VFHLevel >= 2 and HasAllRefs %}
{% for ImplS in Hierarchy.Implementations %}

struct {{ImplS.DataStructure.Name}}
{
  {% for Field in ImplS.DataStructure.Fields %}
  {{Field.Type}} {{Field.Name}};
  {% endfor %}  
};
{% endfor %}
{% for ImplS in Hierarchy.Implementations %}

layout(buffer_reference, std430, buffer_reference_align = 16) buffer {{ImplS.DataStructure.Name}}Buffer
{
	{{ImplS.DataStructure.Name}} {{ImplS.DataStructure.BufferName}}[];
};
{% endfor %}
{% endif %}

{% if HasAllRefs %}
struct AllBufferReferences
{
  {% for ImplS in Hierarchy.Implementations %}
  {{ImplS.DataStructure.Name}}Buffer {{ImplS.DataStructure.Name}}_buffer;
  {% endfor %}
};
{% endif %}

{% endfor %}
## for Arg in Kernel.Args
{% if not Arg.IsUBO %} 
{% if Arg.IsImage %}
layout(binding = {{loop.index}}, set = 0{% if Arg.NeedFmt%}, {{Arg.ImFormat}}{% endif %}) uniform {{Arg.Type}} {{Arg.Name}}; //
{% else if Arg.IsAccelStruct %}
//layout(binding = {{loop.index}}, set = 0) uniform accelerationStructureEXT {{Arg.Name}}; // can't be used inside intersection shader
{% else %}
layout(binding = {{loop.index}}, set = 0) buffer data{{loop.index}} { {{Arg.Type}} {{Arg.Name}}{% if not Arg.IsSingle %}[]{% endif %}; }; //
{% endif %} {# /* Arg.IsImage */ #}
{% endif %} {# /* not Arg.IsUBO */ #}
## endfor
layout(binding = {{length(Kernel.Args)}}, set = 0) buffer dataUBO { {{MainClassName}}{{MainClassSuffix}}_UBO_Data ubo; };

{% for Array in Kernel.ThreadLocalArrays %}
{{Array.Type}} {{Array.Name}}[{{Array.Size}}];
{% endfor %}

{% for ShitFunc in Kernel.ShityFunctions %} 
{{ShitFunc}}

{% endfor %}
{% for MembFunc in Kernel.MemberFunctions %}
{% if not MembFunc.IsRayQuery %}
{{MembFunc.Decl}};
{% endif %}
{% endfor %}

{% for Hierarchy in Kernel.Hierarchies %} {# /*------------------------------ vfh ------------------------------ */ #}
// Virtual Functions of {{Hierarchy.Name}}:
{% for Contant in Hierarchy.Constants %}
{{Contant.Type}} {{Contant.Name}} = {{Contant.Value}};
{% endfor %} 

{% for RetDecl in Hierarchy.AuxDecls %}
struct {{RetDecl.Name}} 
{
  {% for Field in RetDecl.Fields %}
  {{Field.Type}} {{Field.Name}};
  {% endfor %}
};

{% endfor %}

{% for Member in Implementation.MemberFunctions %}
{{Member.Source}}

{% endfor %}
{% endfor %}                        {# /*------------------------------ vfh ------------------------------ */ #}
{% for MembFunc in Kernel.MemberFunctions %}
{% if not MembFunc.IsRayQuery %}
{{MembFunc.Text}}
{% endif %}
{% endfor %} 

hitAttributeEXT CRT_Hit attribs;

void main()
{ 
  vec4 rayPosAndNear  = vec4(gl_ObjectRayOriginEXT,    gl_RayTminEXT);
  vec4 rayDirAndFar   = vec4(gl_ObjectRayDirectionEXT, gl_RayTmaxEXT);
  uint intersected    = {{IntersectionShader.NameRewritten}}(gl_PrimitiveID, rayPosAndNear, rayDirAndFar, attribs);
  if(intersected != TAG_EMPTY)
    reportIntersectionEXT(attribs.t, 0);
}
