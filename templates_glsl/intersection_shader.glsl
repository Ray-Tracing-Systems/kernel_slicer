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
{% for Remap in Kernel.IntersectionShaderRemaps %}

layout(buffer_reference, std430, buffer_reference_align = 16) buffer {{Remap.Name}}Remap
{
	{{Remap.DType}} {{Remap.Name}}_table[];
};

layout(buffer_reference, std430, buffer_reference_align = 16) buffer {{Remap.Name}}Tags
{
	uint {{Remap.Name}}_gtags[];
};
{% endfor %}

{% if HasAllRefs %}
struct AllBufferReferences
{
  {% for ImplS in Hierarchy.Implementations %}
  {{ImplS.DataStructure.Name}}Buffer {{ImplS.DataStructure.Name}}_buffer;
  {% endfor %}
  {% for Remap in Kernel.IntersectionShaderRemaps %}
  {{Remap.Name}}Remap {{Remap.Name}}_remap;
  {{Remap.Name}}Tags  {{Remap.Name}}_gtags;
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
{% for RTName in Kernel.RTXNames %}
CRT_Hit {{RTName}}_RayQuery_NearestHit(vec4 rayPos, vec4 rayDir) { CRT_Hit res; return res; }                // just to make code compilable
CRT_Hit {{RTName}}_RayQuery_NearestHitMotion(vec4 rayPos, vec4 rayDir, float t) { CRT_Hit res; return res; } // just to make code compilable
bool {{RTName}}_RayQuery_AnyHit(vec4 rayPos, vec4 rayDir) { return false; }                                  // just to make code compilable
bool {{RTName}}_RayQuery_AnyHitMotion(vec4 rayPos, vec4 rayDir, float t){ return false; }                    // just to make code compilable
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
  vec4  rayPosAndNear = vec4(gl_ObjectRayOriginEXT,    gl_RayTminEXT);
  vec4  rayDirAndFar  = vec4(gl_ObjectRayDirectionEXT, gl_RayTmaxEXT);
  uvec2 remap         = all_references.{{Kernel.IntersectionHierarhcy.Name}}_remap.{{Kernel.IntersectionHierarhcy.Name}}_table[gl_InstanceCustomIndexEXT];
  CRT_LeafInfo info;
  info.aabbId = gl_PrimitiveID;  
  info.primId = gl_PrimitiveID/remap.y;
  info.instId = gl_InstanceID; 
  info.geomId = gl_InstanceCustomIndexEXT; 
  info.rayxId = gl_LaunchIDEXT[0];
  info.rayyId = gl_LaunchIDEXT[1]; 
  attribs.t   = 1e6f;  
  attribs.primId = 0xFFFFFFFF; 
  attribs.instId = 0xFFFFFFFF;
  attribs.geomId = 0xFFFFFFFF;     
  uint intersected = {{IntersectionShader.NameRewritten}}(remap.x + info.primId, rayPosAndNear, rayDirAndFar, info, attribs);
  if(intersected != {{Kernel.IntersectionHierarhcy.EmptyImplementation.TagName}}) 
    reportIntersectionEXT(attribs.t, 0);
}
