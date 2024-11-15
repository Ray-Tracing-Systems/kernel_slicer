#include "common{{MainClassSuffixLowerCase}}.h"
{% for KSpec in Kernel.SpecConstants %}
layout (constant_id = {{KSpec.Id}}) const int {{KSpec.Name}} = {{KSpec.Id}};
{% endfor %}
{% for Hierarchy in Kernel.Hierarchies %} 
{% if Hierarchy.VFHLevel <= 1 %}
struct {{Hierarchy.Name}}
{
  uint vptr_dummy[2];
  {% for Field in Hierarchy.InterfaceFields %}
  {{Field.Type}} {{Field.Name}};
  {% endfor %}
};
{% endif %}
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
{% endfor %}

{% if HasAllRefs %}
struct AllBufferReferences
{
  {% for ImplS in Hierarchy.Implementations %}
  {{ImplS.DataStructure.Name}}Buffer {{ImplS.DataStructure.Name}}_buffer;
  {% endfor %}
  {% for Remap in Kernel.IntersectionShaderRemaps %}
  {{Remap.Name}}Remap {{Remap.Name}}_remap;
  {% endfor %}
};
{% endif %}

{% endfor %}
## for Arg in Kernel.Args
{% if not Arg.IsUBO %} 
{% if Arg.IsImage %}
layout(binding = {{loop.index}}, set = 0{% if Arg.NeedFmt%}, {{Arg.ImFormat}}{% endif %}) uniform {{Arg.Type}} {{Arg.Name}}; //
{% else if Arg.IsAccelStruct %}
layout(binding = {{loop.index}}, set = 0) uniform accelerationStructureEXT {{Arg.Name}};
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
{{MembFunc.Decl}};
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

{% for Member in Hierarchy.EmptyImplementation.MemberFunctions %}

{{Member.Source}}
{% endfor %}
{% for Impl in Hierarchy.Implementations %}
{% for Member in Impl.MemberFunctions %}
{{Member.Source}}

{% endfor %}

{% for Field in Impl.Fields %}
//{{Field}}
{% endfor %}
{% endfor %}
{% for VirtualFunc in Hierarchy.VirtualFunctions %}
{{VirtualFunc.Decl}} 
{
  {% if Hierarchy.VFHLevel >= 2 %}
  const uvec2 tableVal = {{Hierarchy.ObjBufferName}}[selfId];
  const uint tag = tableVal.x;
  selfId         = tableVal.y;
  {% else %}
  const uint tag = {{Hierarchy.ObjBufferName}}[selfId].m_tag;
  {% endif %}
  switch(tag) 
  {
    {% for Impl in Hierarchy.Implementations %}
    case {{Impl.TagName}}: return {{Impl.ClassName}}_{{VirtualFunc.Name}}_{{Impl.ObjBufferName}}({% for Arg in VirtualFunc.Args %}{{Arg.Name}}{% if loop.index != VirtualFunc.ArgLen %},{% endif %}{% endfor %});
    {% endfor %}
    default: return {{Hierarchy.EmptyImplementation.ClassName}}_{{VirtualFunc.Name}}_{{Hierarchy.EmptyImplementation.ObjBufferName}}({% for Arg in VirtualFunc.Args %}{{Arg.Name}}{% if loop.index != VirtualFunc.ArgLen %},{% endif %}{% endfor %});
  };
}
{% endfor %}                                 
{% endfor %}                                 {# /*------------------------------ vfh ------------------------------ */ #}
{% for MembFunc in Kernel.MemberFunctions %}
{% if not (MembFunc.IsRayQuery and (Kernel.UseRayGen or (length(Kernel.IntersectionHierarhcy.Implementations) >= 1)) ) %}

{{MembFunc.Text}}
{% endif%}
{% endfor %} 
{% for RTName in Kernel.RTXNames %}
// RayScene intersection with '{{RTName}}'
//
{% if Kernel.UseRayGen %}
layout(location = 0) rayPayloadEXT CRT_Hit {{RTName}}_hitValue;
layout(location = 1) rayPayloadEXT bool    {{RTName}}_inShadow;

CRT_Hit {{RTName}}_RayQuery_NearestHit(vec4 rayPos, vec4 rayDir)
{
  traceRayEXT({{RTName}}, gl_RayFlagsOpaqueEXT, 0xff, 0, 0, 0, rayPos.xyz, rayPos.w, rayDir.xyz, rayDir.w, 0);
  return {{RTName}}_hitValue;
}

CRT_Hit {{RTName}}_RayQuery_NearestHitMotion(vec4 rayPos, vec4 rayDir, float t)
{
  {% if UseMotionBlur %}
  traceRayMotionNV({{RTName}}, gl_RayFlagsOpaqueEXT, 0xff, 0, 0, 0, rayPos.xyz, rayPos.w, rayDir.xyz, rayDir.w, t, 0);
  {% else %}
  traceRayEXT({{RTName}}, gl_RayFlagsOpaqueEXT, 0xff, 0, 0, 0, rayPos.xyz, rayPos.w, rayDir.xyz, rayDir.w, 0);
  {% endif %} 
  return {{RTName}}_hitValue;
}

bool {{RTName}}_RayQuery_AnyHit(vec4 rayPos, vec4 rayDir)
{
  {{RTName}}_inShadow = true;
  traceRayEXT({{RTName}}, gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT,
              0xff, 0, 0, 1, rayPos.xyz, rayPos.w, rayDir.xyz, rayDir.w, 1);
  return {{RTName}}_inShadow;
}

bool {{RTName}}_RayQuery_AnyHitMotion(vec4 rayPos, vec4 rayDir, float t)
{
  {{RTName}}_inShadow = true;
  {% if UseMotionBlur %}
  traceRayMotionNV({{RTName}}, gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT,
                   0xff, 0, 0, 1, rayPos.xyz, rayPos.w, rayDir.xyz, rayDir.w, t, 1);
  {% else %}
  traceRayEXT({{RTName}}, gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT,
              0xff, 0, 0, 1, rayPos.xyz, rayPos.w, rayDir.xyz, rayDir.w, 1);
  {% endif %}
  return {{RTName}}_inShadow;
}

{% else %}
CRT_Hit {{RTName}}_RayQuery_NearestHit(vec4 rayPos, vec4 rayDir)
{
  rayQueryEXT rayQuery;
  rayQueryInitializeEXT(rayQuery, {{RTName}}, gl_RayFlagsOpaqueEXT, 0xff, rayPos.xyz, rayPos.w, rayDir.xyz, rayDir.w);
  
  CRT_Hit res;
  res.primId = -1;
  res.instId = -1;
  res.geomId = -1;
  res.t      = rayDir.w;

  while(rayQueryProceedEXT(rayQuery)) 
  { 
    {% if length(Kernel.IntersectionHierarhcy.Implementations) >= 1 %}
    if(rayQueryGetIntersectionTypeEXT(rayQuery, false) == gl_RayQueryCandidateIntersectionTriangleEXT)
    {
      rayQueryConfirmIntersectionEXT(rayQuery);
    }
    else if (rayQueryGetIntersectionTypeEXT(rayQuery, false) == gl_RayQueryCandidateIntersectionAABBEXT)
    {
      res.primId    = rayQueryGetIntersectionPrimitiveIndexEXT(rayQuery, false);
	    res.geomId    = rayQueryGetIntersectionInstanceCustomIndexEXT(rayQuery, false);
      res.instId    = rayQueryGetIntersectionInstanceIdEXT    (rayQuery, false);
	    
      vec4  rayPosAndNear = vec4(rayQueryGetIntersectionObjectRayOriginEXT(rayQuery, false),    rayPos.w);
      vec4  rayDirAndFar  = vec4(rayQueryGetIntersectionObjectRayDirectionEXT(rayQuery, false), rayDir.w);
      uvec2 remap         = all_references.{{Kernel.IntersectionHierarhcy.Name}}_remap.{{Kernel.IntersectionHierarhcy.Name}}_table[rayQueryGetIntersectionInstanceCustomIndexEXT(rayQuery, false)];
     
      CRT_LeafInfo info;
      info.aabbId = res.primId;  
      info.primId = info.aabbId/remap.y;
      info.instId = rayQueryGetIntersectionInstanceIdEXT(rayQuery, false); 
      info.geomId = rayQueryGetIntersectionInstanceCustomIndexEXT(rayQuery, false);
      info.rayxId = gl_GlobalInvocationID[0];
      info.rayyId = gl_GlobalInvocationID[1]; 
      
      const uint tag = {{Kernel.IntersectionHierarhcy.ObjBufferName}}[res.primId].x; // or info.primId ?
      uint intersected = {{Kernel.IntersectionHierarhcy.EmptyImplementation.TagName}};
      switch(tag) 
      {
        {% for Impl in Kernel.IntersectionHierarhcy.Implementations %}
        case {{Impl.TagName}}: 
        intersected = {{Impl.ClassName}}_Intersect_{{Impl.ObjBufferName}}(remap.x + info.primId, rayPosAndNear, rayDirAndFar, info, res);
        break;
        {% endfor %}
      };  
      //uint intersected = {{Kernel.IntersectionHierarhcy.Name}}_Intersect_{{Kernel.IntersectionHierarhcy.ObjBufferName}}(info.aabbId, rayPosAndNear, rayDirAndFar, info, res);
      if(intersected == {{Kernel.IntersectionHierarhcy.EmptyImplementation.TagName}}) 
      {
        res.primId = -1;
        res.instId = -1;
        res.geomId = -1;
        res.t      = rayDir.w;
      } 
      else
        rayQueryConfirmIntersectionEXT(rayQuery);      
      }
    {% endif %}
  } // actually may omit 'while' when 'gl_RayFlagsOpaqueEXT' is used

  if(rayQueryGetIntersectionTypeEXT(rayQuery, true) == gl_RayQueryCommittedIntersectionTriangleEXT)
  {    
	  res.primId    = rayQueryGetIntersectionPrimitiveIndexEXT(rayQuery, true);
	  res.geomId    = rayQueryGetIntersectionInstanceCustomIndexEXT(rayQuery, true);
    res.instId    = rayQueryGetIntersectionInstanceIdEXT    (rayQuery, true);
	  res.t         = rayQueryGetIntersectionTEXT(rayQuery, true);
    vec2 bars     = rayQueryGetIntersectionBarycentricsEXT(rayQuery, true);
    
    res.coords[0] = bars.y;
    res.coords[1] = bars.x;
    res.coords[2] = 1.0f - bars.y - bars.x;
    res.coords[3] = 0.0f;
  }

  return res;
}

CRT_Hit {{RTName}}_RayQuery_NearestHitMotion(vec4 rayPos, vec4 rayDir, float t) { return {{RTName}}_RayQuery_NearestHit(rayPos, rayDir); }

bool {{RTName}}_RayQuery_AnyHit(vec4 rayPos, vec4 rayDir)
{
  rayQueryEXT rayQuery;
  rayQueryInitializeEXT(rayQuery, {{RTName}}, gl_RayFlagsTerminateOnFirstHitEXT, 0xff, rayPos.xyz, rayPos.w, rayDir.xyz, rayDir.w);
  rayQueryProceedEXT(rayQuery);
  return (rayQueryGetIntersectionTypeEXT(rayQuery, true) == gl_RayQueryCommittedIntersectionTriangleEXT);
}

bool {{RTName}}_RayQuery_AnyHitMotion(vec4 rayPos, vec4 rayDir, float t) { return {{RTName}}_RayQuery_AnyHit(rayPos, rayDir); }

{% endif %}
{% endfor %}