#include "common{{MainClassSuffixLowerCase}}.h"
{% for KSpec in Kernel.SpecConstants %}
layout (constant_id = {{KSpec.Id}}) const int {{KSpec.Name}} = {{KSpec.Id}};
{% endfor %}
{% for Hierarchy in Kernel.Hierarchies %} 

struct {{Hierarchy.Name}}
{
  uint vptr_dummy[2];
  {% for Field in Hierarchy.InterfaceFields %}
  {{Field.Type}} {{Field.Name}};
  {% endfor %}
};

{% endfor %}
## for Arg in Kernel.Args
{% if not Arg.IsUBO %} 
{% if Arg.IsImage %}
layout(binding = {{loop.index}}, set = 0{% if Arg.NeedFmt%}, {{Arg.ImFormat}}{% endif %}) uniform {{Arg.Type}} {{Arg.Name}}; //
{% else if Arg.IsAccelStruct %}
layout(binding = {{loop.index}}, set = 0) uniform accelerationStructureEXT {{Arg.Name}};
{% else %}
layout(binding = {{loop.index}}, set = 0) buffer data{{loop.index}} { {{Arg.Type}} {{Arg.Name}}[]; }; //
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
//Impl.ClassName: Empty Imlementation
//Impl.ObjBuffer: {{Hierarchy.EmptyImplementation.ObjBufferName}}
//
{% for Member in Hierarchy.EmptyImplementation.MemberFunctions %}
{{Member}}

{% endfor %}

{% for Impl in Hierarchy.Implementations %}
//Impl.ClassName: {{Impl.ClassName}}
//Impl.ObjBuffer: {{Impl.ObjBufferName}}
//
{% for Member in Impl.MemberFunctions %}
{{Member}}

{% endfor %}

{% for Field in Impl.Fields %}
//{{Field}}
{% endfor %}
{% endfor %}
{% for VirtualFunc in Hierarchy.VirtualFunctions %}
{{VirtualFunc.Decl}} 
{
  const uint tag = {{Hierarchy.ObjBufferName}}[selfId].m_tag;
  switch(tag) 
  {
    {% for Impl in Hierarchy.Implementations %}
    case {{Impl.TagName}}: return {{Impl.ClassName}}_{{VirtualFunc.Name}}_{{Impl.ObjBufferName}}({% for Arg in VirtualFunc.Args %}{{Arg.Name}}{% if loop.index != VirtualFunc.ArgLen %},{% endif %}{% endfor %});
    {% endfor %}
    default: return {{Hierarchy.EmptyImplementation.ClassName}}_{{VirtualFunc.Name}}_{{Hierarchy.EmptyImplementation.ObjBufferName}}({% for Arg in VirtualFunc.Args %}{{Arg.Name}}{% if loop.index != VirtualFunc.ArgLen %},{% endif %}{% endfor %});
  };
}
{% endfor %}
{% endfor %}                        {# /*------------------------------ vfh ------------------------------ */ #}
{% for MembFunc in Kernel.MemberFunctions %}
{{MembFunc.Text}}

{% endfor %} 
{% for RTName in Kernel.RTXNames %}
// RayScene intersection with '{{RTName}}'
//
{% if Kernel.UseRayGen %}
layout(location = 0) rayPayloadEXT CRT_Hit {{RTName}}_hitValue;
layout(location = 1) rayPayloadEXT bool    {{RTName}}_inShadow;

CRT_Hit {{RTName}}_RayQuery_NearestHit(const vec4 rayPos, const vec4 rayDir)
{
  traceRayEXT(m_pAccelStruct, gl_RayFlagsOpaqueEXT, 0xff, 0, 0, 0, rayPos.xyz, rayPos.w, rayDir.xyz, rayDir.w, 0);
  return {{RTName}}_hitValue;
}

CRT_Hit {{RTName}}_RayQuery_NearestHitMotion(const vec4 rayPos, const vec4 rayDir, float t)
{
  {% if UseMotionBlur %}
  traceRayMotionNV(m_pAccelStruct, gl_RayFlagsOpaqueEXT, 0xff, 0, 0, 0, rayPos.xyz, rayPos.w, rayDir.xyz, rayDir.w, t, 0);
  {% else %}
  traceRayEXT(m_pAccelStruct, gl_RayFlagsOpaqueEXT, 0xff, 0, 0, 0, rayPos.xyz, rayPos.w, rayDir.xyz, rayDir.w, 0);
  {% endif %} 
  return {{RTName}}_hitValue;
}

bool {{RTName}}_RayQuery_AnyHit(const vec4 rayPos, const vec4 rayDir)
{
  {{RTName}}_inShadow = true;
  traceRayEXT(m_pAccelStruct, gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT,
              0xff, 0, 0, 1, rayPos.xyz, rayPos.w, rayDir.xyz, rayDir.w, 1);
  return {{RTName}}_inShadow;
}

bool {{RTName}}_RayQuery_AnyHitMotion(const vec4 rayPos, const vec4 rayDir, float t)
{
  {{RTName}}_inShadow = true;
  {% if UseMotionBlur %}
  traceRayMotionNV(m_pAccelStruct, gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT,
                   0xff, 0, 0, 1, rayPos.xyz, rayPos.w, rayDir.xyz, rayDir.w, t, 1);
  {% else %}
  traceRayEXT(m_pAccelStruct, gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT,
              0xff, 0, 0, 1, rayPos.xyz, rayPos.w, rayDir.xyz, rayDir.w, 1);
  {% endif %}
  return {{RTName}}_inShadow;
}

{% else %}
CRT_Hit {{RTName}}_RayQuery_NearestHit(const vec4 rayPos, const vec4 rayDir)
{
  rayQueryEXT rayQuery;
  rayQueryInitializeEXT(rayQuery, {{RTName}}, gl_RayFlagsOpaqueEXT, 0xff, rayPos.xyz, rayPos.w, rayDir.xyz, rayDir.w);
  
  while(rayQueryProceedEXT(rayQuery)) { } // actually may omit 'while' when 'gl_RayFlagsOpaqueEXT' is used
 
  CRT_Hit res;
  res.primId = -1;
  res.instId = -1;
  res.geomId = -1;
  res.t      = rayDir.w;

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

CRT_Hit {{RTName}}_RayQuery_NearestHitMotion(const vec4 rayPos, const vec4 rayDir, float t) { return {{RTName}}_RayQuery_NearestHit(rayPos, rayDir); }

bool {{RTName}}_RayQuery_AnyHit(const vec4 rayPos, const vec4 rayDir)
{
  rayQueryEXT rayQuery;
  rayQueryInitializeEXT(rayQuery, {{RTName}}, gl_RayFlagsTerminateOnFirstHitEXT, 0xff, rayPos.xyz, rayPos.w, rayDir.xyz, rayDir.w);
  rayQueryProceedEXT(rayQuery);
  return (rayQueryGetIntersectionTypeEXT(rayQuery, true) == gl_RayQueryCommittedIntersectionTriangleEXT);
}

bool {{RTName}}_RayQuery_AnyHitMotion(const vec4 rayPos, const vec4 rayDir, float t) { return {{RTName}}_RayQuery_AnyHit(rayPos, rayDir); }

{% endif %}
{% endfor %}