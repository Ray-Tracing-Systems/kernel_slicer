#version 460
#extension GL_GOOGLE_include_directive : require
{% if length(Kernel.RTXNames) > 0 %}
{% if Kernel.UseRayGen %}
#extension GL_EXT_ray_tracing : require 
{% else %}
#extension GL_EXT_ray_query : require
{% endif %}
{% endif %}
{% if Kernel.NeedTexArray %}
#extension GL_EXT_nonuniform_qualifier : require
{% endif %}

#include "common{{MainClassSuffixLowerCase}}.h"
{% for KSpec in Kernel.SpecConstants %}
layout (constant_id = {{KSpec.Id}}) const int {{KSpec.Name}} = {{KSpec.Id}};
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

## for ShitFunc in Kernel.ShityFunctions  
{{ShitFunc}}

## endfor
## for MembFunc in Kernel.MemberFunctions  
{{MembFunc}}

## endfor
{% for RTName in Kernel.RTXNames %}
// RayScene intersection with '{{RTName}}'
//
{% if Kernel.UseRayGen %}
layout(location = 0) rayPayloadEXT CRT_Hit kgen_hitValue;
layout(location = 1) rayPayloadEXT bool    kgen_inShadow;

CRT_Hit {{RTName}}_RayQuery_NearestHit(const vec4 rayPos, const vec4 rayDir)
{
  traceRayEXT(m_pAccelStruct, gl_RayFlagsOpaqueEXT, 0xff, 0, 0, 0, rayPos.xyz, rayPos.w, rayDir.xyz, rayDir.w, 0);
  return kgen_hitValue;
}

bool {{RTName}}_RayQuery_AnyHit(const vec4 rayPos, const vec4 rayDir)
{
  kgen_inShadow = true;
  traceRayEXT(m_pAccelStruct, gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT,
              0xff, 0, 0, 1, rayPos.xyz, rayPos.w, rayDir.xyz, rayDir.w, 1);
  return kgen_inShadow;
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

bool {{RTName}}_RayQuery_AnyHit(const vec4 rayPos, const vec4 rayDir)
{
  rayQueryEXT rayQuery;
  rayQueryInitializeEXT(rayQuery, {{RTName}}, gl_RayFlagsTerminateOnFirstHitEXT, 0xff, rayPos.xyz, rayPos.w, rayDir.xyz, rayDir.w);
  rayQueryProceedEXT(rayQuery);
  return (rayQueryGetIntersectionTypeEXT(rayQuery, true) == gl_RayQueryCommittedIntersectionTriangleEXT);
}
{% endif %}
{% endfor %}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
{% if not Kernel.UseRayGen %}
layout(local_size_x = {{Kernel.WGSizeX}}, local_size_y = {{Kernel.WGSizeY}}, local_size_z = {{Kernel.WGSizeZ}}) in;
{% endif %}
layout( push_constant ) uniform kernelArgs
{
  {% for UserArg in Kernel.UserArgs %} 
  {{UserArg.Type}} {{UserArg.Name}};
  {% endfor %}
  {{Kernel.threadSZType1}} {{Kernel.threadSZName1}}; 
  {{Kernel.threadSZType2}} {{Kernel.threadSZName2}}; 
  {{Kernel.threadSZType3}} {{Kernel.threadSZName3}}; 
  uint tFlagsMask;    
} kgenArgs;

///////////////////////////////////////////////////////////////// subkernels here
{% for sub in Kernel.Subkernels %}
{{sub.Decl}}
{
  {{sub.Source}}
}

{% endfor %}
///////////////////////////////////////////////////////////////// subkernels here

void main()
{
  ///////////////////////////////////////////////////////////////// prolog
  {% for TID in Kernel.ThreadIds %}
  const {{TID.Type}} {{TID.Name}} = {{TID.Type}}({% if Kernel.UseRayGen %}gl_LaunchIDEXT{% else %}gl_GlobalInvocationID{% endif %}[{{ loop.index }}]); 
  {% endfor %}
  ///////////////////////////////////////////////////////////////// prolog

  {# /*------------------------------------------------------------- KERNEL SOURCE ------------------------------------------------------------- */ #}
  {{Kernel.Source}}
  {# /*------------------------------------------------------------- KERNEL SOURCE ------------------------------------------------------------- */ #}
}
