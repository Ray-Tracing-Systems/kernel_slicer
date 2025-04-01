#include "common{{MainClassSuffixLowerCase}}_slang.h"
{% for KSpec in Kernel.SpecConstants %}
layout (constant_id = {{KSpec.Id}}) const int {{KSpec.Name}} = {{KSpec.Id}}; // 
{% endfor %}

{% for Arg in Kernel.Args %}
{% if not Arg.IsUBO %} 
{% if Arg.IsImage %}
RWTexture2D<Arg.Type> {{Arg.Name}}; // TODO: this is Ot a combined image sampler (!!!)
{% else if Arg.IsAccelStruct %}
RaytracingAccelerationStructure {{Arg.Name}};
{% else %}
RWStructuredBuffer<{{Arg.Type}}> {{Arg.Name}}; // 
{% endif %} {# /* Arg.IsImage */ #}
{% endif %} {# /* not Arg.IsUBO */ #}
{% endfor %}
RWStructuredBuffer<{{MainClassName}}{{MainClassSuffix}}_UBO_Data> ubo;

{% for Array in Kernel.ThreadLocalArrays %}
{{Array.Type}} {{Array.Name}}[{{Array.Size}}];
{% endfor %}
{% for MembFunc in Kernel.MemberFunctions %}
{{MembFunc.Decl}};

{% endfor %}
{% for ShitFunc in Kernel.ShityFunctions %} 
{{ShitFunc}}

{% endfor %}
{% for RTName in Kernel.RTXNames %}
// RayScene intersection with '{{RTName}}'
//
CRT_Hit {{RTName}}_RayQuery_NearestHit(float4 rayPos, float4 rayDir)
{
  RayDesc ray;
  ray.Origin    = rayPos.xyz;
  ray.Direction = rayDir.xyz;
  ray.TMin      = rayPos.w;
  ray.TMax      = rayDir.w;

  RayQuery<RAY_FLAG_FORCE_OPAQUE> rayQuery;
  rayQuery.TraceRayInline(m_pAccelStruct, RAY_FLAG_NONE, 0xff, ray);
  
  CRT_Hit res;
  res.primId = -1;
  res.instId = -1;
  res.geomId = -1;
  res.t      = rayDir.w;
  
  while(rayQuery.Proceed()) 
  {

  } // may omit 'while' loop when 'RAY_FLAG_FORCE_OPAQUE' is used

  if(rayQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
  {
    res.primId = rayQuery.CommittedPrimitiveIndex();
    res.geomId = rayQuery.CommittedGeometryIndex();
    res.instId = rayQuery.CommittedInstanceID();
    res.t      = rayQuery.CommittedRayT();

    float2 bars = rayQuery.CommittedTriangleBarycentrics();
  
    res.coords[0] = bars.y;
    res.coords[1] = bars.x;
    res.coords[2] = 1.0f - bars.y - bars.x;
    res.coords[3] = 0.0f;
  }

  return res;
}

bool {{RTName}}_RayQuery_AnyHit(float4 rayPos, float4 rayDir)
{
  RayDesc ray;
  ray.Origin    = rayPos.xyz;
  ray.Direction = rayDir.xyz;
  ray.TMin      = rayPos.w;
  ray.TMax      = rayDir.w;

  RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> rayQuery;
  rayQuery.TraceRayInline(m_pAccelStruct, RAY_FLAG_NONE, 0xff, ray);
  rayQuery.Proceed();

  return (rayQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT);
}

CRT_Hit {{RTName}}_RayQuery_NearestHitMotion(float4 rayPos, float4 rayDir, float t) { return {{RTName}}_RayQuery_NearestHit(rayPos, rayDir); }
bool {{RTName}}_RayQuery_AnyHitMotion(float4 rayPos, float4 rayDir, float t) { return {{RTName}}_RayQuery_AnyHit(rayPos, rayDir); }

{% endfor %}
{% for MembFunc in Kernel.MemberFunctions %}
{% if not (MembFunc.IsRayQuery and (Kernel.UseRayGen or (length(Kernel.IntersectionHierarhcy.Implementations) >= 1)) ) %}

{{MembFunc.Text}}
{% endif%}
{% endfor %}