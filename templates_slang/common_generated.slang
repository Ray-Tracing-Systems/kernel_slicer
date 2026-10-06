#include "common{{MainClassSuffixLowerCase}}_slang.h"
{% for KSpec in Kernel.SpecConstants %}
layout (constant_id = {{KSpec.Id}}) const int {{KSpec.Name}} = {{KSpec.Id}}; // 
{% endfor %}

{% for Arg in Kernel.Args %}
{% if not Arg.IsUBO %}
{% if Arg.IsImage %}
{% if Kernel.UseCombinedImageSampler %}[[vk::binding({{loop.index}}, 0)]] {% endif %}Texture2D<float4> {{Arg.Name}}; //
{% if Arg.IsImageArray %}
[[vk::binding({{loop.index}}, 0)]] SamplerState {{Arg.NameSam}};
{% endif %}
{% else if Arg.IsAccelStruct %}
{% if Kernel.UseCombinedImageSampler %}[[vk::binding({{loop.index}}, 0)]] {% endif %}RaytracingAccelerationStructure {{Arg.Name}};
{% else %}
{% if Kernel.UseCombinedImageSampler %}[[vk::binding({{loop.index}}, 0)]] {% endif %}{%if not Arg.IsConst%}RW{% endif %}StructuredBuffer<{{Arg.Type}}> {{Arg.Name}}; // 
{% endif %} {# /* Arg.IsImage */ #}
{% endif %} {# /* not Arg.IsUBO */ #}
{% endfor %}
{% if Kernel.UseCombinedImageSampler %}[[vk::binding({{length(Kernel.Args)}}, 0)]] {% endif %}{%if not Kernel.ContantUBO%}RW{% endif %}StructuredBuffer<{{MainClassName}}{{MainClassSuffix}}_UBO_Data> ubo;
{% for Map in Kernel.HashMaps %}

// std::unordered_map '{{Map.Name}}': returns slot index of 'a_key' or capacity (index of the last, garbage slot) if 'a_key' is absent
//
uint {{Map.FindFunc}}({{Map.KeyType}} a_key)
{
  const uint capacity = {{Map.Capacity}};
  uint slot = kslicer_hmap_hash(asuint(a_key)) & (capacity - 1);
  for(uint probe = 0; probe < capacity; probe++)
  {
    const {{Map.KeyType}} key = {{Map.Name}}[slot].key;
    if(key == a_key)
      return slot;
    if(key == {{Map.Sentinel}})
      break;
    slot = (slot + 1) & (capacity - 1);
  }
  return capacity;
}

// std::unordered_map '{{Map.Name}}': inserts 'a_key' if it is absent and returns its slot index;
// returns capacity (index of the last, garbage slot) if table is full
//
uint {{Map.InsertFunc}}({{Map.KeyType}} a_key)
{
  const uint capacity = {{Map.Capacity}};
  uint slot = kslicer_hmap_hash(asuint(a_key)) & (capacity - 1);
  for(uint probe = 0; probe < capacity; probe++)
  {
    {{Map.KeyType}} prev = {{Map.Name}}[slot].key;
    if(prev == {{Map.Sentinel}})
    {
      InterlockedCompareExchange({{Map.Name}}[slot].key, {{Map.Sentinel}}, a_key, prev);
      if(prev == {{Map.Sentinel}})
      {
        {% if not Kernel.ContantUBO %}
        InterlockedAdd({{Map.Size}}, 1u);
        {% endif %}
        return slot;
      }
    }
    if(prev == a_key)
      return slot;
    slot = (slot + 1) & (capacity - 1);
  }
  {% if not Kernel.ContantUBO %}
  InterlockedAdd({{Map.Overflow}}, 1u); // table is full, use reserve(...) on CPU to increase it
  {% endif %}
  return capacity;
}
{% for Func in HashMapAddFuncs %}
{% if Func.MapName == Map.Name %}

// '{{Map.Name}}[a_key]{{Func.Path}} += a_value' with aggregation inside subgroup: one insertion and one atomic per different key
//
void {{Func.Name}}({{Map.KeyType}} a_key, {{Func.ValueType}} a_value)
{
  // (1) group lanes by key for several iterations, only subgroup operations here
  //
  bool grouped = false, isLeader = false;
  {{Func.ValueType}} sum = {{Func.ValueType}}(0);
  for(uint iter = 0; iter < 4; iter++)
  {
    const {{Map.KeyType}} leader = WaveReadLaneFirst(a_key);
    const bool match = (a_key == leader);
    const {{Func.ValueType}} groupSum = WaveActiveSum(match ? a_value : {{Func.ValueType}}(0));
    if(match)
    {
      grouped  = true;
      isLeader = WaveIsFirstLane();
      sum      = groupSum;
      break;
    }
  }
  
  // (2) few different keys in subgroup: leaders insert and add in parallel
  //
  if(grouped)
  {
    if(isLeader)
      {{Func.AtomicFunc}}({{Map.Name}}[{{Map.InsertFunc}}(a_key)]{{Func.Path}}, sum);
    return;
  }

  // (3) many different keys: all lanes insert in parallel, then group by slot
  //
  const uint slot = {{Map.InsertFunc}}(a_key);
  for(;;)
  {
    const uint leader = WaveReadLaneFirst(slot);
    const bool match  = (slot == leader);
    const {{Func.ValueType}} groupSum = WaveActiveSum(match ? a_value : {{Func.ValueType}}(0));
    if(match)
    {
      if(WaveIsFirstLane())
        {{Func.AtomicFunc}}({{Map.Name}}[slot]{{Func.Path}}, groupSum);
      break;
    }
  }
}
{% endif %}
{% endfor %}
{% endfor %}

{% for Array in Kernel.ThreadLocalArrays %}
static {{Array.Type}} {{Array.Name}}[{{Array.Size}}];
{% endfor %}
{% for Array in ThreadLocalArrays %}
static {{Array.Type}} {{Array.Name}}[{{Array.Size}}];
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
  rayQuery.TraceRayInline({{RTName}}, RAY_FLAG_NONE, 0xff, ray);
  
  CRT_Hit res;
  res.primId = -1;
  res.instId = -1;
  res.geomId = -1;
  res.t      = rayDir.w;
  
  while(rayQuery.Proceed())
  {
    {% if Kernel.HasIntersectionShader2 %}
    if (rayQuery.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
    {
      // TODO: add opacity check here, use rayQuery.Abort() if didn't find the hit;
      rayQuery.CommitNonOpaqueTriangleHit();
    }
    else if (rayQuery.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE)
    {
      float4 rayPosAndNear = float4(rayQuery.CandidateObjectRayOrigin(), rayPos.w);
      float4 rayDirAndFar  = float4(rayQuery.CandidateObjectRayDirection(), rayDir.w);
      const uint geomIndex = rayQuery.CandidateInstanceID(); // user defined geometry per-inatsnce index
      uint2 remap = {{ RTName }}_remap[geomIndex];
      CRT_LeafInfo info;
      info.aabbId = rayQuery.CandidatePrimitiveIndex();
      info.primId = remap.x + info.aabbId / remap.y;
      info.instId = rayQuery.CandidateInstanceIndex();
      info.geomId = geomIndex;
      info.rayxId = 0; // TODO: gl_GlobalInvocationID[0];
      info.rayyId = 0; // TODO: gl_GlobalInvocationID[1];
      uint intersected = {{ Kernel.IS2_AccObjName }}_{{ Kernel.IS2_ShaderName}}(rayPosAndNear, rayDirAndFar, info, res);
      if (intersected != 0)
        rayQuery.CommitProceduralPrimitiveHit(res.t);
    }
    {% endif %}
  } // may omit 'while' loop when 'RAY_FLAG_FORCE_OPAQUE' is used

  if(rayQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
  {
    res.primId = rayQuery.CommittedPrimitiveIndex(); // auto generated index
    res.geomId = rayQuery.CommittedInstanceID();     // user defined geometry per-inatsnce index
    res.instId = rayQuery.CommittedInstanceIndex();  // auto generated index
    res.t      = rayQuery.CommittedRayT();

    float2 bars = rayQuery.CommittedTriangleBarycentrics();
  
    res.coords[0] = bars.y;
    res.coords[1] = bars.x;
    res.coords[2] = 1.0f - bars.y - bars.x;
    res.coords[3] = 0.0f;
  }
  {% if Kernel.IS2_HasFinish %}
  else
  {
    {{ Kernel.IS2_AccObjName }}_{{ Kernel.IS2_FinishName}}(rayPos, rayDir, res);
  }
  {% endif %}

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
  rayQuery.TraceRayInline({{RTName}}, RAY_FLAG_NONE, 0xff, ray);
  rayQuery.Proceed();

  return (rayQuery.CommittedStatus() == COMMITTED_TRIANGLE_HIT);
}

CRT_Hit {{RTName}}_RayQuery_NearestHitMotion(float4 rayPos, float4 rayDir, float t) { return {{RTName}}_RayQuery_NearestHit(rayPos, rayDir); }
bool {{RTName}}_RayQuery_AnyHitMotion(float4 rayPos, float4 rayDir, float t) { return {{RTName}}_RayQuery_AnyHit(rayPos, rayDir); }

{% endfor %}
{% if UsePersistentThreads %}

static uint g_persistentIter;
static uint g_persistentTotalSize;
uint RTVPersistent_ThreadId(uint a_tid) { return (a_tid + g_persistentIter * g_persistentTotalSize) / WaveGetLaneCount(); }
void RTVPersistent_SetIter(uint a_pid)  { g_persistentIter = a_pid; }
uint RTVPersistent_Iters()   { return WaveGetLaneCount(); }
bool RTVPersistent_IsFirst() { return (WaveGetLaneIndex() == 0); }
float4 RTVPersistent_ReduceAdd4f(float4 color) { return WaveActiveSum(color); }
float  RTVPersistent_ReduceAdd1f(float  color) { return WaveActiveSum(color); }
{% endif %}

{% for MembFunc in Kernel.MemberFunctions %}
{% if not (MembFunc.IsRayQuery and (Kernel.UseRayGen or Kernel.HasIntersectionShader2 or (length(Kernel.IntersectionHierarhcy.Implementations) >= 1)) ) %}

{{MembFunc.Text}}
{% endif%}
{% endfor %}