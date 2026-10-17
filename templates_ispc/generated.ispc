/////////////////////////////////////////////////////////////////////
/////////////////// include files ///////////////////////////////////
/////////////////////////////////////////////////////////////////////
#include "LiteMathISPC.h"
{% if UseAccelStruct %}
#ifdef USE_EMBREE4
#include "embree4/rtcore.isph"
#else
#include "embree3/rtcore.isph"
#endif
{% endif %}

## for Incl in Includes  
#include "{{Incl}}"
## endfor

/////////////////////////////////////////////////////////////////////
/////////////////// declarations in class ///////////////////////////
/////////////////////////////////////////////////////////////////////
## for Decl in ClassDecls  
{{Decl.Text}}
## endfor

struct {{MainClassName}}{{MainClassSuffix}}_UBO_Data
{
  {% for Field in UBO.UBOStructFields %}
  {% if Field.IsDummy %} 
  uint {{Field.Name}}; 
  {% else %}
  {{Field.Type}} {{Field.Name}}{% if Field.IsArray %}[{{Field.ArraySize}}]{% endif %}; 
  {% endif %}
  {% endfor %}
  uint dummy_last;
};

/////////////////////////////////////////////////////////////////////
/////////////////// local functions /////////////////////////////////
/////////////////////////////////////////////////////////////////////

## for LocalFunc in LocalFunctions  
{{LocalFunc}}

## endfor

/////////////////////////////////////////////////////////////////////
/////////////////// class data for member functions /////////////////
/////////////////////////////////////////////////////////////////////

// member functions access class data directly, so it is placed to global variables which are set by '<Kernel>_Bind_ISPC' 
// before kernel call; thus, kernels of different objects of '{{MainClassName}}' should not be executed concurrently 
//
static uniform struct {{MainClassName}}{{MainClassSuffix}}_UBO_Data * uniform ubo;
{% for Buff in MemberBuffersISPC %}
static uniform {{Buff.Type}} * uniform {{Buff.Name}};
{% endfor %}
{% for AccName in AccelStructsISPC %}
static RTCScene {{AccName}}_scene;
static uniform uint * uniform {{AccName}}_geomIdByInst; // no "const" here: ISPC makes the global variable itself constant

// '{{AccName}}->RayQuery_NearestHit(pos, dir)' ==> '{{AccName}}_RayQuery_NearestHit(pos, dir)': 
// rtcIntersectV traces all active program instances as a packet of programCount rays 
//
static inline CRT_Hit {{AccName}}_RayQuery_NearestHit(float4 posAndNear, float4 dirAndFar)
{
  RTCRayHit rayhit;
  rayhit.ray.org_x = posAndNear.x;
  rayhit.ray.org_y = posAndNear.y;
  rayhit.ray.org_z = posAndNear.z;
  rayhit.ray.tnear = posAndNear.w;
  rayhit.ray.dir_x = dirAndFar.x;
  rayhit.ray.dir_y = dirAndFar.y;
  rayhit.ray.dir_z = dirAndFar.z;
  rayhit.ray.tfar  = dirAndFar.w;
  rayhit.ray.time  = 0.0f;
  rayhit.ray.mask  = -1;
  rayhit.ray.flags = 0;
  rayhit.hit.geomID    = RTC_INVALID_GEOMETRY_ID;
  rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

  #ifdef USE_EMBREE4
  rtcIntersectV({{AccName}}_scene, &rayhit);
  #else
  uniform RTCIntersectContext context;
  rtcInitIntersectContext(&context);
  rtcIntersectV({{AccName}}_scene, &context, &rayhit);
  #endif

  CRT_Hit result;
  result.t = rayhit.ray.tfar;
  if(rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID)
  {
    result.geomId    = {{AccName}}_geomIdByInst[rayhit.hit.instID[0]];
    result.instId    = rayhit.hit.instID[0];
    result.primId    = rayhit.hit.primID;
    result.coords[1] = rayhit.hit.u;
    result.coords[0] = rayhit.hit.v;
    result.coords[2] = 1.0f - rayhit.hit.v - rayhit.hit.u;
    result.coords[3] = 0.0f;
  }
  else
  {
    result.geomId = (uint)(-1);
    result.instId = (uint)(-1);
    result.primId = (uint)(-1);
  }
  return result;
}

static inline bool {{AccName}}_RayQuery_AnyHit(float4 posAndNear, float4 dirAndFar)
{
  RTCRay ray;
  ray.org_x = posAndNear.x;
  ray.org_y = posAndNear.y;
  ray.org_z = posAndNear.z;
  ray.tnear = posAndNear.w;
  ray.dir_x = dirAndFar.x;
  ray.dir_y = dirAndFar.y;
  ray.dir_z = dirAndFar.z;
  ray.tfar  = dirAndFar.w;
  ray.time  = 0.0f;
  ray.mask  = -1;
  ray.flags = 0;

  #ifdef USE_EMBREE4
  rtcOccludedV({{AccName}}_scene, &ray);
  #else
  uniform RTCIntersectContext context;
  rtcInitIntersectContext(&context);
  rtcOccludedV({{AccName}}_scene, &context, &ray);
  #endif

  return (ray.tfar < 0.0f);
}

static inline CRT_Hit {{AccName}}_RayQuery_NearestHitMotion(float4 posAndNear, float4 dirAndFar, float time) { return {{AccName}}_RayQuery_NearestHit(posAndNear, dirAndFar); }
static inline bool    {{AccName}}_RayQuery_AnyHitMotion(float4 posAndNear, float4 dirAndFar, float time)     { return {{AccName}}_RayQuery_AnyHit(posAndNear, dirAndFar); }

{% endfor %}
/////////////////////////////////////////////////////////////////////
/////////////////// member functions ////////////////////////////////
/////////////////////////////////////////////////////////////////////

{% for MembFunc in AllMemberFunctions %}
{{MembFunc.Decl}};
{% endfor %}

{% for MembFunc in AllMemberFunctions %}
{{MembFunc.Text}}

{% endfor %}
/////////////////////////////////////////////////////////////////////
/////////////////// kernels /////////////////////////////////////////
/////////////////////////////////////////////////////////////////////

## for Kernel in Kernels
export void {{Kernel.Name}}_Bind_ISPC( // sets class data for member functions, call it before kernel
{% include "inc_bind_args.ispc" %}
  uniform struct {{MainClassName}}{{MainClassSuffix}}_UBO_Data a_ubo[1])
{
  ubo = a_ubo;
  {% for Arg in Kernel.Args %}
  {% if Arg.IsMember and Arg.IsAccelStruct %}
  {{Arg.Name}}_scene = (RTCScene)a_{{Arg.Name}}_scene;
  {{Arg.Name}}_geomIdByInst = (uniform uint * uniform)a_{{Arg.Name}}_geomIdByInst;
  {% else if Arg.IsMember and Arg.IsPointer %}
  {{Arg.Name}} = a_{{Arg.Name}};
  {% endif %}
  {% endfor %}
}

{% if Kernel.SingleThreadISPC %}
export void {{Kernel.Name}}_ISPC(
{% include "inc_args.ispc" %}
  uniform struct {{MainClassName}}{{MainClassSuffix}}_UBO_Data ubo[1])
{
  {# /* declare varyings which are subjected to reduction, they are initialized with neutral element of reduction */ #}
  {% for Var in Kernel.SubjToRed %}
  {{Var.Type}} {{Var.Name}} = {{Var.Init}};
  {% endfor %}
  {{Kernel.InitSource}}
  {% if Kernel.ExplicitIdISPC %}
  {% if Kernel.threadDim == 1 %}
  for (uniform {{Kernel.ThreadId0.Type}} {{Kernel.ThreadId0.Name}} = {{Kernel.ThreadId0.Start}}; {{Kernel.ThreadId0.Name}} < {{Kernel.ThreadId0.Size}}; {{Kernel.ThreadId0.Name}} += programCount) {
  {% else %}
  for (uniform {{Kernel.ThreadId1.Type}} {{Kernel.ThreadId1.Name}} = {{Kernel.ThreadId1.Start}}; {{Kernel.ThreadId1.Name}} < {{Kernel.ThreadId1.Size}}; {{Kernel.ThreadId1.Name}} ++) {
  for (uniform {{Kernel.ThreadId0.Type}} {{Kernel.ThreadId0.Name}} = {{Kernel.ThreadId0.Start}}; {{Kernel.ThreadId0.Name}} < {{Kernel.ThreadId0.Size}}; {{Kernel.ThreadId0.Name}} += programCount) {
  {% endif %}
  {% else %}
  foreach ({% for TID in Kernel.ThreadIds %}{{TID.Name}} = {{TID.Start}} ... {{TID.Size}}{% if loop.index1 != Kernel.threadDim %}, {% endif %}{% endfor %}) { 
  {% endif %}
  {{Kernel.Source}}
  {% if Kernel.ExplicitIdISPC and Kernel.threadDim == 2 %}
  }}
  {% else %}
  }
  {% endif %}
  {% for Var in Kernel.SubjToRed %}
  {% if Var.Op2 == "add" %}
  ubo[0].{{Var.Name}} += {{Var.Op}}({{Var.Name}});
  {% else %}
  ubo[0].{{Var.Name}} = {{Var.Op2}}(ubo[0].{{Var.Name}}, {{Var.Op}}({{Var.Name}}));
  {% endif %}
  {% endfor %}
  {{Kernel.FinishSource}}
}
{% else %} {% if Kernel.OpenMPAndISPC %}
{% if Kernel.HasInitISPC %}
export void {{Kernel.Name}}_Init_ISPC( // executed once before parallel loop
{% include "inc_args.ispc" %}
  uniform struct {{MainClassName}}{{MainClassSuffix}}_UBO_Data ubo[1])
{
  {{Kernel.InitSource}}
}

{% endif %}
export void {{Kernel.Name}}_ISPC(
{% include "inc_args.ispc" %}
  uniform struct {{MainClassName}}{{MainClassSuffix}}_UBO_Data ubo[1],
  uniform int _start, uniform int _end, uniform int _ycoord)
{
  {# /* declare varyings which are subjected to reduction, they are initialized with neutral element of reduction */ #}
  {% for Var in Kernel.SubjToRed %}
  {{Var.Type}} {{Var.Name}} = {{Var.Init}};
  {% endfor %}
  {% if Kernel.threadDim == 1 %}
  foreach({{Kernel.ThreadId0.Name}} = _start ... _end) {
  {% else %}
  const uniform int {{Kernel.ThreadId0.Name}} = _ycoord;
  foreach({{Kernel.ThreadId1.Name}} = _start ... _end) {
  {% endif %}
  {{Kernel.Source}}
  }
  {% for Var in Kernel.SubjToRed %}
  atomic_{{Var.Op2}}_global(&ubo[0].{{Var.Name}}, {{Var.Op}}({{Var.Name}}));
  {% endfor %}
}
{% if Kernel.HasFinishISPC %}

export void {{Kernel.Name}}_Finish_ISPC( // executed once after parallel loop
{% include "inc_args.ispc" %}
  uniform struct {{MainClassName}}{{MainClassSuffix}}_UBO_Data ubo[1])
{
  {{Kernel.FinishSource}}
}
{% endif %}
{% else %}
task void {{Kernel.Name}}_Task(
{% include "inc_args.ispc" %}
  uniform struct {{MainClassName}}{{MainClassSuffix}}_UBO_Data ubo[1],
  uniform int _start, uniform int _end, uniform int _span)
{
  {# /* declare varyings which are subjected to reduction, they are initialized with neutral element of reduction */ #}
  {% for Var in Kernel.SubjToRed %}
  {{Var.Type}} {{Var.Name}} = {{Var.Init}};
  {% endfor %}
  {% if Kernel.threadDim == 1 %}
  const uniform int xstart = _start + taskIndex * _span;
  const uniform int xend   = _start + min((uniform int)((taskIndex+1) * _span), _end);
  foreach({{Kernel.ThreadId0.Name}} = xstart ... xend) {
  {% else %}
  const uniform int ystart = _start + taskIndex * _span;
  const uniform int yend   = _start + min((uniform int)((taskIndex+1) * _span), _end);
  foreach({{Kernel.ThreadId0.Name}} = ystart ... yend, {{Kernel.ThreadId1.Name}} = {{Kernel.ThreadId1.Start}} ... {{Kernel.ThreadId1.Size}}) {
  {% endif %}
  {{Kernel.Source}}
  }
  {% for Var in Kernel.SubjToRed %}
  atomic_{{Var.Op2}}_global(&ubo[0].{{Var.Name}}, {{Var.Op}}({{Var.Name}}));
  {% endfor %}
}

export void {{Kernel.Name}}_ISPC(
{% include "inc_args.ispc" %}
  uniform struct {{MainClassName}}{{MainClassSuffix}}_UBO_Data ubo[1])
{ 
  {{Kernel.InitSource}}
  {% for TID in Kernel.ThreadIds %}
  const uniform int l_start{{loop.index}} = {{TID.Start}};
  const uniform int l_size{{loop.index}}  = {{TID.Size}};
  {% endfor %}
  {% if Kernel.threadDim == 1 %}
  const uniform int span   = 1024;
  {% else %}
  const uniform int span   = 4;
  {% endif %}
  const uniform int length = l_size0-l_start0;
  launch[(length + span - 1)/span] {{Kernel.Name}}_Task({% include "inc_call_args.ispc" %}ubo,l_start0,l_size0,span);
  sync;
  {{Kernel.FinishSource}}
}
{% endif %}
{% endif %}

## endfor
