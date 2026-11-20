/////////////////////////////////////////////////////////////////////
/////////////////// include files ///////////////////////////////////
/////////////////////////////////////////////////////////////////////
#include "LiteMathISPC.h"
#include "LiteImageISPC.h"
{% if HasGraphics %}
#include "raster_ispc.h"
{% endif %}
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
{% if Buff.IsImage %}
static uniform LiteImageInfo {{Buff.Name}}_info;
{% endif %}
{% endfor %}
{% for Sampler in SamplersISPC %}
static uniform LiteSamplerInfo {{Sampler}};
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
  {% if Arg.IsMember and Arg.IsImage and existsIn(Arg, "ElemISPC") %}
  {{Arg.Name}} = a_{{Arg.Name}};
  {{Arg.Name}}_info.width  = a_{{Arg.Name}}_info[0];
  {{Arg.Name}}_info.height = a_{{Arg.Name}}_info[1];
  {{Arg.Name}}_info.pitch  = a_{{Arg.Name}}_info[2];
  {{Arg.Name}}_info.layout = a_{{Arg.Name}}_info[3];
  {% else if Arg.IsMember and Arg.IsAccelStruct %}
  {{Arg.Name}}_scene = (RTCScene)a_{{Arg.Name}}_scene;
  {{Arg.Name}}_geomIdByInst = (uniform uint * uniform)a_{{Arg.Name}}_geomIdByInst;
  {% else if Arg.IsMember and Arg.IsPointer %}
  {{Arg.Name}} = a_{{Arg.Name}};
  {% endif %}
  {% endfor %}
  {% for Sampler in Kernel.SamplersISPC %}
  {{Sampler}}.addressU = a_{{Sampler}}[0];
  {{Sampler}}.addressV = a_{{Sampler}}[1];
  {{Sampler}}.filter   = a_{{Sampler}}[2];
  {% endfor %}
}

{% if not Kernel.IsGraphics %}

{% if Kernel.EnableBlockExpansion %}
{% for TID in Kernel.ThreadSizeBE %}
#define {{TID.Name}} {{TID.Value}}u // template parameter of '{{Kernel.Name}}'
{% endfor %}

// block expansion: one iteration of the outer loop processes a block; code of a single thread is uniform,
// parallel loops of the block are 'foreach' over the gang, arrays of the block are uniform local arrays;
// blocks are numbered linearly: [a_blockStart, a_blockEnd) of 0 ... number of blocks
//
static void {{Kernel.Name}}_Blocks(
{% include "inc_args.ispc" %}
  uniform struct {{MainClassName}}{{MainClassSuffix}}_UBO_Data ubo[1], uniform int a_blockStart, uniform int a_blockEnd)
{
  for(uniform int kgen_block = a_blockStart; kgen_block < a_blockEnd; kgen_block++)
  {
    {% if Kernel.threadDim == 1 %}
    const uniform {{Kernel.ThreadId0.Type}} {{Kernel.ThreadId0.Name}} = {% if Kernel.ThreadId0.Start != "0" %}{{Kernel.ThreadId0.Start}} + {% endif %}kgen_block;
    {% else %}
    const uniform {{Kernel.ThreadId0.Type}} {{Kernel.ThreadId0.Name}} = {% if Kernel.ThreadId0.Start != "0" %}{{Kernel.ThreadId0.Start}} + {% endif %}kgen_block % {% if Kernel.ThreadId0.Start == "0" %}({{Kernel.ThreadId0.Size}}){% else %}({% if Kernel.ThreadId0.Start == "0" %}({{Kernel.ThreadId0.Size}}){% else %}(({{Kernel.ThreadId0.Size}}) - ({{Kernel.ThreadId0.Start}})){% endif %}){% endif %};
    const uniform {{Kernel.ThreadId1.Type}} {{Kernel.ThreadId1.Name}} = {% if Kernel.ThreadId1.Start != "0" %}{{Kernel.ThreadId1.Start}} + {% endif %}kgen_block / {% if Kernel.ThreadId0.Start == "0" %}({{Kernel.ThreadId0.Size}}){% else %}({% if Kernel.ThreadId0.Start == "0" %}({{Kernel.ThreadId0.Size}}){% else %}(({{Kernel.ThreadId0.Size}}) - ({{Kernel.ThreadId0.Start}})){% endif %}){% endif %};
    {% endif %}
    {% for Var in Kernel.SharedBE %}
    {{Var}}
    {% endfor %}
{{Kernel.SourceBE}}
  }
}

{% if Kernel.SingleThreadISPC %}
export void {{Kernel.Name}}_ISPC(
{% include "inc_args.ispc" %}
  uniform struct {{MainClassName}}{{MainClassSuffix}}_UBO_Data ubo[1])
{
  {{Kernel.InitSource}}
  {{Kernel.Name}}_Blocks({% include "inc_call_args.ispc" %}ubo, 0, {% if Kernel.threadDim == 1 %}{% if Kernel.ThreadId0.Start == "0" %}({{Kernel.ThreadId0.Size}}){% else %}(({{Kernel.ThreadId0.Size}}) - ({{Kernel.ThreadId0.Start}})){% endif %}{% else %}{% if Kernel.ThreadId0.Start == "0" %}({{Kernel.ThreadId0.Size}}){% else %}({% if Kernel.ThreadId0.Start == "0" %}({{Kernel.ThreadId0.Size}}){% else %}(({{Kernel.ThreadId0.Size}}) - ({{Kernel.ThreadId0.Start}})){% endif %}){% endif %}*{% if Kernel.ThreadId1.Start == "0" %}({{Kernel.ThreadId1.Size}}){% else %}({% if Kernel.ThreadId1.Start == "0" %}({{Kernel.ThreadId1.Size}}){% else %}(({{Kernel.ThreadId1.Size}}) - ({{Kernel.ThreadId1.Start}})){% endif %}){% endif %}{% endif %});
  {{Kernel.FinishSource}}
}
{% else if Kernel.OpenMPAndISPC %}
export void {{Kernel.Name}}_ISPC(
{% include "inc_args.ispc" %}
  uniform struct {{MainClassName}}{{MainClassSuffix}}_UBO_Data ubo[1],
  uniform int _start, uniform int _end, uniform int _ycoord)
{
  {{Kernel.Name}}_Blocks({% include "inc_call_args.ispc" %}ubo, _start, _end);
}
{% else %}
task void {{Kernel.Name}}_Task(
{% include "inc_args.ispc" %}
  uniform struct {{MainClassName}}{{MainClassSuffix}}_UBO_Data ubo[1],
  uniform int _start, uniform int _end, uniform int _span)
{
  const uniform int blockStart = _start + taskIndex * _span;
  const uniform int blockEnd   = min(blockStart + _span, _end);
  {{Kernel.Name}}_Blocks({% include "inc_call_args.ispc" %}ubo, blockStart, blockEnd);
}

export void {{Kernel.Name}}_ISPC(
{% include "inc_args.ispc" %}
  uniform struct {{MainClassName}}{{MainClassSuffix}}_UBO_Data ubo[1])
{
  {{Kernel.InitSource}}
  const uniform int blocks = {% if Kernel.threadDim == 1 %}{% if Kernel.ThreadId0.Start == "0" %}({{Kernel.ThreadId0.Size}}){% else %}(({{Kernel.ThreadId0.Size}}) - ({{Kernel.ThreadId0.Start}})){% endif %}{% else %}{% if Kernel.ThreadId0.Start == "0" %}({{Kernel.ThreadId0.Size}}){% else %}({% if Kernel.ThreadId0.Start == "0" %}({{Kernel.ThreadId0.Size}}){% else %}(({{Kernel.ThreadId0.Size}}) - ({{Kernel.ThreadId0.Start}})){% endif %}){% endif %}*{% if Kernel.ThreadId1.Start == "0" %}({{Kernel.ThreadId1.Size}}){% else %}({% if Kernel.ThreadId1.Start == "0" %}({{Kernel.ThreadId1.Size}}){% else %}(({{Kernel.ThreadId1.Size}}) - ({{Kernel.ThreadId1.Start}})){% endif %}){% endif %}{% endif %};
  const uniform int span   = max(1, 16384/(uniform int)({% for TID in Kernel.ThreadSizeBE %}{% if loop.index > 0 %}*{% endif %}{{TID.Name}}{% endfor %})); // blocks per task: about 16K threads
  launch[(blocks + span - 1)/span] {{Kernel.Name}}_Task({% include "inc_call_args.ispc" %}ubo, 0, blocks, span);
  sync;
  {{Kernel.FinishSource}}
}
{% endif %}
{% for TID in Kernel.ThreadSizeBE %}
#undef {{TID.Name}}
{% endfor %}
{% else %}
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
{% endif %} {# /* Kernel.EnableBlockExpansion */ #}
{% endif %} {# /* not Kernel.IsGraphics */ #}

## endfor
{% if HasGraphics %}
/////////////////////////////////////////////////////////////////////
/////////////////// draws of graphics pipeline (LiteRS) /////////////
/////////////////////////////////////////////////////////////////////
{% for Draw in GPDrawsISPC %}

// '{{Draw.Name}}': triangle after setup, the same layout as LiteRS::detail::Triangle<detail::FloatArray<{{Draw.NumFloats}}>>
//
struct {{Draw.Name}}_Tri
{
  int64  A[3];
  int64  B[3];
  int64  C[3];
  int64  bias[3];
  float  invArea;
  float  z[3];
  float  invW[3];
  int32  bbox[4];
  float  attr[{{Draw.NumFloats3}}];
};

// vertex shader for vertices [a_first, a_first + a_count): writes LiteRS::Vertex<detail::FloatArray<{{Draw.NumFloats}}>> with stride a_stride floats
//
export void {{Draw.Name}}_VS_ISPC(uniform uint32 a_first, uniform uint32 a_count,{% for Arg in Draw.UserArgs %} const void * uniform a_arg{{loop.index}},{% endfor %}
                                  uniform float a_out[], uniform uint32 a_stride)
{
  {% for Arg in Draw.UserArgs %}
  const uniform {{Arg.TypeISPC}} {{Arg.Name}} = *((uniform const {{Arg.TypeISPC}} * uniform)a_arg{{loop.index}});
  {% endfor %}
  foreach(i = 0 ... a_count)
  {
    {% if Draw.HasPS %}
    {{Draw.Varyings}} varyings;
    const float4 pos = {{Draw.VSOriginal}}(a_first + i{% for Arg in Draw.UserArgs %}, {{Arg.Name}}{% endfor %}, varyings);
    {% else %}
    const float4 pos = {{Draw.VSOriginal}}(a_first + i{% for Arg in Draw.UserArgs %}, {{Arg.Name}}{% endfor %});
    {% endif %}
    const int base = i*a_stride;
    a_out[base + 0] = pos.x;
    a_out[base + 1] = pos.y;
    a_out[base + 2] = pos.z;
    a_out[base + 3] = pos.w;
    {% for Comp in Draw.VaryingComps %}
    a_out[base + 4 + {{Comp.Offset}}] = varyings.{{Comp.Access}};
    {% endfor %}
  }
}

// shades pixel with edge functions (e0, e1, e2) if it is covered: depth test, perspective correct interpolation of
// varyings (detail::Interpolate), pixel shader, blending; the same formulas as in C++ (raster.h, DrawVertexStream)
//
static inline void {{Draw.Name}}_Shade(uniform const {{Draw.Name}}_Tri& t, const int64 e0, const int64 e1, const int64 e2, const int offs,
                                       uniform const int32 a_state[RS_STATE_SIZE], uniform float a_depth[]{% for Out in Draw.Outputs %},
                                       uniform {% if Out.IsUint %}uint32 {% else %}float4 {% endif %}a_color{{loop.index}}[]{% endfor %})
{
  if(((e0 + t.bias[0]) | (e1 + t.bias[1]) | (e2 + t.bias[2])) < 0)
    return;
  const float b0 = (float)e0*t.invArea;
  const float b1 = (float)e1*t.invArea;
  const float b2 = (float)e2*t.invArea;
  const float z  = b0*t.z[0] + b1*t.z[1] + b2*t.z[2];
  if(a_state[RS_STATE_DEPTH_TEST] != 0)
  {
    if(!RS_Compare(a_state[RS_STATE_DEPTH_COMPARE], z, a_depth[offs]))
      return;
    if(a_state[RS_STATE_DEPTH_WRITE] != 0)
      a_depth[offs] = z;
  }
  {% if Draw.HasPS %}
  const uniform int blendMode = a_state[RS_STATE_BLEND_MODE];
  const float q0 = b0*t.invW[0];
  const float q1 = b1*t.invW[1];
  const float q2 = b2*t.invW[2];
  const float s  = 1.0f/(q0 + q1 + q2);
  const float p0 = q0*s, p1 = q1*s, p2 = q2*s;
  {{Draw.Varyings}} varyings;
  {% for Comp in Draw.VaryingComps %}
  varyings.{{Comp.Access}} = p0*t.attr[{{Comp.Offset}}] + p1*t.attr[{{Comp.Offset1}}] + p2*t.attr[{{Comp.Offset2}}];
  {% endfor %}
  const {{Draw.OutputType}} psOut = {{Draw.PSOriginal}}(varyings);
  {% for Out in Draw.Outputs %}
  {
    const float4 c = psOut{% if Draw.OutputIsStruct %}.{{Out.Field}}{% endif %};
    {% if Out.IsUint %}
    a_color{{loop.index}}[offs] = RS_PackRGBA8((blendMode == 0) ? c : RS_Blend(blendMode, c, RS_UnpackRGBA8(a_color{{loop.index}}[offs])));
    {% else %}
    a_color{{loop.index}}[offs] = (blendMode == 0) ? c : RS_Blend(blendMode, c, a_color{{loop.index}}[offs]);
    {% endif %}
  }
  {% endfor %}
  {% endif %}
}

{% if Draw.HasPS %}
// shades pixels of small triangles collected in queue by full gangs (they already passed the depth test);
// a pixel is in the queue only once (pending[] in the bin), so colors are written in the order of triangles;
// without blending a pixel of the next triangle replaces the waiting one, so only visible pixels are shaded
//
static inline void {{Draw.Name}}_ShadeQueue(uniform const {{Draw.Name}}_Tri a_tris[], uniform RSPixelQueue& q, uniform int16 a_pending[],
                                            uniform const int32 a_state[RS_STATE_SIZE]{% for Out in Draw.Outputs %}, uniform {% if Out.IsUint %}uint32 {% else %}float4 {% endif %}a_color{{loop.index}}[]{% endfor %})
{
  const uniform int blendMode = a_state[RS_STATE_BLEND_MODE];
  foreach(i = 0 ... q.count)
  {
    const int   triId = q.tri[i];
    const int   offs  = q.offs[i];
    const float b0    = floatbits(q.b0[i]);
    const float b1    = floatbits(q.b1[i]);
    const float b2    = floatbits(q.b2[i]);
    const float q0    = b0*a_tris[triId].invW[0];
    const float q1    = b1*a_tris[triId].invW[1];
    const float q2    = b2*a_tris[triId].invW[2];
    const float s     = 1.0f/(q0 + q1 + q2);
    const float p0 = q0*s, p1 = q1*s, p2 = q2*s;
    {{Draw.Varyings}} varyings;
    {% for Comp in Draw.VaryingComps %}
    varyings.{{Comp.Access}} = p0*a_tris[triId].attr[{{Comp.Offset}}] + p1*a_tris[triId].attr[{{Comp.Offset1}}] + p2*a_tris[triId].attr[{{Comp.Offset2}}];
    {% endfor %}
    const {{Draw.OutputType}} psOut = {{Draw.PSOriginal}}(varyings);
    {% for Out in Draw.Outputs %}
    {
      const float4 c = psOut{% if Draw.OutputIsStruct %}.{{Out.Field}}{% endif %};
      {% if Out.IsUint %}
      a_color{{loop.index}}[offs] = RS_PackRGBA8((blendMode == 0) ? c : RS_Blend(blendMode, c, RS_UnpackRGBA8(a_color{{loop.index}}[offs])));
      {% else %}
      a_color{{loop.index}}[offs] = (blendMode == 0) ? c : RS_Blend(blendMode, c, a_color{{loop.index}}[offs]);
      {% endif %}
    }
    {% endfor %}
    a_pending[q.local[i]] = 0;
  }
  q.count = 0;
}

{% endif %}
// rasterizes triangles a_tris[a_ids[0 ... a_triNum-1]] inside rectangle a_bin = [x0, x1) x [y0, y1):
// small triangles per pixel (a gang over pixels of bounding box), the others by 4x4 blocks (a gang over 16 pixels of block,
// empty blocks are skipped by their corners, blocks of TILED4x4 targets are accessed as 16 subsequent pixels)
//
export void {{Draw.Name}}_Raster_ISPC(uniform const {{Draw.Name}}_Tri a_tris[], uniform const uint32 a_ids[], uniform uint32 a_triNum,
                                      uniform const int32 a_bin[4], uniform const int32 a_state[RS_STATE_SIZE], uniform float a_depth[]{% for Out in Draw.Outputs %},
                                      uniform {% if Out.IsUint %}uint32 {% else %}float4 {% endif %}a_color{{loop.index}}[]{% endfor %})
{
  const uniform int layout       = a_state[RS_STATE_LAYOUT];
  const uniform int pitch        = a_state[RS_STATE_PITCH];
  const uniform int algo         = a_state[RS_STATE_ALGO];
  const uniform int smallTriArea = a_state[RS_STATE_SMALL_TRI_AREA];
  const uniform int binShift     = a_state[RS_STATE_BIN_SHIFT];
  const uniform int tinyTriArea  = RS_TINY_TRI_AREA;
  {% if Draw.HasPS %}
  const uniform int  blendMode   = a_state[RS_STATE_BLEND_MODE];
  const uniform bool useQueue    = (a_bin[2] - a_bin[0] <= RS_MAX_BIN) && (a_bin[3] - a_bin[1] <= RS_MAX_BIN);
  uniform RSPixelQueue q;
  q.count = 0;
  uniform int16 pending[RS_MAX_BIN*RS_MAX_BIN]; // index + 1 of the pixel in the queue, 0 if it is not there
  if(useQueue)
  {
    foreach(i = 0 ... RS_MAX_BIN*RS_MAX_BIN)
      pending[i] = 0;
  }
  {% endif %}

  for(uniform uint32 k = 0; k < a_triNum; k++)
  {
    uniform const {{Draw.Name}}_Tri& t = a_tris[a_ids[k]];
    const uniform int x0 = max(t.bbox[0], a_bin[0]);
    const uniform int y0 = max(t.bbox[1], a_bin[1]);
    const uniform int x1 = min(t.bbox[2], a_bin[2]);
    const uniform int y1 = min(t.bbox[3], a_bin[3]);
    if(x0 >= x1 || y0 >= y1)
      continue;

    const uniform int  bboxArea = (t.bbox[2] - t.bbox[0])*(t.bbox[3] - t.bbox[1]);
    const uniform bool perPixel = (algo == 1) || (algo == 0 && bboxArea < smallTriArea);
    const uniform bool fitsInt  = (bboxArea < RS_MAX_INT32_AREA);          // edge functions inside the triangle fit int32
    {% if Draw.HasPS %}
    if(perPixel && useQueue && bboxArea <= tinyTriArea) // tiny triangles: scalar code, pixels go to the queue one by one
    {
      const uniform int64 px0 = (int64)x0*RS_SUBPIXEL_ONE + RS_SUBPIXEL_ONE/2;
      const uniform int64 py0 = (int64)y0*RS_SUBPIXEL_ONE + RS_SUBPIXEL_ONE/2;
      uniform int64 row0 = t.A[0]*px0 + t.B[0]*py0 + t.C[0];
      uniform int64 row1 = t.A[1]*px0 + t.B[1]*py0 + t.C[1];
      uniform int64 row2 = t.A[2]*px0 + t.B[2]*py0 + t.C[2];
      for(uniform int y = y0; y < y1; y++, row0 += t.B[0]*RS_SUBPIXEL_ONE, row1 += t.B[1]*RS_SUBPIXEL_ONE, row2 += t.B[2]*RS_SUBPIXEL_ONE)
      {
        uniform int64 e0 = row0, e1 = row1, e2 = row2;
        for(uniform int x = x0; x < x1; x++, e0 += t.A[0]*RS_SUBPIXEL_ONE, e1 += t.A[1]*RS_SUBPIXEL_ONE, e2 += t.A[2]*RS_SUBPIXEL_ONE)
        {
          if(((e0 + t.bias[0]) | (e1 + t.bias[1]) | (e2 + t.bias[2])) < 0)
            continue;
          const uniform float b0   = (float)e0*t.invArea;
          const uniform float b1   = (float)e1*t.invArea;
          const uniform float b2   = (float)e2*t.invArea;
          const uniform float z    = b0*t.z[0] + b1*t.z[1] + b2*t.z[2];
          const uniform int   offs = RS_PixelOffset(layout, pitch, binShift, x, y);
          if(a_state[RS_STATE_DEPTH_TEST] != 0)
          {
            if(!RS_Compare(a_state[RS_STATE_DEPTH_COMPARE], z, a_depth[offs]))
              continue;
            if(a_state[RS_STATE_DEPTH_WRITE] != 0)
              a_depth[offs] = z;
          }
          const uniform int local = (x - a_bin[0]) + (y - a_bin[1])*RS_MAX_BIN;
          if(q.count == RS_QUEUE_SIZE || (blendMode != 0 && pending[local] != 0))
            {{Draw.Name}}_ShadeQueue(a_tris, q, pending, a_state{% for Out in Draw.Outputs %}, a_color{{loop.index}}{% endfor %});
          const uniform int idx = (pending[local] != 0) ? pending[local] - 1 : q.count++;
          q.offs [idx] = offs;
          q.local[idx] = local;
          q.tri  [idx] = (int)a_ids[k];
          q.b0   [idx] = intbits(b0);
          q.b1   [idx] = intbits(b1);
          q.b2   [idx] = intbits(b2);
          pending[local] = idx + 1;
        }
      }
      continue;
    }
    if(perPixel && useQueue && fitsInt) // coverage and depth test now, pixel shader later for full gangs of pixels from many triangles
    {
      // edge functions are incremented as in C++ (RasterizePixels): AVX2 has no vector multiplication of int64
      //
      const uniform int64 px0 = (int64)x0*RS_SUBPIXEL_ONE + RS_SUBPIXEL_ONE/2;
      const uniform int64 py0 = (int64)y0*RS_SUBPIXEL_ONE + RS_SUBPIXEL_ONE/2;
      uniform int64 row0 = t.A[0]*px0 + t.B[0]*py0 + t.C[0];
      uniform int64 row1 = t.A[1]*px0 + t.B[1]*py0 + t.C[1];
      uniform int64 row2 = t.A[2]*px0 + t.B[2]*py0 + t.C[2];
      const int64 lane0 = t.A[0]*RS_SUBPIXEL_ONE*(int64)programIndex;
      const int64 lane1 = t.A[1]*RS_SUBPIXEL_ONE*(int64)programIndex;
      const int64 lane2 = t.A[2]*RS_SUBPIXEL_ONE*(int64)programIndex;
      for(uniform int y = y0; y < y1; y++, row0 += t.B[0]*RS_SUBPIXEL_ONE, row1 += t.B[1]*RS_SUBPIXEL_ONE, row2 += t.B[2]*RS_SUBPIXEL_ONE)
      {
        uniform int64 chunk0 = row0, chunk1 = row1, chunk2 = row2;
        for(uniform int xs = x0; xs < x1; xs += programCount, chunk0 += t.A[0]*RS_SUBPIXEL_ONE*programCount,
                                                              chunk1 += t.A[1]*RS_SUBPIXEL_ONE*programCount,
                                                              chunk2 += t.A[2]*RS_SUBPIXEL_ONE*programCount)
        {
          const int   x  = xs + programIndex;
          const int64 e0 = chunk0 + lane0;
          const int64 e1 = chunk1 + lane1;
          const int64 e2 = chunk2 + lane2;
          bool pass = (x < x1) && (((e0 + t.bias[0]) | (e1 + t.bias[1]) | (e2 + t.bias[2])) >= 0);
          if(!any(pass))
            continue;
          // inside a small triangle 0 <= e < 2*area*SUBPIXEL_ONE^2 fits int32: conversion of int32 to float is a vector
          // instruction (int64 is converted lane by lane) and gives the same value as in C++
          //
          const float b0   = (float)((int32)e0)*t.invArea;
          const float b1   = (float)((int32)e1)*t.invArea;
          const float b2   = (float)((int32)e2)*t.invArea;
          const float z    = b0*t.z[0] + b1*t.z[1] + b2*t.z[2];
          const int   offs = RS_PixelOffset(layout, pitch, binShift, x, y);
          if(pass && a_state[RS_STATE_DEPTH_TEST] != 0)
            pass = RS_Compare(a_state[RS_STATE_DEPTH_COMPARE], z, a_depth[offs]);
          if(!any(pass))
            continue;
          const int local = (x - a_bin[0]) + (y - a_bin[1])*RS_MAX_BIN;
          if(q.count + programCount > RS_QUEUE_SIZE || (blendMode != 0 && any(pass && pending[local] != 0))) // blending: keep order of colors in a pixel
            {{Draw.Name}}_ShadeQueue(a_tris, q, pending, a_state{% for Out in Draw.Outputs %}, a_color{{loop.index}}{% endfor %});
          if(pass)
          {
            if(a_state[RS_STATE_DEPTH_TEST] != 0 && a_state[RS_STATE_DEPTH_WRITE] != 0)
              a_depth[offs] = z;
            // without blending the pixel waiting in the queue is replaced: its color would be overwritten by this one
            //
            const int pend = pending[local];
            const int idx  = (pend != 0) ? pend - 1 : q.count + exclusive_scan_add((pend == 0) ? 1 : 0);
            q.offs [idx] = offs;
            q.local[idx] = local;
            q.tri  [idx] = (int)a_ids[k];
            q.b0   [idx] = intbits(b0);
            q.b1   [idx] = intbits(b1);
            q.b2   [idx] = intbits(b2);
            pending[local] = idx + 1;
            q.count += reduce_add((pend == 0) ? 1 : 0);
          }
        }
      }
      continue;
    }
    if(q.count != 0) // large triangle writes colors directly, pixels of previous triangles are shaded before it
      {{Draw.Name}}_ShadeQueue(a_tris, q, pending, a_state{% for Out in Draw.Outputs %}, a_color{{loop.index}}{% endfor %});
    {% endif %}
    if(perPixel)
    {
      foreach(y = y0 ... y1, x = x0 ... x1)
      {
        const int64 px = (int64)x*RS_SUBPIXEL_ONE + RS_SUBPIXEL_ONE/2;
        const int64 py = (int64)y*RS_SUBPIXEL_ONE + RS_SUBPIXEL_ONE/2;
        const int64 e0 = t.A[0]*px + t.B[0]*py + t.C[0];
        const int64 e1 = t.A[1]*px + t.B[1]*py + t.C[1];
        const int64 e2 = t.A[2]*px + t.B[2]*py + t.C[2];
        if(!any(((e0 + t.bias[0]) | (e1 + t.bias[1]) | (e2 + t.bias[2])) >= 0)) // most of small triangles cover no pixel centers
          continue;
        {{Draw.Name}}_Shade(t, e0, e1, e2, RS_PixelOffset(layout, pitch, binShift, x, y), a_state, a_depth{% for Out in Draw.Outputs %}, a_color{{loop.index}}{% endfor %});
      }
      continue;
    }

    // edge function steps and offsets from the top left pixel of a block to its corner with minimal/maximal value (RasterizeBlocks)
    //
    uniform int64 stepX[3], stepY[3], maxOffs[3];
    for(uniform int i = 0; i < 3; i++)
    {
      stepX[i]   = t.A[i]*RS_SUBPIXEL_ONE;
      stepY[i]   = t.B[i]*RS_SUBPIXEL_ONE;
      maxOffs[i] = max((uniform int64)0, stepX[i]*3) + max((uniform int64)0, stepY[i]*3);
    }
    int64 blockOffs[3][RS_BLOCK_PARTS]; // edge function offsets of the 16 pixels of a block from its top left pixel, gang by gang
    for(uniform int part = 0; part*programCount < 16; part++)
    {
      const int i = part*programCount + programIndex;
      for(uniform int k = 0; k < 3; k++)
        blockOffs[k][part] = stepX[k]*(int64)(i & 3) + stepY[k]*(int64)(i >> 2);
    }

    for(uniform int by = RS_FloorDiv(y0, 4)*4; by < y1; by += 4)
    {
      for(uniform int bx = RS_FloorDiv(x0, 4)*4; bx < x1; bx += 4)
      {
        const uniform int64 px = (int64)bx*RS_SUBPIXEL_ONE + RS_SUBPIXEL_ONE/2;
        const uniform int64 py = (int64)by*RS_SUBPIXEL_ONE + RS_SUBPIXEL_ONE/2;
        uniform int64 e[3];
        uniform bool  empty = false;
        for(uniform int i = 0; i < 3; i++)
        {
          e[i]  = t.A[i]*px + t.B[i]*py + t.C[i];
          empty = empty || (e[i] + maxOffs[i] + t.bias[i] < 0);
        }
        if(empty)
          continue;

        const uniform bool inside = (bx >= x0 && by >= y0 && bx + 4 <= x1 && by + 4 <= y1);
        const uniform int  base   = RS_PixelOffset(layout, pitch, binShift, bx, by);
        for(uniform int part = 0; part*programCount < 16; part++)
        {
          const int i = part*programCount + programIndex;
          if(i >= 16)
            continue;
          const int x = bx + (i & 3);
          const int y = by + (i >> 2);
          if(!inside && (x < x0 || x >= x1 || y < y0 || y >= y1))
            continue;
          const int offs = (inside && layout != 0) ? base + i : RS_PixelOffset(layout, pitch, binShift, x, y); // TILED4x4(_BINS): the block is 16 subsequent pixels
          {{Draw.Name}}_Shade(t, e[0] + blockOffs[0][part], e[1] + blockOffs[1][part], e[2] + blockOffs[2][part],
                              offs, a_state, a_depth{% for Out in Draw.Outputs %}, a_color{{loop.index}}{% endfor %});
        }
      }
    }
  }
  {% if Draw.HasPS %}
  if(q.count != 0)
    {{Draw.Name}}_ShadeQueue(a_tris, q, pending, a_state{% for Out in Draw.Outputs %}, a_color{{loop.index}}{% endfor %});
  {% endif %}
}
{% endfor %}
{% endif %}
