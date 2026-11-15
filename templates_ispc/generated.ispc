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

// rasterizes triangles a_tris[a_ids[0 ... a_triNum-1]] inside rectangle a_bin = [x0, x1) x [y0, y1)
//
export void {{Draw.Name}}_Raster_ISPC(uniform const {{Draw.Name}}_Tri a_tris[], uniform const uint32 a_ids[], uniform uint32 a_triNum,
                                      uniform const int32 a_bin[4], uniform const int32 a_state[8], uniform float a_depth[]{% for Out in Draw.Outputs %},
                                      uniform {% if Out.IsUint %}uint32 {% else %}float4 {% endif %}a_color{{loop.index}}[]{% endfor %})
{
  const uniform int  layout       = a_state[RS_STATE_LAYOUT];
  const uniform int  pitch        = a_state[RS_STATE_PITCH];
  const uniform bool depthTest    = (a_state[RS_STATE_DEPTH_TEST] != 0);
  const uniform bool depthWrite   = (a_state[RS_STATE_DEPTH_WRITE] != 0);
  const uniform int  depthCompare = a_state[RS_STATE_DEPTH_COMPARE];
  const uniform int  blendMode    = a_state[RS_STATE_BLEND_MODE];

  for(uniform uint32 k = 0; k < a_triNum; k++)
  {
    uniform const {{Draw.Name}}_Tri& t = a_tris[a_ids[k]];
    const uniform int x0 = max(t.bbox[0], a_bin[0]);
    const uniform int y0 = max(t.bbox[1], a_bin[1]);
    const uniform int x1 = min(t.bbox[2], a_bin[2]);
    const uniform int y1 = min(t.bbox[3], a_bin[3]);
    if(x0 >= x1 || y0 >= y1)
      continue;

    for(uniform int by = RS_FloorDiv(y0, 4)*4; by < y1; by += 4)
    {
      for(uniform int bx = RS_FloorDiv(x0, 4)*4; bx < x1; bx += 4)
      {
        foreach(i = 0 ... 16) // 4x4 block, the same as a tile of LiteImage::Layout::TILED4x4
        {
          const int x = bx + (i & 3);
          const int y = by + (i >> 2);
          if(x < x0 || x >= x1 || y < y0 || y >= y1)
            continue;
          const int64 px = (int64)x*RS_SUBPIXEL_ONE + RS_SUBPIXEL_ONE/2;
          const int64 py = (int64)y*RS_SUBPIXEL_ONE + RS_SUBPIXEL_ONE/2;
          const int64 e0 = t.A[0]*px + t.B[0]*py + t.C[0];
          const int64 e1 = t.A[1]*px + t.B[1]*py + t.C[1];
          const int64 e2 = t.A[2]*px + t.B[2]*py + t.C[2];
          if(((e0 + t.bias[0]) | (e1 + t.bias[1]) | (e2 + t.bias[2])) < 0)
            continue;

          const float b0   = (float)e0*t.invArea;
          const float b1   = (float)e1*t.invArea;
          const float b2   = (float)e2*t.invArea;
          const int   offs = RS_PixelOffset(layout, pitch, x, y);
          const float z    = b0*t.z[0] + b1*t.z[1] + b2*t.z[2];
          if(depthTest)
          {
            if(!RS_Compare(depthCompare, z, a_depth[offs]))
              continue;
            if(depthWrite)
              a_depth[offs] = z;
          }
          {% if Draw.HasPS %}

          // perspective correct interpolation of varyings (detail::Interpolate) and pixel shader
          //
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
      }
    }
  }
}
{% endfor %}
{% endif %}
