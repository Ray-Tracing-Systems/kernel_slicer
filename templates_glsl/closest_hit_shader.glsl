#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_buffer_reference     : require

{% include "common_generated.glsl" %}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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

struct CRT_Hit 
{
  float    t;         ///< intersection distance from ray origin to object
  uint     primId; 
  uint     instId;
  uint     geomId;    ///< use 4 most significant bits for geometry type; thay are zero for triangles 
  float    coords[4]; ///< custom intersection data; for triangles coords[0] and coords[1] stores baricentric coords (u,v)
};

layout(location = 0) rayPayloadInEXT CRT_Hit kgen_hitValue;
hitAttributeEXT CRT_Hit attribs;

void main()
{
  kgen_hitValue = attribs;
  //kgen_hitValue.primId = gl_PrimitiveID;
  //kgen_hitValue.geomId = gl_InstanceCustomIndexEXT; 
  //kgen_hitValue.instId = gl_InstanceID;
  //kgen_hitValue.t      = gl_HitTEXT;
  //
  //kgen_hitValue.coords[0] = attribs.y;
  //kgen_hitValue.coords[1] = attribs.x;
  //kgen_hitValue.coords[2] = 1.0f - attribs.x - attribs.y;
  //kgen_hitValue.coords[3] = 0.0f;
}

