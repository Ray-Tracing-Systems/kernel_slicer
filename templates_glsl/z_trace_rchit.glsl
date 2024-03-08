#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : require

struct CRT_Hit 
{
  float    t;         ///< intersection distance from ray origin to object
  uint     primId; 
  uint     instId;
  uint     geomId;    ///< use 4 most significant bits for geometry type; thay are zero for triangles 
  float    coords[4]; ///< custom intersection data; for triangles coords[0] and coords[1] stores baricentric coords (u,v)
};

layout(location = 0) rayPayloadInEXT CRT_Hit kgen_hitValue;
hitAttributeEXT vec3 attribs;

void main()
{
  kgen_hitValue.primId = gl_PrimitiveID;
  kgen_hitValue.geomId = gl_InstanceCustomIndexEXT;
  kgen_hitValue.instId = gl_InstanceID;
  kgen_hitValue.t      = gl_HitTEXT;
 
  kgen_hitValue.coords[0] = attribs.y;
  kgen_hitValue.coords[1] = attribs.x;
  kgen_hitValue.coords[2] = 1.0f - attribs.x - attribs.y;
  kgen_hitValue.coords[3] = 0.0f;
}
