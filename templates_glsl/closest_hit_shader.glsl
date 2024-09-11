#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_buffer_reference     : require
#extension GL_EXT_ray_tracing          : require

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
}

