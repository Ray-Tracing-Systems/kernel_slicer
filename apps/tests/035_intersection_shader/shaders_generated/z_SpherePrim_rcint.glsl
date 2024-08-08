#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : require

#include "common_generated.h"
layout(binding = 0, set = 0) buffer data0 { uint out_color[]; }; //
layout(binding = 1, set = 0) uniform accelerationStructureEXT m_pRayTraceImpl;
layout(binding = 2, set = 0) buffer dataUBO { TestClass_Generated_UBO_Data ubo; };
layout(binding = 3, set = 0) buffer data1 { AbtractPrimitive m_pRayTraceImpl_primitives[]; }; //

//struct CRT_Hit 
//{
//  float    t;         ///< intersection distance from ray origin to object
//  uint     primId; 
//  uint     instId;
//  uint     geomId;    ///< use 4 most significant bits for geometry type; thay are zero for triangles 
//  float    coords[4]; ///< custom intersection data; for triangles coords[0] and coords[1] stores baricentric coords (u,v)
//};

hitAttributeEXT vec3 attribs;

// this method is documented in raytracing gems book
vec2 gems_intersections(vec3 orig, vec3 dir, vec3 center, float radius)
{
	vec3 f = orig - center;
	float a = dot(dir, dir);
	float bi = dot(-f, dir);
	float c = dot(f, f) - radius * radius;
	vec3 s = f + (bi/a)*dir;
	float discr = radius * radius - dot(s, s);

	vec2 t = vec2(-1.0, -1.0);
	if (discr >= 0) {
		float q = bi + sign(bi) * sqrt(a*discr);
		float t1 = c / q;
		float t2 = q / a;
		t = vec2(t1, t2);
	}
	return t;
}

void main()
{
  //vec4  boxMin = m_pRayTraceImpl_primitives[gl_PrimitiveID].boxMin; // SEEMS using gl_PrimitiveID is INCORRECT HERE 
  //vec4  boxMax = m_pRayTraceImpl_primitives[gl_PrimitiveID].boxMax; // SEEMS using gl_PrimitiveID is INCORRECT HERE
  //vec3  center = (boxMin + boxMax).xyz*0.5f;
  //float radius = (boxMax.x - boxMin.x)*0.5f;
  //
  //vec3 orig = gl_WorldRayOriginEXT;
  //vec3 dir  = gl_WorldRayDirectionEXT;
  //vec2 t    = gems_intersections(orig, dir, center, radius);
  //
  //attribs = orig + t.x * dir;
  //reportIntersectionEXT(t.x, 0);
  //attribs = orig + t.y * dir;
  //reportIntersectionEXT(t.y, 0);	

  reportIntersectionEXT(1.0f, 0);
}

