#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_ray_tracing : require 

#include "common_generated.h"
layout(binding = 0, set = 0) buffer data0 { uint out_color[]; }; //
layout(binding = 1, set = 0) uniform accelerationStructureEXT m_pRayTraceImpl;
layout(binding = 2, set = 0) buffer dataUBO { TestClass_Generated_UBO_Data ubo; };


uint pitchOffset(uint x, uint y) ;

uint pitchOffset(uint x, uint y) { return y*uint(WIN_WIDTH) + x; }

// RayScene intersection with 'm_pRayTraceImpl'
//
layout(location = 0) rayPayloadEXT CRT_Hit m_pRayTraceImpl_hitValue;
layout(location = 1) rayPayloadEXT bool    m_pRayTraceImpl_inShadow;

CRT_Hit m_pRayTraceImpl_RayQuery_NearestHit(const vec4 rayPos, const vec4 rayDir)
{
  traceRayEXT(m_pRayTraceImpl, gl_RayFlagsOpaqueEXT, 0xff, 0, 0, 0, rayPos.xyz, rayPos.w, rayDir.xyz, rayDir.w, 0);
  return m_pRayTraceImpl_hitValue;
}

CRT_Hit m_pRayTraceImpl_RayQuery_NearestHitMotion(const vec4 rayPos, const vec4 rayDir, float t)
{
  traceRayEXT(m_pRayTraceImpl, gl_RayFlagsOpaqueEXT, 0xff, 0, 0, 0, rayPos.xyz, rayPos.w, rayDir.xyz, rayDir.w, 0);
  return m_pRayTraceImpl_hitValue;
}

bool m_pRayTraceImpl_RayQuery_AnyHit(const vec4 rayPos, const vec4 rayDir)
{
  m_pRayTraceImpl_inShadow = true;
  traceRayEXT(m_pRayTraceImpl, gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT,
              0xff, 0, 0, 1, rayPos.xyz, rayPos.w, rayDir.xyz, rayDir.w, 1);
  return m_pRayTraceImpl_inShadow;
}

bool m_pRayTraceImpl_RayQuery_AnyHitMotion(const vec4 rayPos, const vec4 rayDir, float t)
{
  m_pRayTraceImpl_inShadow = true;
  traceRayEXT(m_pRayTraceImpl, gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT,
              0xff, 0, 0, 1, rayPos.xyz, rayPos.w, rayDir.xyz, rayDir.w, 1);
  return m_pRayTraceImpl_inShadow;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
layout( push_constant ) uniform kernelArgs
{
  uint iNumElementsX; 
  uint iNumElementsY; 
  uint iNumElementsZ; 
  uint tFlagsMask;    
} kgenArgs;

///////////////////////////////////////////////////////////////// subkernels here
void kernel_TestColor_out_color(in int in_hit, uint out_colorOffset, uint tidX, uint tidY) 
{
  
  if(in_hit != -1)
    out_color[pitchOffset(tidX,tidY) + out_colorOffset] = 0x0000FFFF;
  else
    out_color[pitchOffset(tidX,tidY) + out_colorOffset] = 0x00FF0000;

}

void kernel_RayTrace(in vec4 rayPosAndNear, inout vec4 rayDirAndFar, inout int out_hit, uint tidX, uint tidY) 
{
  
  CRT_Hit hit = m_pRayTraceImpl_RayQuery_NearestHit(rayPosAndNear, rayDirAndFar); 
  out_hit = int(hit.primId);

}

void kernel_InitEyeRay(inout uint flags, inout vec4 rayPosAndNear, inout vec4 rayDirAndFar, uint tidX, uint tidY) 
{
  
  const float x = float(tidX)*ubo.m_widthInv;
  const float y = float(tidY)*ubo.m_heightInv;
  (rayPosAndNear) = vec4(x,y,-1.0f,0.0f);
  (rayDirAndFar ) = vec4(0,0,1,FLT_MAX);
  flags           = 0;

}

///////////////////////////////////////////////////////////////// subkernels here

void main()
{
  ///////////////////////////////////////////////////////////////// prolog
  const uint tidX = uint(gl_LaunchIDEXT[0]); 
  const uint tidY = uint(gl_LaunchIDEXT[1]); 
  ///////////////////////////////////////////////////////////////// prolog

  
  vec4 rayPosAndNear,  rayDirAndFar;
  uint   flags;
  int    hit;

  kernel_InitEyeRay(flags, rayPosAndNear, rayDirAndFar, tidX, tidY);
  kernel_RayTrace  (rayPosAndNear, rayDirAndFar, hit, tidX, tidY);
  kernel_TestColor_out_color(hit, 0, tidX, tidY);

}

