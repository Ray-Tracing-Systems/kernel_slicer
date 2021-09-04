// Original Copyright 2017 Vladimir Frolov, Ray Tracing Systems
// Updated by Vadim Sanzharov, 2021
#pragma once

#include "LiteMath.h"

using LiteMath::float3;
using LiteMath::float4;
using LiteMath::float4x4;
using LiteMath::DEG_TO_RAD;

struct Camera
{
  Camera() : pos(0.0f, 0.0f, +5.0f), lookAt(0, 0, 0), up(0, 1, 0), fov(45.0f), tdist(100.0f) {}

  float3 pos;
  float3 lookAt;
  float3 up;
  float  fov;
  float  tdist;

  float3 forward() const { return normalize(lookAt - pos); }
  float3 right()   const { return cross(forward(), up); }

  void offsetOrientation(float a_upAngle, float rightAngle)
  {
    if (a_upAngle != 0.0f)  // rotate vertical
    {
      float3 direction = normalize(forward() * cosf(-DEG_TO_RAD*a_upAngle) + up * sinf(-DEG_TO_RAD*a_upAngle));

      up     = normalize(cross(right(), direction));
      lookAt = pos + tdist*direction;
    }

    if (rightAngle != 0.0f)  // rotate horizontal
    {
      float4x4 rot;

      rot[0][0] = rot[2][2] = cosf(DEG_TO_RAD * rightAngle);
      rot[0][2] = -sinf(DEG_TO_RAD * rightAngle);
      rot[2][0] = +sinf(DEG_TO_RAD * rightAngle);

      float3 direction2 = LiteMath::normalize(mul(rot, forward()));
      up     = normalize(mul(rot, up));
      lookAt = pos + tdist*direction2;
    }
  }

  void offsetPosition(float3 a_offset)
  {
    pos    += a_offset;
    lookAt += a_offset;
  }
};


// http://matthewwellings.com/blog/the-new-vulkan-coordinate-system/
//
static inline float4x4 OpenglToVulkanProjectionMatrixFix()
{
  float4x4 res;
  res[1][1] = -1.0f;
  res[2][2] = 0.5f;
  res[2][3] = 0.5f;
  return res;
}

static inline float4x4 projectionMatrix(float fovy, float aspect, float zNear, float zFar)
{
  float4x4 res;
  const float ymax = zNear * tanf(fovy * DEG_TO_RAD * 0.5f);
  const float xmax = ymax * aspect;

  const float left   = -xmax;
  const float right  = +xmax;
  const float bottom = -ymax;
  const float top    = +ymax;

  const float temp = 2.0f * zNear;
  const float temp2 = right - left;
  const float temp3 = top - bottom;
  const float temp4 = zFar - zNear;

  res(0,0) = temp / temp2;
  res(1,0) = 0.0;
  res(2,0) = 0.0;
  res(3,0) = 0.0;

  res(0,1) = 0.0;
  res(1,1) = temp / temp3;
  res(2,1) = 0.0;
  res(3,1) = 0.0;

  res(0, 2) = (right + left) / temp2;
  res(1, 2) = (top + bottom) / temp3;
  res(2, 2) = (-zFar - zNear) / temp4;
  res(3, 2) = -1.0;

  res(0, 3) = 0.0;
  res(1, 3) = 0.0;
  res(2, 3) = (-temp * zFar) / temp4;
  res(3, 3) = 0.0;

  return res;
}

static inline float4x4 perspectiveMatrix(float fovy, float aspect, float zNear, float zFar)
{
  const float ymax = zNear * tanf(fovy * 3.14159265358979323846f / 360.0f);
  const float xmax = ymax * aspect;
  const float left = -xmax;
  const float right = +xmax;
  const float bottom = -ymax;
  const float top = +ymax;
  const float temp = 2.0f * zNear;
  const float temp2 = right - left;
  const float temp3 = top - bottom;
  const float temp4 = zFar - zNear;
  float4x4 res;
  res.set_col(0, float4{ temp / temp2, 0.0f, 0.0f, 0.0f });
  res.set_col(1, float4{ 0.0f, temp / temp3, 0.0f, 0.0f });
  res.set_col(2, float4{ (right + left) / temp2,  (top + bottom) / temp3, (-zFar - zNear) / temp4, -1.0 });
  res.set_col(3, float4{ 0.0f, 0.0f, (-temp * zFar) / temp4, 0.0f });
  return res;
}

static inline float4x4 ortoMatrix(const float l, const float r, const float b, const float t, const float n, const float f)
{
  float4x4 res;
  res(0,0) = 2.0f / (r - l);
  res(0,1) = 0;
  res(0,2) = 0;
  res(0,3) = -(r + l) / (r - l);
  res(1,0) = 0;
  res(1,1) = -2.0f / (t - b);  // why minus ??? check it for OpenGL please
  res(1,2) = 0;
  res(1,3) = -(t + b) / (t - b);
  res(2,0) = 0;
  res(2,1) = 0;
  res(2,2) = -2.0f / (f - n);
  res(2,3) = -(f + n) / (f - n);
  res(3,0) = 0.0f;
  res(3,1) = 0.0f;
  res(3,2) = 0.0f;
  res(3,3) = 1.0f;
  return res;
}