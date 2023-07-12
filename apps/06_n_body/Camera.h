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
