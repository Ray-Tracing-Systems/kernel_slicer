#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "shader_common.h"

layout(location = 0) out vec4 out_fragColor;

layout (location = 0 ) in VS_OUT
{
    vec4 wPosWeight;
    vec4 wVelCharge;
} point;

layout(binding = 0) uniform sampler1D colormapTex;

void main()
{
//    out_fragColor = vec4(colorMap(point.wVel, colormapTex), 1.0f);
    out_fragColor = vec4(colorMapWeight(point.wPosWeight.w, colormapTex), 1.0f);
}