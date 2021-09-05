#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "shader_common.h"

layout(location = 0) out vec4 out_fragColor;

layout (location = 0 ) in VS_OUT
{
    vec3 wPos;
    vec3 wVel;
} point;

void main()
{
    out_fragColor = vec4(abs(point.wVel * velocityColorScale), 1.0f);
}