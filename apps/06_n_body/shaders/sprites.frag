#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "shader_common.h"

layout(location = 0) out vec4 out_fragColor;

layout (location = 0 ) in VS_OUT
{
    vec4 wPosWeight;
    vec4 wVelCharge;
    vec2 wTexCoord;
} point;

layout(binding = 0) uniform sampler2D spriteTex;
layout(binding = 1) uniform sampler1D colormapTex;

void main()
{
    out_fragColor =  texture(spriteTex, point.wTexCoord);
    if(point.wPosWeight.w > weightThres)
        out_fragColor.rgb *= vec3(0.0f, 0.75f, 0.0f);
    else
    {
        out_fragColor.rgb *= colorMap(point.wVelCharge.rgb, colormapTex);
        if(point.wVelCharge.w > 0.0f)
        {
            vec3 tmp = out_fragColor.rgb;
            out_fragColor.r = tmp.g;
            out_fragColor.g = tmp.b;
            out_fragColor.b = tmp.r;
        }

    }



}