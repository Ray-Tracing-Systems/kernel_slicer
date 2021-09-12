#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "shader_common.h"

layout(location = 0) in vec4 vPos;
layout(location = 1) in vec4 vVel;

layout(push_constant) uniform params_t
{
    mat4 mProjView;
    vec4 mCamPos;
} params;

layout (location = 0 ) out VS_OUT
{
    vec4 wPosWeight;
    vec4 wVelCharge;
} vOut;

out gl_PerVertex { vec4 gl_Position; float gl_PointSize;};
void main(void)
{
    vOut.wPosWeight = vPos;
    vOut.wVelCharge = vVel;
    gl_Position = params.mProjView * vec4(vOut.wPosWeight.xyz, 1.0);
    gl_PointSize = 2.0f;
}
