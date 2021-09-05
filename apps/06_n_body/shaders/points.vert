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
    vec3 wPos;
    vec3 wVel;
} vOut;

out gl_PerVertex { vec4 gl_Position; float gl_PointSize;};
void main(void)
{
    vOut.wPos   = vec4(vPos.xyz, 1.0f).xyz;
    vOut.wVel   = vVel.xyz;
    gl_Position = params.mProjView * vec4(vOut.wPos, 1.0);
    gl_PointSize = 2.0f;
}
