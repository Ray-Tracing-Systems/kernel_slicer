#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "shader_common.h"

layout (points) in;
layout (triangle_strip, max_vertices = 4) out;
//layout (line_strip, max_vertices = 2) out;

layout(push_constant) uniform params_t
{
    mat4 mProjView;
    vec4 mCamPos;
} params;

layout (location = 0 ) in GS_IN
{
    vec4 wPosWeight;
    vec4 wVelCharge;
} point[];

layout (location = 0 ) out GS_OUT
{
    vec4 wPosWeight;
    vec4 wVelCharge;
    vec2 wTexCoord;
} vOut;

void gen_quad(float size)
{
    vec3 pos = point[0].wPosWeight.xyz;
    vec3 toCam = normalize(params.mCamPos.xyz - pos);
    vec3 up = vec3(0.0f, 1.0f, 0.0f);
    vec3 right = cross(toCam, up);

    // bottom left
    vOut.wPosWeight = point[0].wPosWeight;
    vOut.wVelCharge = point[0].wVelCharge;
    vOut.wTexCoord = vec2(0.0f, 1.0f);
    pos.y -= size;
    pos -= size * right;
    gl_Position = params.mProjView * vec4(pos, 1.0);
    EmitVertex();

    // upper left
    vOut.wPosWeight = point[0].wPosWeight;
    vOut.wVelCharge = point[0].wVelCharge;
    vOut.wTexCoord = vec2(0.0f, 0.0f);
    pos.y += 2 * size;
    gl_Position = params.mProjView * vec4(pos, 1.0);
    EmitVertex();

    // bottom right
    vOut.wPosWeight = point[0].wPosWeight;
    vOut.wVelCharge = point[0].wVelCharge;
    vOut.wTexCoord = vec2(1.0f, 1.0f);
    pos.y -= 2 * size;
    pos += 2 * size * right;
    gl_Position = params.mProjView * vec4(pos, 1.0);
    EmitVertex();

    // top right
    vOut.wPosWeight = point[0].wPosWeight;
    vOut.wVelCharge = point[0].wVelCharge;
    vOut.wTexCoord = vec2(1.0f, 0.0f);
    pos.y += 2 * size;
    gl_Position = params.mProjView * vec4(pos, 1.0);
    EmitVertex();
    EndPrimitive();
}


void gen_line(float size)
{
    vec3 pos = vOut.wPosWeight.xyz;
    gl_Position = params.mProjView * vec4(pos, 1.0);
    EmitVertex();
    gl_Position = params.mProjView * vec4(pos + vec3(0, +size, 0), 1.0);
    EmitVertex();
    EndPrimitive();
}

void main ()
{
    if(point[0].wPosWeight.w > weightThres)
        gen_quad(spriteSize * 2.f);
    else
        gen_quad(spriteSize);
}