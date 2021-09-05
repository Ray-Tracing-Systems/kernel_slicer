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
    vec3 wPos;
    vec3 wVel;
} point[];

layout (location = 0 ) out GS_OUT
{
    vec3 wPos;
    vec3 wVel;
    vec2 wTexCoord;
} vOut;

void gen_quad(float size)
{
    vec3 pos = point[0].wPos;
    vec3 toCam = normalize(params.mCamPos.xyz - pos);
    vec3 up = vec3(0.0f, 1.0f, 0.0f);
    vec3 right = cross(toCam, up);

    // bottom left
    vOut.wPos = point[0].wPos;
    vOut.wVel = point[0].wVel;
    vOut.wTexCoord = vec2(0.0f, 1.0f);
    pos.y -= size;
    pos -= size * right;
    gl_Position = params.mProjView * vec4(pos, 1.0);
    EmitVertex();

    // upper left
    vOut.wPos = point[0].wPos;
    vOut.wVel = point[0].wVel;
    vOut.wTexCoord = vec2(0.0f, 0.0f);
    pos.y += 2 * size;
    gl_Position = params.mProjView * vec4(pos, 1.0);
    EmitVertex();

    // bottom right
    vOut.wPos = point[0].wPos;
    vOut.wVel = point[0].wVel;
    vOut.wTexCoord = vec2(1.0f, 1.0f);
    pos.y -= 2 * size;
    pos += 2 * size * right;
    gl_Position = params.mProjView * vec4(pos, 1.0);
    EmitVertex();

    // top right
    vOut.wPos = point[0].wPos;
    vOut.wVel = point[0].wVel;
    vOut.wTexCoord = vec2(1.0f, 0.0f);
    pos.y += 2 * size;
    gl_Position = params.mProjView * vec4(pos, 1.0);
    EmitVertex();
    EndPrimitive();
}


void gen_line(float size)
{
    vec3 pos = vOut.wPos.xyz;
    gl_Position = params.mProjView * vec4(pos, 1.0);
    EmitVertex();
    gl_Position = params.mProjView * vec4(pos + vec3(0, +size, 0), 1.0);
    EmitVertex();
    EndPrimitive();
}

void main ()
{
    gen_quad(spriteSize);
}