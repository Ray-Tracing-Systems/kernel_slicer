#version 450
#extension GL_ARB_separate_shader_objects : enable

layout (points) in;
layout (triangle_strip, max_vertices = 4) out;
//layout (line_strip, max_vertices = 2) out;

layout(push_constant) uniform params_t
{
    mat4 mProjView;
    mat4 mModel;
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
} vOut;

void gen_quad(float size)
{
    vec3 pos = vOut.wPos.xyz;
    gl_Position = params.mProjView * vec4(pos + vec3(-size, -size, 0), 1.0);
    EmitVertex();
    gl_Position = params.mProjView * vec4(pos + vec3(-size, +size, 0), 1.0);
    EmitVertex();
    gl_Position = params.mProjView * vec4(pos + vec3(+size, -size, 0), 1.0);
    EmitVertex();
    gl_Position = params.mProjView * vec4(pos + vec3(+size, +size, 0), 1.0);
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
    vOut.wPos = point[0].wPos;
    vOut.wVel = point[0].wVel;
    gen_quad(0.01f);
//    gen_line(1.0f);
}