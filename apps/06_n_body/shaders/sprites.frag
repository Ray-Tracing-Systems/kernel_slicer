#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) out vec4 out_fragColor;

layout (location = 0 ) in VS_OUT
{
    vec3 wPos;
    vec3 wVel;
} point;


//layout(binding = 0, set = 0) uniform AppData
//{
//    UniformParams Params;
//};


void main()
{
    const float velocityScale = 0.1f;
    out_fragColor = vec4(abs(point.wVel * velocityScale), 1.0f);
//    out_fragColor = vec4(1.0f);
}