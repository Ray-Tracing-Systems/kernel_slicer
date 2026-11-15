#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_samplerless_texture_functions : require
{% if Kernel.NeedTexArray %}
#extension GL_EXT_nonuniform_qualifier : require
{% endif %}
{% if existsIn(Kernel, "UseScalarLayout") and Kernel.UseScalarLayout %}
#extension GL_EXT_scalar_block_layout : require
{% endif %}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Graphics pipeline, {% if Kernel.GPStage == 1 %}vertex shader '{% else %}pixel shader '{% endif %}{{Kernel.Name}}'. Bindings of this shader are in descriptor set {{Kernel.DSetId}}.
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

{% include "common_generated.glsl" %}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

{% if Kernel.GPStage == 1 %}
layout( push_constant ) uniform kernelArgs
{
  {% for UserArg in Kernel.UserArgs %}
  {{UserArg.Type}} {{UserArg.Name}};
  {% endfor %}
  uint gpDummy;
} kgenArgs;

{% if Kernel.GPHasVaryings %}
layout(location = 0) out {{Kernel.GPVaryings}} gpVaryings; // output of vertex shader and input of pixel shader, must be the same in both shaders

{% endif %}
void main()
{
  {% if Kernel.GPHasVaryings %}
  {{Kernel.GPVaryings}} varyings;
  gl_Position = {{Kernel.GPFuncName}}(uint(gl_VertexIndex){% for UserArg in Kernel.UserArgs %}, kgenArgs.{{UserArg.Name}}{% endfor %}, varyings);
  gpVaryings  = varyings;
  {% else %}
  gl_Position = {{Kernel.GPFuncName}}(uint(gl_VertexIndex){% for UserArg in Kernel.UserArgs %}, kgenArgs.{{UserArg.Name}}{% endfor %});
  {% endif %}
}
{% else %}
{% if Kernel.GPHasVaryings %}
layout(location = 0) in {{Kernel.GPVaryings}} gpVaryings; // output of vertex shader and input of pixel shader, must be the same in both shaders
{% endif %}

{% for Out in Kernel.GPOutputs %}
layout(location = {{loop.index}}) out {% if Out.IsUint %}uint {% else %}vec4 {% endif %}gpOut_{{Out.Name}};
{% endfor %}

uint GPPackRGBA8(vec4 a_color) // RGBA8 targets (Image2D<uint32_t>) are R32_UINT images, R is in the lowest byte as on CPU
{
  const uvec4 c = uvec4(clamp(a_color, 0.0f, 1.0f)*255.0f + 0.5f);
  return c.x | (c.y << 8) | (c.z << 16) | (c.w << 24);
}

void main()
{
  {% if Kernel.GPOutputIsStruct %}
  const {{Kernel.GPOutputType}} psOut = {{Kernel.GPFuncName}}({% if Kernel.GPHasVaryings %}gpVaryings{% endif %});
  {% for Out in Kernel.GPOutputs %}
  gpOut_{{Out.Name}} = {% if Out.IsUint %}GPPackRGBA8(psOut.{{Out.Name}}){% else %}psOut.{{Out.Name}}{% endif %};
  {% endfor %}
  {% else %}
  gpOut_color0 = {% if Kernel.GPOutputs.0.IsUint %}GPPackRGBA8({{Kernel.GPFuncName}}({% if Kernel.GPHasVaryings %}gpVaryings{% endif %})){% else %}{{Kernel.GPFuncName}}({% if Kernel.GPHasVaryings %}gpVaryings{% endif %}){% endif %};
  {% endif %}
}
{% endif %}
