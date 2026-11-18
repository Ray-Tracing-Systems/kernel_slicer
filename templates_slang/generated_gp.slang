//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Graphics pipeline, {% if Kernel.GPStage == 1 %}vertex shader '{% else %}pixel shader '{% endif %}{{Kernel.Name}}'. Bindings of this shader are in descriptor set {{Kernel.DSetId}}.
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

{% include "common_generated.slang" %}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct GPVertexOut // output of vertex shader and input of pixel shader, must be the same in both shaders
{
  float4 pos : SV_Position;
  {% if Kernel.GPHasVaryings %}
  {{Kernel.GPVaryings}} v : VARYINGS;
  {% endif %}
};

{% if Kernel.GPStage == 1 %}
struct KernelArgs
{
  {% for UserArg in Kernel.UserArgs %}
  {{UserArg.Type}} {{UserArg.Name}};
  {% endfor %}
  uint gpDummy;
};

[shader("vertex")]
GPVertexOut main(uint a_vertexId : SV_VertexID, {% if Kernel.GPInstanced %}uint a_instanceId : SV_InstanceID, {% endif %}uniform KernelArgs kgenArgs)
{
  GPVertexOut res;
  res.pos = {{Kernel.GPFuncName}}(a_vertexId{% if Kernel.GPInstanced %}, a_instanceId{% endif %}{% for UserArg in Kernel.UserArgs %}, kgenArgs.{{UserArg.Name}}{% endfor %}{% if Kernel.GPHasVaryings %}, res.v{% endif %});
  return res;
}
{% else %}
struct GPPixelOut
{
  {% for Out in Kernel.GPOutputs %}
  {{Out.Type}} {{Out.Name}} : SV_Target{{loop.index}};
  {% endfor %}
};

uint GPPackRGBA8(float4 a_color) // RGBA8 targets (Image2D<uint32_t>) are R32_UINT images, R is in the lowest byte as on CPU
{
  const uint4 c = uint4(clamp(a_color, 0.0f, 1.0f)*255.0f + 0.5f);
  return c.x | (c.y << 8) | (c.z << 16) | (c.w << 24);
}

[shader("fragment")]
GPPixelOut main(GPVertexOut a_in)
{
  GPPixelOut res;
  {% if Kernel.GPOutputIsStruct %}
  const {{Kernel.GPOutputType}} psOut = {{Kernel.GPFuncName}}(a_in.v);
  {% for Out in Kernel.GPOutputs %}
  res.{{Out.Name}} = {% if Out.PackRGBA8 %}GPPackRGBA8(psOut.{{Out.Name}}){% else %}psOut.{{Out.Name}}{% endif %};
  {% endfor %}
  {% else %}
  res.color0 = {% if Kernel.GPOutputs.0.PackRGBA8 %}GPPackRGBA8({{Kernel.GPFuncName}}(a_in.v)){% else %}{{Kernel.GPFuncName}}(a_in.v){% endif %};
  {% endif %}
  return res;
}
{% endif %}
