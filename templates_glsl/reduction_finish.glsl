#version 460
#extension GL_GOOGLE_include_directive : require
{% if Kernel.UseSubGroups %}
#extension GL_KHR_shader_subgroup_arithmetic: enable
{% endif %}

#include "common_generated.h"

## for Arg in Kernel.Args
{% if not Arg.IsUBO %} 
{% if Arg.IsImage %}
layout(binding = {{loop.index}}, set = 0{% if Arg.NeedFmt%}, {{Arg.ImFormat}}{% endif %}) uniform {{Arg.Type}} {{Arg.Name}}; //
{% else %}
layout(binding = {{loop.index}}, set = 0) buffer data{{loop.index}} { {{Arg.Type}} {{Arg.Name}}[]; }; //
{% endif %} {# /* Arg.IsImage */ #}
{% endif %} {# /* not Arg.IsUBO */ #}
## endfor
layout(binding = {{length(Kernel.Args)}}, set = 0) buffer dataUBO { {{MainClassName}}_UBO_Data ubo; };

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout( push_constant ) uniform kernelArgs
{
  {% for UserArg in Kernel.UserArgs %} 
  {{UserArg.Type}} {{UserArg.Name}};
  {% endfor %}
  {{Kernel.threadSZType1}} {{Kernel.threadSZName1}}; 
  {{Kernel.threadSZType2}} {{Kernel.threadSZName2}}; 
  {{Kernel.threadSZType3}} {{Kernel.threadSZName3}}; 
  uint tFlagsMask;    
} kgenArgs;

{% for redvar in Kernel.SubjToRed %} 
{% if not redvar.SupportAtomic %}
shared {{redvar.Type}} {{redvar.Name}}Shared[{{Kernel.WGSizeX}}*{{Kernel.WGSizeY}}*{{Kernel.WGSizeZ}}];
{% endif %} 
{% endfor %}
{% for redvar in Kernel.ArrsToRed %} 
{% if not redvar.SupportAtomic %}
shared {{redvar.Type}} {{redvar.Name}}Shared[{{redvar.ArraySize}}][{{Kernel.WGSizeX}}*{{Kernel.WGSizeY}}*{{Kernel.WGSizeZ}}]; 
{% endif %}
{% endfor %}

void main()
{
  const uint globalId = gl_GlobalInvocationID[0];
  const uint localId  = gl_LocalInvocationID[0];
  {% for redvar in Kernel.SubjToRed %}
  {% if not redvar.SupportAtomic %}
  {{redvar.Name}}Shared[localId] = (globalId < kgenArgs.{{Kernel.threadSZName1}}) ?  {{ redvar.OutTempName }}[kgenArgs.{{Kernel.threadSZName2}} + globalId]  :  {{redvar.Init}}; // use kgenArgs.{{Kernel.threadSZName2}} for 'InputOffset'
  {% endif %}
  {% endfor %}
  {% for redvar in Kernel.ArrsToRed %}
  {% for outName in redvar.OutTempNameA %}
  {% if not redvar.SupportAtomic %}
  {{redvar.Name}}Shared[{{loop.index}}][localId] = (globalId < kgenArgs.{{Kernel.threadSZName1}}) ? {{ outName }}[kgenArgs.{{Kernel.threadSZName2}} + globalId] : {{redvar.Init}}; // use kgenArgs.{{Kernel.threadSZName2}} for 'InputOffset'
  {% endif %}
  {% endfor %}
  {% endfor %}
  barrier();
  {% for offset in Kernel.RedLoop1 %} 
  if (localId < {{offset}}) 
  {
    {% for redvar in Kernel.SubjToRed %}
    {% if not redvar.SupportAtomic %}
    {% if redvar.BinFuncForm %}
    {{redvar.Name}}Shared[localId] = {{redvar.Op}}({{redvar.Name}}Shared[localId], {{redvar.Name}}Shared[localId + {{offset}}]);
    {% else %}
    {{redvar.Name}}Shared[localId] {{redvar.Op}} {{redvar.Name}}Shared[localId + {{offset}}];
    {% endif %}
    {% endif %}
    {% endfor %}
    {% for redvar in Kernel.ArrsToRed %}
    {% for index in range(redvar.ArraySize) %}
    {% if not redvar.SupportAtomic %}
    {% if redvar.BinFuncForm %}
    {{redvar.Name}}Shared[{{loop.index}}][localId] = {{redvar.Op}}({{redvar.Name}}Shared[{{loop.index}}][localId], {{redvar.Name}}Shared[{{loop.index}}][localId + {{offset}}]);
    {% else %}
    {{redvar.Name}}Shared[{{loop.index}}][localId] {{redvar.Op}} {{redvar.Name}}Shared[{{loop.index}}][localId + {{offset}}];
    {% endif %}
    {% endif %}
    {% endfor %}
    {% endfor %}
  }
  barrier();
  {% endfor %}
  {% if Kernel.UseSubGroups  %}                      {# /* begin put subgroup Kernel.UseSubGroups */ #}
  if(localId < {{Kernel.WarpSize}})
  {
    {% for redvar in Kernel.SubjToRed %}
    {% if not redvar.SupportAtomic %}
    {% if redvar.BinFuncForm %}
    {{redvar.Name}}Shared[0] = {{redvar.SubgroupOp}}({{redvar.Op2}}({{redvar.Name}}Shared[localId], {{redvar.Name}}Shared[localId + {{Kernel.WarpSize}}]) );
    {% else %}
    {{redvar.Name}}Shared[0] = {{redvar.SubgroupOp}}({{redvar.Name}}Shared[localId] {{redvar.Op2}} {{redvar.Name}}Shared[localId + {{Kernel.WarpSize}}] );
    {% endif %}
    {% endif %}
    {% endfor %}                                      {# /* end put subgroup here */ #}
    {% for redvar in Kernel.ArrsToRed %}              {# /* begin put subgroup */ #}
    {% if not redvar.SupportAtomic %}
    {% for index in range(redvar.ArraySize) %}        {# /* begin put subgroup */ #}
    {% if redvar.BinFuncForm %}
    {{redvar.Name}}Shared[{{loop.index}}][0] = {{redvar.SubgroupOp}}( {{redvar.Op}}({{redvar.Name}}Shared[{{loop.index}}][localId], {{redvar.Name}}Shared[{{loop.index}}][localId + {{offset}}]) );
    {% else %}
    {{redvar.Name}}Shared[{{loop.index}}][0] = {{redvar.SubgroupOp}}( {{redvar.Name}}Shared[{{loop.index}}][localId] {{redvar.Op}} {{redvar.Name}}Shared[{{loop.index}}][localId + {{offset}}] );
    {% endif %}
    {% endfor %}                                      {# /* end put subgroup here */ #}
    {% endif %}
    {% endfor %}                                      {# /* end put subgroup here */ #}
  }
  {% else %}
  {% for offset in Kernel.RedLoop2 %} 
  if (localId < {{offset}}) 
  {
    {% for redvar in Kernel.SubjToRed %}
    {% if not redvar.SupportAtomic %}
    {% if redvar.BinFuncForm %}
    {{redvar.Name}}Shared[localId] = {{redvar.Op}}({{redvar.Name}}Shared[localId], {{redvar.Name}}Shared[localId + {{offset}}]);
    {% else %}
    {{redvar.Name}}Shared[localId] {{redvar.Op}} {{redvar.Name}}Shared[localId + {{offset}}];
    {% endif %}
    {% endif %}
    {% endfor %}
    {% for redvar in Kernel.ArrsToRed %}
    {% for index in range(redvar.ArraySize) %}
    {% if not redvar.SupportAtomic %}
    {% if redvar.BinFuncForm %}
    {{redvar.Name}}Shared[{{loop.index}}][localId] = {{redvar.Op}}({{redvar.Name}}Shared[{{loop.index}}][localId], {{redvar.Name}}Shared[{{loop.index}}][localId + {{offset}}]);
    {% else %}
    {{redvar.Name}}Shared[{{loop.index}}][localId] {{redvar.Op}} {{redvar.Name}}Shared[{{loop.index}}][localId + {{offset}}];
    {% endif %}
    {% endif %}
    {% endfor %}
    {% endfor %}
  }
  barrier();
  {% endfor %}
  {% endif %}  {# /* else branch of if Kernel.UseSubGroups */ #}
 
  if(localId == 0)
  {
    if((kgenArgs.tFlagsMask & KGEN_REDUCTION_LAST_STEP) != 0)
    {
      {% for redvar in Kernel.SubjToRed %}
      {% if not redvar.SupportAtomic %}
      {% if redvar.NegLastStep %}
      ubo.{{redvar.Name}} -= {{redvar.Name}}Shared[0];
      {% else %}
      {% if redvar.BinFuncForm %}
      ubo.{{redvar.Name}} = {{redvar.Op}}(ubo.{{redvar.Name}}, {{redvar.Name}}Shared[0]);
      {% else %}
      ubo.{{redvar.Name}} {{redvar.Op}} {{redvar.Name}}Shared[0];
      {% endif %}
      {% endif %}
      {% endif %}
      {% endfor %}
      {% for redvar in Kernel.ArrsToRed %}
      {% for index in range(redvar.ArraySize) %}
      {% if not redvar.SupportAtomic %}
      {% if redvar.NegLastStep %}
      ubo.{{redvar.Name}}[{{loop.index}}] -= {{redvar.Name}}Shared[{{loop.index}}][0];
      {% else %}
      {% if redvar.BinFuncForm %}
      ubo.{{redvar.Name}}[{{loop.index}}] = {{redvar.Op}}(ubo.{{redvar.Name}}[{{loop.index}}], {{redvar.Name}}Shared[{{loop.index}}][0]);
      {% else %}
      ubo.{{redvar.Name}}[{{loop.index}}] {{redvar.Op}} {{redvar.Name}}Shared[{{loop.index}}][0];
      {% endif %}
      {% endif %}
      {% endif %}
      {% endfor %}
      {% endfor %}
    }
    else
    {
      {% for redvar in Kernel.SubjToRed %}
      {% if not redvar.SupportAtomic %}
      {{ redvar.OutTempName }}[kgenArgs.{{Kernel.threadSZName3}} + gl_WorkGroupID[0]] = {{redvar.Name}}Shared[0];     // use kgenArgs.{{Kernel.threadSZName3}} for 'OutputOffset'
      {% endif %}
      {% endfor %}
      {% for redvar in Kernel.ArrsToRed %}
      {% for outName in redvar.OutTempNameA %}
      {% if not redvar.SupportAtomic %}
      {{ outName }}[kgenArgs.{{Kernel.threadSZName3}} + gl_WorkGroupID[0]] = {{redvar.Name}}Shared[{{loop.index}}][0]; // use kgenArgs.{{Kernel.threadSZName3}} for 'OutputOffset'
      {% endif %}
      {% endfor %}
      {% endfor %}
    }
  }
}
