
#include "common{{MainClassSuffixLowerCase}}_slang.h"

{% for Arg in Kernel.Args %}
{% if not Arg.IsUBO %} 
{% if Arg.IsImage %}
RWTexture2D<Arg.Type> {{Arg.Name}}; // TODO: this is Ot a combined image sampler (!!!)
{% else if Arg.IsAccelStruct %}
RaytracingAccelerationStructure {{Arg.Name}};
{% else %}
RWStructuredBuffer<{{Arg.Type}}> {{Arg.Name}}; // 
{% endif %} {# /* Arg.IsImage */ #}
{% endif %} {# /* not Arg.IsUBO */ #}
{% endfor %}
RWStructuredBuffer<{{MainClassName}}{{MainClassSuffix}}_UBO_Data> ubo;

{% for Array in Kernel.ThreadLocalArrays %}
{{Array.Type}} {{Array.Name}}[{{Array.Size}}];
{% endfor %}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct KernelArgs
{
  {% for UserArg in Kernel.UserArgs %} 
  {{UserArg.Type}} {{UserArg.Name}};
  {% endfor %}
  {{Kernel.threadSZType1}} {{Kernel.threadSZName1}}; 
  {{Kernel.threadSZType2}} {{Kernel.threadSZName2}}; 
  {{Kernel.threadSZType3}} {{Kernel.threadSZName3}}; 
  uint tFlagsMask;    
};

static const uint KGEN_REDUCTION_LAST_STEP = 16;
{% for redvar in Kernel.SubjToRed %} 
{% if not redvar.SupportAtomic %}
groupshared {{redvar.Type}} {{redvar.Name}}Shared[{{Kernel.WGSizeX}}*{{Kernel.WGSizeY}}*{{Kernel.WGSizeZ}}];
{% endif %} 
{% endfor %}
{% for redvar in Kernel.ArrsToRed %} 
{% if not redvar.SupportAtomic %}
groupshared {{redvar.Type}} {{redvar.Name}}Shared[{{redvar.ArraySize}}][{{Kernel.WGSizeX}}*{{Kernel.WGSizeY}}*{{Kernel.WGSizeZ}}]; 
{% endif %}
{% endfor %}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 a_globalTID: SV_DispatchThreadID, uint3 a_groupID : SV_GroupID, uint3 a_localTID : SV_GroupThreadID, uniform KernelArgs kgenArgs)
{
  const uint globalId = a_globalTID[0];
  const uint localId  = a_localTID[0];
  {% for redvar in Kernel.SubjToRed %}
  {% if not redvar.SupportAtomic %}
  {{redvar.Name}}Shared[localId] = (globalId < kgenArgs.{{Kernel.threadSZName1}}) ?  {{ redvar.OutTempName }}[kgenArgs.{{Kernel.threadSZName2}} + globalId]  :  {{redvar.Type}}({{redvar.Init}}); // use kgenArgs.{{Kernel.threadSZName2}} for 'InputOffset'
  {% endif %}
  {% endfor %}
  {% for redvar in Kernel.ArrsToRed %}
  {% for outName in redvar.OutTempNameA %}
  {% if not redvar.SupportAtomic %}
  {{redvar.Name}}Shared[{{loop.index}}][localId] = (globalId < kgenArgs.{{Kernel.threadSZName1}}) ? {{ outName }}[kgenArgs.{{Kernel.threadSZName2}} + globalId] : {{redvar.Type}}({{redvar.Init}}); // use kgenArgs.{{Kernel.threadSZName2}} for 'InputOffset'
  {% endif %}
  {% endfor %}
  {% endfor %}
  GroupMemoryBarrierWithGroupSync();
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
  GroupMemoryBarrierWithGroupSync();
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
  GroupMemoryBarrierWithGroupSync();
  {% endfor %}
  {% endif %}  {# /* else branch of if Kernel.UseSubGroups */ #}
 
  if(localId == 0)
  {
    if((kgenArgs.tFlagsMask & KGEN_REDUCTION_LAST_STEP) != 0)
    {
      {% for redvar in Kernel.SubjToRed %}
      {% if not redvar.SupportAtomic %}
      {% if redvar.NegLastStep %}
      ubo[0].{{redvar.Name}} -= {{redvar.Name}}Shared[0];
      {% else %}
      {% if redvar.BinFuncForm %}
      ubo[0].{{redvar.Name}} = {{redvar.Op}}(ubo[0].{{redvar.Name}}, {{redvar.Name}}Shared[0]);
      {% else %}
      ubo[0].{{redvar.Name}} {{redvar.Op}} {{redvar.Name}}Shared[0];
      {% endif %}
      {% endif %}
      {% endif %}
      {% endfor %}
      {% for redvar in Kernel.ArrsToRed %}
      {% for index in range(redvar.ArraySize) %}
      {% if not redvar.SupportAtomic %}
      {% if redvar.NegLastStep %}
      ubo[0].{{redvar.Name}}[{{loop.index}}] -= {{redvar.Name}}Shared[{{loop.index}}][0];
      {% else %}
      {% if redvar.BinFuncForm %}
      ubo[0].{{redvar.Name}}[{{loop.index}}] = {{redvar.Op}}(ubo[0].{{redvar.Name}}[{{loop.index}}], {{redvar.Name}}Shared[{{loop.index}}][0]);
      {% else %}
      ubo[0].{{redvar.Name}}[{{loop.index}}] {{redvar.Op}} {{redvar.Name}}Shared[{{loop.index}}][0];
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
      {{ redvar.OutTempName }}[kgenArgs.{{Kernel.threadSZName3}} + a_groupID[0]] = {{redvar.Name}}Shared[0];     // use kgenArgs.{{Kernel.threadSZName3}} for 'OutputOffset'
      {% endif %}
      {% endfor %}
      {% for redvar in Kernel.ArrsToRed %}
      {% for outName in redvar.OutTempNameA %}
      {% if not redvar.SupportAtomic %}
      {{ outName }}[kgenArgs.{{Kernel.threadSZName3}} + a_groupID[0]] = {{redvar.Name}}Shared[{{loop.index}}][0]; // use kgenArgs.{{Kernel.threadSZName3}} for 'OutputOffset'
      {% endif %}
      {% endfor %}
      {% endfor %}
    }
  }
}
