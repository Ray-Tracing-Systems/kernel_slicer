{% if not UseSpecConstWgSize %}
__attribute__((reqd_work_group_size(256, 1, 1)))
{% endif %}
__kernel void {{Kernel.Name}}_Reduction({% include "inc_args.cl" %})
{
  const uint globalId = get_global_id(0);
  const uint localId  = get_local_id(0);

  {% for redvar in Kernel.SubjToRed %}
  {% if not redvar.SupportAtomic %}
  __local {{redvar.Type}} {{redvar.Name}}Shared[256]; 
  {{redvar.Name}}Shared[localId] = (globalId < {{Kernel.threadSZName1}}) ?  {{ redvar.OutTempName }}[{{Kernel.threadSZName2}} + globalId]  :  {{redvar.Init}}; // use {{Kernel.threadSZName2}} for 'InputOffset'
  {% endif %}
  {% endfor %}
  {% for redvar in Kernel.ArrsToRed %}
  __local {{redvar.Type}} {{redvar.Name}}Shared[{{redvar.ArraySize}}][256]; 
  {% for outName in redvar.OutTempNameA %}
  {% if not redvar.SupportAtomic %}
  {{redvar.Name}}Shared[{{loop.index}}][localId] = (globalId < {{Kernel.threadSZName1}}) ? {{ outName }}[{{Kernel.threadSZName2}} + globalId] : {{redvar.Init}}; // use {{Kernel.threadSZName2}} for 'InputOffset'
  {% endif %}
  {% endfor %}
  {% endfor %}
  barrier(CLK_LOCAL_MEM_FENCE);
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
  barrier(CLK_LOCAL_MEM_FENCE);
  {% endfor %}
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
  {% endfor %}
  if(localId == 0)
  {
    if((kgen_tFlagsMask & KGEN_REDUCTION_LAST_STEP) != 0)
    {
      {% for redvar in Kernel.SubjToRed %}
      {% if not redvar.SupportAtomic %}
      {% if redvar.NegLastStep %}
      ubo->{{redvar.Name}} -= {{redvar.Name}}Shared[0];
      {% else %}
      {% if redvar.BinFuncForm %}
      ubo->{{redvar.Name}} = {{redvar.Op}}(ubo->{{redvar.Name}}, {{redvar.Name}}Shared[0]);
      {% else %}
      ubo->{{redvar.Name}} {{redvar.Op}} {{redvar.Name}}Shared[0];
      {% endif %}
      {% endif %}
      {% endif %}
      {% endfor %}
      {% for redvar in Kernel.ArrsToRed %}
      {% for index in range(redvar.ArraySize) %}
      {% if not redvar.SupportAtomic %}
      {% if redvar.NegLastStep %}
      ubo->{{redvar.Name}}[{{loop.index}}] -= {{redvar.Name}}Shared[{{loop.index}}][0];
      {% else %}
      {% if redvar.BinFuncForm %}
      ubo->{{redvar.Name}}[{{loop.index}}] = {{redvar.Op}}(ubo->{{redvar.Name}}[{{loop.index}}], {{redvar.Name}}Shared[{{loop.index}}][0]);
      {% else %}
      ubo->{{redvar.Name}}[{{loop.index}}] {{redvar.Op}} {{redvar.Name}}Shared[{{loop.index}}][0];
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
      {{ redvar.OutTempName }}[{{Kernel.threadSZName3}} + get_group_id(0)] = {{redvar.Name}}Shared[0]; // use {{Kernel.threadSZName3}} for 'OutputOffset'
      {% endif %}
      {% endfor %}
      {% for redvar in Kernel.ArrsToRed %}
      {% for outName in redvar.OutTempNameA %}
      {% if not redvar.SupportAtomic %}
      {{ outName }}[{{Kernel.threadSZName3}} + get_group_id(0)] = {{redvar.Name}}Shared[{{loop.index}}][0]; // use {{Kernel.threadSZName3}} for 'OutputOffset'
      {% endif %}
      {% endfor %}
      {% endfor %}
    }
  }
}