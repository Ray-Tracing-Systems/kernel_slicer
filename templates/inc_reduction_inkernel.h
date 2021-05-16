  {
    {% if Kernel.threadDim == 1 %}
    const uint localId = get_local_id(0); 
    {% else %}
    const uint localId = get_local_id(0) + {{Kernel.WGSizeX}}*get_local_id(1); 
    {% endif %}
    barrier(CLK_LOCAL_MEM_FENCE);
    {% for offset in Kernel.RedLoop1 %} 
    if (localId < {{offset}}) 
    {
      {% for redvar in Kernel.SubjToRed %}
      {% if redvar.BinFuncForm %}
      {{redvar.Name}}Shared[localId] = {{redvar.Op}}({{redvar.Name}}Shared[localId], {{redvar.Name}}Shared[localId + {{offset}}]);
      {% else %}
      {{redvar.Name}}Shared[localId] {{redvar.Op}} {{redvar.Name}}Shared[localId + {{offset}}];
      {% endif %}
      {% endfor %}
      {% for redvar in Kernel.ArrsToRed %}
      {% for index in range(redvar.ArraySize) %}
      {% if redvar.BinFuncForm %}
      {{redvar.Name}}Shared[{{loop.index}}][localId] = {{redvar.Op}}({{redvar.Name}}Shared[{{loop.index}}][localId], {{redvar.Name}}Shared[{{loop.index}}][localId + {{offset}}]);
      {% else %}
      {{redvar.Name}}Shared[{{loop.index}}][localId] {{redvar.Op}} {{redvar.Name}}Shared[{{loop.index}}][localId + {{offset}}];
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
      {% if redvar.BinFuncForm %}
      {{redvar.Name}}Shared[localId] = {{redvar.Op}}({{redvar.Name}}Shared[localId], {{redvar.Name}}Shared[localId + {{offset}}]);
      {% else %}
      {{redvar.Name}}Shared[localId] {{redvar.Op}} {{redvar.Name}}Shared[localId + {{offset}}];
      {% endif %}
      {% endfor %}
      {% for redvar in Kernel.ArrsToRed %}
      {% for index in range(redvar.ArraySize) %}
      {% if redvar.BinFuncForm %}
      {{redvar.Name}}Shared[{{loop.index}}][localId] = {{redvar.Op}}({{redvar.Name}}Shared[{{loop.index}}][localId], {{redvar.Name}}Shared[{{loop.index}}][localId + {{offset}}]);
      {% else %}
      {{redvar.Name}}Shared[{{loop.index}}][localId] {{redvar.Op}} {{redvar.Name}}Shared[{{loop.index}}][localId + {{offset}}];
      {% endif %}
      {% endfor %}
      {% endfor %}
    }
    {% if Kernel.threadDim > 1 %}
    barrier(CLK_LOCAL_MEM_FENCE);
    {% endif %}
    {% endfor %}
    if(localId == 0)
    {
      {% if Kernel.threadDim == 1 %}
      const uint offset = get_group_id(0);
      {% else %}
      const uint offset = get_group_id(0) + get_num_groups(0)*get_group_id(1);
      {% endif %}
      {% for redvar in Kernel.SubjToRed %}
      {% if redvar.SupportAtomic %}
      {{redvar.AtomicOp}}(&ubo->{{redvar.Name}}, {{redvar.Name}}Shared[0]);
      {% else %}
      {{ redvar.OutTempName }}[offset] = {{redvar.Name}}Shared[0]; // finish reduction in subsequent kernel passes
      {% endif %}
      {% endfor %}
      {% for redvar in Kernel.ArrsToRed %}
      {% for outName in redvar.OutTempNameA %}
      {% if redvar.SupportAtomic %}
      {{redvar.AtomicOp}}(&(ubo->{{redvar.Name}}[{{loop.index}}]), {{redvar.Name}}Shared[{{loop.index}}][0]);
      {% else %}
      {{ outName }}[offset] = {{redvar.Name}}Shared[{{loop.index}}][0]; // finish reduction in subsequent kernel passes
      {% endif %}
      {% endfor %}
      {% endfor %}
    }
  }