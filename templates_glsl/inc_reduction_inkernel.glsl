  {
    {% if Kernel.threadDim == 1 %}
    const uint localId = gl_LocalInvocationID[0];
    {% else %}
    const uint localId = gl_LocalInvocationID[0] + uint({{Kernel.WGSizeX}})*gl_LocalInvocationID[1]; 
    {% endif %}
    barrier();
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
    barrier();
    {% endfor %}                                      {# /* end of common reduction via shared memory */ #}
    {% if Kernel.UseSubGroups  %}                     {# /* begin put subgroup */ #}
    if(localId < {{Kernel.WarpSize}})
    {
      {% for redvar in Kernel.SubjToRed %}
      {% if redvar.BinFuncForm %}
      {{redvar.Name}}Shared[0] = {{redvar.SubgroupOp}}({{redvar.Op2}}({{redvar.Name}}Shared[localId], {{redvar.Name}}Shared[localId + {{Kernel.WarpSize}}]) );
      {% else %}
      {{redvar.Name}}Shared[0] = {{redvar.SubgroupOp}}({{redvar.Name}}Shared[localId] {{redvar.Op2}} {{redvar.Name}}Shared[localId + {{Kernel.WarpSize}}] );
      {% endif %}
      {% endfor %}                                      {# /* end put subgroup here */ #}
      {% for redvar in Kernel.ArrsToRed %}              {# /* begin put subgroup */ #}
      {% for index in range(redvar.ArraySize) %}        {# /* begin put subgroup */ #}
      {% if redvar.BinFuncForm %}
      {{redvar.Name}}Shared[{{loop.index}}][0] = {{redvar.SubgroupOp}}( {{redvar.Op}}({{redvar.Name}}Shared[{{loop.index}}][localId], {{redvar.Name}}Shared[{{loop.index}}][localId + {{offset}}]) );
      {% else %}
      {{redvar.Name}}Shared[{{loop.index}}][0] = {{redvar.SubgroupOp}}( {{redvar.Name}}Shared[{{loop.index}}][localId] {{redvar.Op}} {{redvar.Name}}Shared[{{loop.index}}][localId + {{offset}}] );
      {% endif %}
      {% endfor %}                                      {# /* end put subgroup here */ #}
      {% endfor %}                                      {# /* end put subgroup here */ #}
    }
    {% else %}
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
    barrier();
    {% if Kernel.threadDim > 1 %}
    barrier();
    {% endif %}
    {% endfor %}  {# /* for offset in Kernel.RedLoop2 */ #}
    {% endif %} {# /* if Kernel.UseSubGroups */ #}
    if(localId == 0)
    {
      {% if Kernel.threadDim == 1 %}
      const uint offset = gl_WorkGroupID[0];
      {% else %}
      const uint offset = gl_WorkGroupID[0] + gl_NumWorkGroups[0]*gl_WorkGroupID[1];
      {% endif %}
      {% for redvar in Kernel.SubjToRed %}
      {% if redvar.SupportAtomic %}
      {{redvar.AtomicOp}}(ubo.{{redvar.Name}}, {{redvar.Name}}Shared[0]);
      {% else %}
      {{ redvar.OutTempName }}[offset] = {{redvar.Name}}Shared[0]; // finish reduction in subsequent kernel passes
      {% endif %}
      {% endfor %}
      {% for redvar in Kernel.ArrsToRed %}
      {% for outName in redvar.OutTempNameA %}
      {% if redvar.SupportAtomic %}
      {{redvar.AtomicOp}}(ubo.{{redvar.Name}}[{{loop.index}}], {{redvar.Name}}Shared[{{loop.index}}][0]);
      {% else %}
      {{ outName }}[offset] = {{redvar.Name}}Shared[{{loop.index}}][0]; // finish reduction in subsequent kernel passes
      {% endif %}
      {% endfor %}
      {% endfor %}
    }
  }