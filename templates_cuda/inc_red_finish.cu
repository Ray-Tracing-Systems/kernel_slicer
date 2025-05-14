    const uint localId = {% if Kernel.threadDim == 1 %}threadIdx.x{% else %}threadIdx.x + uint({{Kernel.WGSizeX}})*blockIdx.y{% endif %};
    __syncthreads();
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
    __syncthreads();
    {% endfor %}                                      {# /* end of common reduction via shared memory */ #}
    {% if Kernel.UseSubGroups  %}                     {# /* begin put subgroup */ #}
    if(localId < {{Kernel.WarpSize}})
    {
    {% for redvar in Kernel.SubjToRed %}
    {{redvar.SubgroupOp}}({{redvar.Name}}Shared, localId);
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
    __syncthreads();
    {% if Kernel.threadDim > 1 %}
    __syncthreads();
    {% endif %}
    {% endfor %}  {# /* for offset in Kernel.RedLoop2 */ #}
    {% endif %} {# /* if Kernel.UseSubGroups */ #}
    if(localId == 0)
    {
    {% for redvar in Kernel.SubjToRed %}
    {% if redvar.SupportAtomic %}
    {% if redvar.NegLastStep %}
    {{redvar.AtomicOp}}(&ubo.{{redvar.Name}}, -{{redvar.Name}}Shared[0]);
    {% else %}
    {{redvar.AtomicOp}}(&ubo.{{redvar.Name}}, {{redvar.Name}}Shared[0]);
    {% endif %}
    {% else %}
    {{ redvar.OutTempName }}[offset] = {{redvar.Name}}Shared[0]; // finish reduction in subsequent kernel passes
    {% endif %}
    {% endfor %}
    {% for redvar in Kernel.ArrsToRed %}
    {% for outName in redvar.OutTempNameA %}
    {% if redvar.SupportAtomic %}
    {% if redvar.NegLastStep %}
    {{redvar.AtomicOp}}(&ubo.{{redvar.Name}}[{{loop.index}}], -{{redvar.Name}}Shared[{{loop.index}}][0]);
    {% else %}
    {{redvar.AtomicOp}}(&ubo.{{redvar.Name}}[{{loop.index}}], {{redvar.Name}}Shared[{{loop.index}}][0]);
    {% endif %}
    {% else %}
    {{ outName }}[blockIdx.x] = {{redvar.Name}}Shared[{{loop.index}}][0]; // finish reduction in subsequent kernel passes
    {% endif %}
    {% endfor %}
    {% endfor %}
    }
