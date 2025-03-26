    // INIT SHARED MEMORY DATA:
    //
    {% if not Kernel.InitKPass %}
    {% for redvar in Kernel.SubjToRed %}
    __shared__ {{redvar.Type}} {{redvar.Name}}Shared[{{Kernel.WGSizeX}}*{{Kernel.WGSizeY}}*{{Kernel.WGSizeZ}}]; 
    {% endfor %}
    {% for redvar in Kernel.ArrsToRed %}
    __shared__ {{redvar.Type}} {{redvar.Name}}Shared[{{redvar.ArraySize}}][{{Kernel.WGSizeX}}*{{Kernel.WGSizeY}}*{{Kernel.WGSizeZ}}]; 
    {% endfor %}
    {% endif %}
    {% if Kernel.EnableBlockExpansion %}
    {% for TID in Kernel.ThreadSizeBE %}
    const {{TID.Type}} {{TID.Name}} = {{TID.Value}}; 
    {% endfor %}
    {% for Var in Kernel.SharedBE %}
    __shared__ {{Var}}
    {% endfor %}
    {% endif %}
    {% for redvar in Kernel.SubjToRed %} 
    {{redvar.Name}}Shared[{% if Kernel.threadDim == 1 %}threadIdx.x{% else %}threadIdx.x + uint({{Kernel.WGSizeX}})*blockIdx.y{% endif %}] = {{redvar.Init}}; 
    {% endfor %}
    {% for redvar in Kernel.ArrsToRed %} 
    {% for index in range(redvar.ArraySize) %}
    {{redvar.Name}}Shared[{{loop.index}}][{% if Kernel.threadDim == 1 %}threadIdx.x{% else %}threadIdx.x + uint({{Kernel.WGSizeX}})*blockIdx.y{% endif %}] = {{redvar.Init}}; 
    {% endfor %}
    {% endfor %}
    
