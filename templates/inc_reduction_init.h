  {% for redvar in Kernel.SubjToRed %} 
  __local {{redvar.Type}} {{redvar.Name}}Shared[{{Kernel.WGSizeX}}*{{Kernel.WGSizeY}}*{{Kernel.WGSizeZ}}]; 
  {% endfor %}
  {% for redvar in Kernel.ArrsToRed %} 
  __local {{redvar.Type}} {{redvar.Name}}Shared[{{redvar.ArraySize}}][{{Kernel.WGSizeX}}*{{Kernel.WGSizeY}}*{{Kernel.WGSizeZ}}]; 
  {% endfor %}
  {
    {% if Kernel.threadDim == 1 %}
    const uint localId = get_local_id(0); 
    {% else %}
    const uint localId = get_local_id(0) + {{Kernel.WGSizeX}}*get_local_id(1); 
    {% endif %}
    {% for redvar in Kernel.SubjToRed %} 
    {{redvar.Name}}Shared[localId] = {{redvar.Init}}; 
    {% endfor %}
    {% for redvar in Kernel.ArrsToRed %} 
    {% for index in range(redvar.ArraySize) %}
    {{redvar.Name}}Shared[{{loop.index}}][localId] = {{redvar.Init}}; 
    {% endfor %}
    {% endfor %}
  }
  SYNCTHREADS; 