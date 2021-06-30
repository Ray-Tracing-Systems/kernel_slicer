  {
    {% if Kernel.threadDim == 1 %}
    const uint localId = gl_LocalInvocationID[0];
    {% else %}
    const uint localId = gl_LocalInvocationID[0] + uint({{Kernel.WGSizeX}})*gl_LocalInvocationID[1]; 
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
  // barrier(); we don't need it actually on init
  