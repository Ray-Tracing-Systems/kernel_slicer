  // INIT SHARED MEMORY DATA:
  //
  {
    {% if Kernel.threadDim == 1 %}
    const uint localId = a_localTID[0];
    {% else %}
    const uint localId = a_localTID[0] + uint({{Kernel.WGSizeX}})*a_localTID[1]; 
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
