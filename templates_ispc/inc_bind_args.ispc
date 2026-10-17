  {% for Arg in Kernel.Args %}
  {% if not Arg.IsUBO %} 
  {% if Arg.IsAccelStruct %}
  void* uniform a_{{Arg.Name}}_scene, uniform const uint a_{{Arg.Name}}_geomIdByInst[],
  {% else if Arg.IsPointer %}
  uniform {{Arg.Type}} a_{{Arg.Name}}[], 
  {% else %}
  uniform {{Arg.Type}} a_{{Arg.Name}},
  {% endif %}
  {% endif %}
  {% endfor %}
