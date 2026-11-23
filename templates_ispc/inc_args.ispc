  {% for Arg in Kernel.Args %}
  {% if not Arg.IsUBO and not Arg.SkipISPC %} 
  {% if Arg.IsImage and existsIn(Arg, "ElemISPC") %}
  uniform {{Arg.ElemISPC}} {{Arg.Name}}[],
  {% else if Arg.IsAccelStruct %}
  void* uniform {{Arg.Name}}_scene, uniform const uint {{Arg.Name}}_geomIdByInst[],
  {% else if Arg.IsPointer %}
  uniform {{Arg.Type}} {{Arg.Name}}[], 
  {% else %}
  uniform {{Arg.Type}} {{Arg.Name}},
  {% endif %}
  {% endif %}
  {% endfor %}
