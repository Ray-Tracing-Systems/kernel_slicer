  {% for Arg in Kernel.Args %}
  {% if not Arg.IsUBO %} 
  {% if Arg.IsImage and existsIn(Arg, "ElemISPC") %}
  uniform {{Arg.ElemISPC}} a_{{Arg.Name}}[], uniform const uint32 a_{{Arg.Name}}_info[4],
  {% else if Arg.IsAccelStruct %}
  void* uniform a_{{Arg.Name}}_scene, uniform const uint a_{{Arg.Name}}_geomIdByInst[],
  {% else if Arg.IsPointer %}
  uniform {{Arg.Type}} a_{{Arg.Name}}[], 
  {% else %}
  uniform {{Arg.Type}} a_{{Arg.Name}},
  {% endif %}
  {% endif %}
  {% endfor %}
  {% for Sampler in Kernel.SamplersISPC %}
  uniform const uint32 a_{{Sampler}}[4],
  {% endfor %}
