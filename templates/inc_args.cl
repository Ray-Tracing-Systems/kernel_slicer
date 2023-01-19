
## for Arg in Kernel.Args
  {% if not Arg.IsUBO %} 
  __global {{Arg.Type}} {{Arg.Name}},
  {% endif %}
## endfor
## for UserArg in Kernel.UserArgs 
  {{UserArg.Type}} {{UserArg.Name}},
## endfor
  __global struct {{MainClassName}}{{MainClassSuffix}}_UBO_Data* ubo,
  const uint {{Kernel.threadSZName1}}, 
  const uint {{Kernel.threadSZName2}},
  const uint {{Kernel.threadSZName3}},
  const uint kgen_tFlagsMask