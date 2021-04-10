{% if not UseSpecConstWgSize %}
__attribute__((reqd_work_group_size({{Kernel.WGSizeX}}, {{Kernel.WGSizeY}}, {{Kernel.WGSizeZ}})))
{% endif %} 
__kernel void {{Kernel.Name}}(
## for Arg in Kernel.Args 
  __global {{Arg.Type}} {{Arg.Name}},
## endfor
## for UserArg in Kernel.UserArgs 
  {{UserArg.Type}} {{UserArg.Name}},
## endfor
  __global unsigned int* kgen_objPtrData,
  __global unsigned int* kgen_objData,
  const uint {{Kernel.threadIdName1}}, 
  const uint {{Kernel.threadIdName2}},
  const uint {{Kernel.threadIdName3}},
  const uint kgen_tFlagsMask)
{
  ///////////////////////////////////////////////////////////////// prolog
  {% for name in Kernel.threadNames %}
  const uint {{name}} = get_global_id({{ loop.index }}); 
  {% endfor %}
  {# /*------------------------------------------------------------- BEG. CHECK EXIT COND ------------------------------------------------------------- */ #}
  {% include "inc_exit_cond.cl" %}
  {# /*------------------------------------------------------------- END. CHECK EXIT COND ------------------------------------------------------------- */ #}
  ///////////////////////////////////////////////////////////////// prolog
  
  const uint kgen_objPtr    = kgen_objPtrData[get_global_id(0)];
  const uint kgen_objTag    = (kgen_objPtr & {{Kernel.Hierarchy.Name}}_TAG_MASK) >> (32 - {{Kernel.Hierarchy.Name}}_TAG_BITS);
  const uint kgen_objOffset = (kgen_objPtr & {{Kernel.Hierarchy.Name}}_OFS_MASK);

  switch(kgen_objTag)
  {
  {% for Impl in Kernel.Hierarchy.Implementations %}
    case {{Kernel.Hierarchy.Name}}_{{Impl.TagName}}: // implementation for {{Impl.ClassName}}
    ((__global {{Impl.ClassName}}*)(kgen_objData + kgen_objOffset))->{{Kernel.Name}}(get_global_id(0){%for Arg in Kernel.Args %}{% if loop.index == length(Kernel.Args)-1 %}){%else%}, {{Arg.Name}}{% endif %}{% endfor %};
    break;
  {% endfor %}
  default:
  break;
  };
}



