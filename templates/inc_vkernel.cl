{% if not UseSpecConstWgSize %}
__attribute__((reqd_work_group_size({{Kernel.WGSizeX}}, {{Kernel.WGSizeY}}, {{Kernel.WGSizeZ}})))
{% endif %} 
__kernel void {{Kernel.Name}}({% include "inc_args.cl" %})
{
  ///////////////////////////////////////////////////////////////// prolog
  const uint tid = get_global_id(0); 
  {# /*------------------------------------------------------------- BEG. CHECK EXIT COND ------------------------------------------------------------- */ #}
  {% include "inc_exit_cond.cl" %}
  {# /*------------------------------------------------------------- END. CHECK EXIT COND ------------------------------------------------------------- */ #}
  ///////////////////////////////////////////////////////////////// prolog
  
  const uint kgen_objPtr    = kgen_objPtrData[tid];
  const uint kgen_objTag    = (kgen_objPtr & {{Kernel.Hierarchy.Name}}_TAG_MASK) >> (32 - {{Kernel.Hierarchy.Name}}_TAG_BITS);
  const uint kgen_objOffset = (kgen_objPtr & {{Kernel.Hierarchy.Name}}_OFS_MASK);

  switch(kgen_objTag)
  {
  {% for Impl in Kernel.Hierarchy.Implementations %}
    case {{Kernel.Hierarchy.Name}}_{{Impl.TagName}}: // implementation for {{Impl.ClassName}}
    {
      __global {{Impl.ClassName}}* pSelf = (__global {{Impl.ClassName}}*)(kgen_objData + kgen_objOffset + 2); // '+ 2' due to vptr (assume 64 bit mode)
      {{Impl.ClassName}}_{{Kernel.Name}}(pSelf, tid{%for Arg in Kernel.Args %}{% if loop.index == length(Kernel.Args)-1 %}){%else%}, {{Arg.Name}}{% endif %}{% endfor %};
    }
    break;
  {% endfor %}
  default:
  break;
  };
}



