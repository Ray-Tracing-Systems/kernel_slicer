{% if Kernel.Hierarchy.IndirectDispatch %}
{% for Impl in Kernel.Hierarchy.Implementations %}

{% if not UseSpecConstWgSize %}
__attribute__((reqd_work_group_size({{Kernel.WGSizeX}}, {{Kernel.WGSizeY}}, {{Kernel.WGSizeZ}})))
{% endif %} 
__kernel void {{Kernel.Name}}_{{Impl.ClassName}}(
## for Arg in Kernel.Args 
  __global {{Arg.Type}} {{Arg.Name}},
## endfor
## for UserArg in Kernel.UserArgs 
  {{UserArg.Type}} {{UserArg.Name}},
## endfor
  __global uint2       * kgen_objPtrData,
  __global unsigned int* kgen_objData,
  __global struct {{MainClassName}}_UBO_Data* ubo,
  const uint {{Kernel.threadIdName1}}, 
  const uint {{Kernel.threadIdName2}},
  const uint {{Kernel.threadIdName3}},
  const uint kgen_tFlagsMask)
{
  ///////////////////////////////////////////////////////////////// prolog
  const uint tid = get_global_id(0); 
  if(tid >= ubo->objNum_{{Kernel.Hierarchy.Name}}Src[{{loop.index}}])
    return;

  const uint realThreadId   = ubo->objNum_{{Kernel.Hierarchy.Name}}Off[{{loop.index}}] + tid;
  const uint2 kgen_objPtr   = kgen_objPtrData[realThreadId];
  const uint kgen_objTag    = (kgen_objPtr.x & {{Kernel.Hierarchy.Name}}_TAG_MASK) >> (32 - {{Kernel.Hierarchy.Name}}_TAG_BITS);
  const uint kgen_objOffset = (kgen_objPtr.x & {{Kernel.Hierarchy.Name}}_OFS_MASK);

  __global {% if Kernel.IsConstObj %}const {% endif %} {{Impl.ClassName}}* pSelf = (__global {{Impl.ClassName}}*)(kgen_objData + kgen_objOffset + 2); // '+ 2' due to vptr (assume 64 bit mode)
  {{Impl.ClassName}}_{{Kernel.Name}}(pSelf, kgen_objPtr.y{%for Arg in Kernel.Args %}{% if loop.index == length(Kernel.Args)-1 %}){%else%}, {{Arg.Name}}{% endif %}{% endfor %}; 
}

{% endfor %} {# /* Impl in Kernel.Hierarchy.Implementations */ #}
{% else %}

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
  __global uint2       * kgen_objPtrData,
  __global unsigned int* kgen_objData,
  const uint {{Kernel.threadIdName1}}, 
  const uint {{Kernel.threadIdName2}},
  const uint {{Kernel.threadIdName3}},
  const uint kgen_tFlagsMask)
{
  ///////////////////////////////////////////////////////////////// prolog
  const uint tid = get_global_id(0); 
  {# /*------------------------------------------------------------- BEG. CHECK EXIT COND ------------------------------------------------------------- */ #}
  {% include "inc_exit_cond.cl" %}
  {# /*------------------------------------------------------------- END. CHECK EXIT COND ------------------------------------------------------------- */ #}
  ///////////////////////////////////////////////////////////////// prolog
  
  const uint kgen_objPtr    = kgen_objPtrData[tid].x;
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

{% endif %} {# /* else branch of 'if Kernel.Hierarchy.IndirectDispatch' */ #}
