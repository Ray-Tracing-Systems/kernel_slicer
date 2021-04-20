{% if not UseSpecConstWgSize %}
__attribute__((reqd_work_group_size({{Kernel.WGSizeX}}, {{Kernel.WGSizeY}}, {{Kernel.WGSizeZ}})))
{% endif %} 
__kernel void {{Kernel.Name}}({% include "inc_args.cl" %})
{
  ///////////////////////////////////////////////////////////////// prolog
  {% for name in Kernel.threadNames %}
  const uint {{name}} = get_global_id({{ loop.index }}); 
  {% endfor %}
  uint kgen_objPtr = 0;
  {# /*------------------------------------------------------------- BEG. CHECK EXIT COND ------------------------------------------------------------- */ #}
  {% include "inc_exit_cond.cl" %}
  {# /*------------------------------------------------------------- END. CHECK EXIT COND ------------------------------------------------------------- */ #}
  {% for Member in Kernel.Members %}
  const {{Member.Type}} {{Member.Name}} = ubo->{{Member.Name}};
  {% endfor %}
  ///////////////////////////////////////////////////////////////// prolog
  {# /*------------------------------------------------------------- KERNEL SOURCE ------------------------------------------------------------- */ #}
  {{Kernel.Source}}
  {# /*------------------------------------------------------------- KERNEL SOURCE ------------------------------------------------------------- */ #}
  //KGEN_EPILOG:
  {% if not Kernel.Hierarchy.IndirectDispatch %}
  kgen_objPtrData[get_global_id(0)] = kgen_objPtr;
  {% else %}
  const uint kgen_objTag    = (kgen_objPtr & {{Kernel.Hierarchy.Name}}_TAG_MASK) >> (32 - {{Kernel.Hierarchy.Name}}_TAG_BITS);
  const uint kgen_objOffset = (kgen_objPtr & {{Kernel.Hierarchy.Name}}_OFS_MASK);
  {% if Kernel.threadDim == 1 %}
  const uint localId        = get_local_id(0); 
  {% else %}
  const uint localId        = get_local_id(0) + {{Kernel.WGSizeX}}*get_local_id(1); 
  {% endif %}
  __local uint objNum[{{Kernel.WGSizeX}}*{{Kernel.WGSizeY}}*{{Kernel.WGSizeZ}}];
  {% for Impl in Kernel.Hierarchy.Implementations %}
  
  //// count objects of {{Impl.ClassName}}
  //
  objNum[localId] = (kgen_objTag == {{Kernel.Hierarchy.Name}}_{{Impl.TagName}}) ? 1 : 0;
  barrier(CLK_LOCAL_MEM_FENCE);
  {% for offset in Kernel.Hierarchy.RedLoop1 %} 
  if (localId < {{offset}}) 
    objNum[localId] += objNum[localId + {{offset}}];
  barrier(CLK_LOCAL_MEM_FENCE);
  {% endfor %}
  {% for offset in Kernel.Hierarchy.RedLoop2 %} 
  if (localId < {{offset}}) 
    objNum[localId] += objNum[localId + {{offset}}];
  {% endfor %}

  if (localId == 0)
    atomic_add(&ubo->objNum_{{Impl.ClassName}}, objNum[0]); 
  {% endfor %}
  {% endif %}
}



