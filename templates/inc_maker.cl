{% if Kernel.Hierarchy.IndirectDispatch %}
{% if not UseSpecConstWgSize %}
__attribute__((reqd_work_group_size(32, 1, 1)))
{% endif %} 
__kernel void {{Kernel.Name}}_ZeroObjCounters({% include "inc_args.cl" %})
{ 
  const uint lid = get_local_id(0); 
  if(lid < {{length(Kernel.Hierarchy.Implementations)}})
    ubo->objNum_{{Kernel.Hierarchy.Name}}Src[lid] = 0;
}

{% endif %}
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
  ///////////////////////////////////////////////////////////////// prolog
  {# /*------------------------------------------------------------- KERNEL SOURCE ------------------------------------------------------------- */ #}
  {{Kernel.Source}}
  {# /*------------------------------------------------------------- KERNEL SOURCE ------------------------------------------------------------- */ #}
  //KGEN_EPILOG:
  {% if not Kernel.Hierarchy.IndirectDispatch %}
  kgen_objPtrData[get_global_id(0)] = make_uint2(kgen_objPtr, get_global_id(0)); // put old threadId instead of zero
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
  barrier(CLK_LOCAL_MEM_FENCE);  
  {% endfor %}
  barrier(CLK_LOCAL_MEM_FENCE);
  if (localId == 0)
    atomic_add(&ubo->objNum_{{Kernel.Hierarchy.Name}}Src[{{loop.index}}], objNum[0]); // {{Impl.ClassName}}
  
  //if(kgen_objTag == {{Kernel.Hierarchy.Name}}_{{Impl.TagName}})
  //  atomic_add(&ubo->objNum_{{Kernel.Hierarchy.Name}}Src[{{loop.index}}], 1);

  {% endfor %} {# /* Impl in Kernel.Hierarchy.Implementations */ #}
  {% endif %}  {# /* if not Kernel.Hierarchy.IndirectDispatch */ #}
}
{% if Kernel.Hierarchy.IndirectDispatch %}

{% if not UseSpecConstWgSize %}
__attribute__((reqd_work_group_size(32, 1, 1))) 
{% endif %} 
__kernel void {{Kernel.Name}}_CountTypeIntervals({% include "inc_args.cl" %})
{
  const uint lid = get_local_id(0); 
  __local uint objNum[32*2];
  
  uint currObjNum = ubo->objNum_{{Kernel.Hierarchy.Name}}Src[lid];
  uint summResult = 0;
  
  PREFIX_SUMM_MACRO(currObjNum, summResult, objNum, 32);
  objNum[lid] = summResult;
  barrier(CLK_LOCAL_MEM_FENCE);

  if(lid < {{length(Kernel.Hierarchy.Implementations)}}+1)
  {
    ubo->objNum_{{Kernel.Hierarchy.Name}}Acc[lid] = (lid == 0) ? 0 : objNum[lid-1];
    ubo->objNum_{{Kernel.Hierarchy.Name}}Off[lid] = (lid == 0) ? 0 : objNum[lid-1];
  }

  //uint currSize = 0;
  //for(int i=0; i<{{length(Kernel.Hierarchy.Implementations)}}; i++)
  //{
  //  ubo->objNum_{{Kernel.Hierarchy.Name}}Acc[i] = currSize;
  //  currSize += ubo->objNum_{{Kernel.Hierarchy.Name}}Src[i];
  //}
  //ubo->objNum_{{Kernel.Hierarchy.Name}}Acc[{{length(Kernel.Hierarchy.Implementations)}}] = currSize;
}

{% if not UseSpecConstWgSize %}
__attribute__((reqd_work_group_size({{Kernel.WGSizeX}}, {{Kernel.WGSizeY}}, {{Kernel.WGSizeZ}})))
{% endif %} 
__kernel void {{Kernel.Name}}_Sorter({% include "inc_args.cl" %})
{
  ///////////////////////////////////////////////////////////////// prolog
  {% for name in Kernel.threadNames %}
  const uint {{name}} = get_global_id({{ loop.index }}); 
  {% endfor %}
  uint kgen_objPtr = 0;
  ///////////////////////////////////////////////////////////////// prolog
  {# /*------------------------------------------------------------- KERNEL SOURCE ------------------------------------------------------------- */ #}
  {{Kernel.Source}}
  {# /*------------------------------------------------------------- KERNEL SOURCE ------------------------------------------------------------- */ #}
  //KGEN_EPILOG:
  const uint kgen_objTag    = (kgen_objPtr & {{Kernel.Hierarchy.Name}}_TAG_MASK) >> (32 - {{Kernel.Hierarchy.Name}}_TAG_BITS);
  const uint kgen_objOffset = (kgen_objPtr & {{Kernel.Hierarchy.Name}}_OFS_MASK);
  // use parallel prefix summ
  {% if Kernel.threadDim == 1 %}
  const uint localId = get_local_id(0); 
  {% else %}
  const uint localId = get_local_id(0) + {{Kernel.WGSizeX}}*get_local_id(1); 
  {% endif %}
  const uint lastId  = {{Kernel.WGSizeX}}*{{Kernel.WGSizeY}}*{{Kernel.WGSizeZ}}-1;
  __local uint objNum[2*{{Kernel.WGSizeX}}*{{Kernel.WGSizeY}}*{{Kernel.WGSizeZ}}];
  __local uint blockOffset;
  {% for Impl in Kernel.Hierarchy.Implementations %}
  
  //// count offsets for {{Impl.ClassName}}
  //
  {
    uint isThisType  = (kgen_objTag == {{Kernel.Hierarchy.Name}}_{{Impl.TagName}}) ? 1 : 0;
    uint localOffset = 0;
    
    PREFIX_SUMM_MACRO(isThisType, localOffset, objNum, {{Kernel.WGSizeX}}*{{Kernel.WGSizeY}}*{{Kernel.WGSizeZ}});
    objNum[localId] = localOffset;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if(localId == 0)
      blockOffset = atomic_add(&ubo->objNum_{{Kernel.Hierarchy.Name}}Acc[{{loop.index}}], objNum[lastId]);
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if(isThisType == 1)
      kgen_objPtrData[blockOffset + localOffset - 1] = make_uint2(kgen_objPtr, get_global_id(0));

    //if(isThisType == 1)
    //{
    //  uint offset = atomic_add(&ubo->objNum_{{Kernel.Hierarchy.Name}}Tst[{{loop.index}}], 1);
    //  kgen_objPtrData[offset] = make_uint2(kgen_objPtr, get_global_id(0));
    //}
  }

  {% endfor %} {# /* Impl in Kernel.Hierarchy.Implementations */ #}
}

{% if not UseSpecConstWgSize %}
__attribute__((reqd_work_group_size(32, 1, 1))) 
{% endif %} 
__kernel void {{Kernel.Name}}_UpdateIndirect(__global struct {{MainClassName}}_UBO_Data* ubo, __global uint4* indirectBuffer)
{
  const uint lid = get_local_id(0); 
  if(lid < {{length(Kernel.Hierarchy.Implementations)}})
  {
    uint4 blocksNum = {1,1,1,0};
    uint4 zero      = {0,0,0,0};
    blocksNum.x = (ubo->objNum_{{Kernel.Hierarchy.Name}}Src[lid] + {{Kernel.WGSizeX}} - 1)/{{Kernel.WGSizeX}};
    blocksNum   = (blocksNum.x == 0) ? zero : blocksNum;
    indirectBuffer[{{Kernel.Hierarchy.IndirectOffset}}+lid] = blocksNum;
  }
} 

{% endif %} {# /* Kernel.Hierarchy.IndirectDispatch */ #}
