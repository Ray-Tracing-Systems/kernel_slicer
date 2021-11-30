/////////////////////////////////////////////////////////////////////
/////////////////// include files ///////////////////////////////////
/////////////////////////////////////////////////////////////////////
#include "LiteMath.h"
## for Incl in Includes  
#include "{{Incl}}"
## endfor

/////////////////////////////////////////////////////////////////////
/////////////////// declarations in class ///////////////////////////
/////////////////////////////////////////////////////////////////////
## for Decl in ClassDecls  
{{Decl}}
## endfor

#include "include/{{UBOIncl}}"

{% if length(Hierarchies) > 0 %}
{% for Hierarchy in Hierarchies %}
/////////////////////////////////////////////////////////////////////
/////////////////// declarations of {{Hierarchy.Name}}  
/////////////////////////////////////////////////////////////////////
{% for Decl in Hierarchy.Constants %}
{{Decl.Type}} {{Hierarchy.Name}}_{{Decl.Name}} = {{Decl.Value}};
{% endfor %}
{% for Impl in Hierarchy.Implementations %}

  typedef struct {{Impl.ClassName}}T 
  {
    {% for Field in Impl.Fields %}
    {{Field}};
    {% endfor %}
  }{{Impl.ClassName}};  
{% for MemberSrc in Impl.MemberFunctions %}

{{MemberSrc}}
{% endfor %} {# /* for Impl in Hierarchy.Implementations */ #} 
{% endfor %} {# /* for Decl in Hierarchy.Constants */ #}
{% endfor %} {# /* for Hierarchy in Hierarchies */ #}

#define PREFIX_SUMM_MACRO(idata,odata,l_Data,_bsize)       \
{                                                          \
  uint pos = 2 * get_local_id(0) - (get_local_id(0) & (_bsize - 1)); \
  l_Data[pos] = 0;                                         \
  pos += _bsize;                                           \
  l_Data[pos] = idata;                                     \
                                                           \
  for (uint offset = 1; offset < _bsize; offset <<= 1)     \
  {                                                        \
    barrier(CLK_LOCAL_MEM_FENCE);                          \
    uint t = l_Data[pos] + l_Data[pos - offset];           \
    barrier(CLK_LOCAL_MEM_FENCE);                          \
    l_Data[pos] = t;                                       \
  }                                                        \
                                                           \
  odata = l_Data[pos];                                     \
}                                                          \

{% endif %}
/////////////////////////////////////////////////////////////////////
/////////////////// local functions /////////////////////////////////
/////////////////////////////////////////////////////////////////////

## for LocalFunc in LocalFunctions  
{{LocalFunc}}

## endfor
#define KGEN_FLAG_RETURN            1
#define KGEN_FLAG_BREAK             2
#define KGEN_FLAG_DONT_SET_EXIT     4
#define KGEN_FLAG_SET_EXIT_NEGATIVE 8
#define KGEN_REDUCTION_LAST_STEP    16

/////////////////////////////////////////////////////////////////////
/////////////////// kernels /////////////////////////////////////////
/////////////////////////////////////////////////////////////////////

## for Kernel in Kernels
{% if Kernel.IsIndirect %}
{% if not UseSpecConstWgSize %}
__attribute__((reqd_work_group_size(1, 1, 1)))
{% endif %}
__kernel void {{Kernel.Name}}_UpdateIndirect(__global struct {{MainClassName}}_UBO_Data* ubo, __global uint4* indirectBuffer)
{
  uint4 blocksNum = {1,1,1,0};
  blocksNum.x = ({{Kernel.IndirectSizeX}} - {{Kernel.IndirectStartX}} + {{Kernel.WGSizeX}} - 1)/{{Kernel.WGSizeX}};
  {% if Kernel.threadDim == 2 %}
  blocksNum.y = ({{Kernel.IndirectSizeY}} - {{Kernel.IndirectStartY}} + {{Kernel.WGSizeY}} - 1)/{{Kernel.WGSizeY}};
  {% endif %}
  {% if Kernel.threadDim == 3 %}
  blocksNum.z = ({{Kernel.IndirectSizeZ}} - {{Kernel.IndirectStartZ}} + {{Kernel.WGSizeZ}} - 1)/{{Kernel.WGSizeZ}};
  {% endif %}
  indirectBuffer[{{Kernel.IndirectOffset}}] = blocksNum;
} 
{% endif %}
{% if Kernel.IsMaker %}
{% include "inc_maker.cl" %}
{% else if Kernel.IsVirtual %}
{% include "inc_vkernel.cl" %}
{% else %}

{% if not UseSpecConstWgSize %}
__attribute__((reqd_work_group_size({{Kernel.WGSizeX}}, {{Kernel.WGSizeY}}, {{Kernel.WGSizeZ}})))
{% endif %} 
__kernel void {{Kernel.Name}}({% include "inc_args.cl" %})
{
  ///////////////////////////////////////////////////////////////// prolog
  {% if Kernel.InitKPass %}
  if(get_global_id(0)!=0)
    return;
  {% else %}
  {# /*------------------------------------------------------------- BEG. REDUCTION INIT ------------------------------------------------------------- */ #}
  {% if length(Kernel.SubjToRed) > 0 or length(Kernel.ArrsToRed) > 0 %}                        
  {% include "inc_reduction_init.h" %}
  {% endif %} 
  {# /*------------------------------------------------------------- END. REDUCTION INIT ------------------------------------------------------------- */ #}
  {# {% for name in Kernel.threadNames %} #}
  {# const uint {{name}} = get_global_id({{ loop.index }}); #}
  {# {% endfor %} #}
  {% for TID in Kernel.ThreadIds %}
  {% if TID.Simple %}
  const {{TID.Type}} {{TID.Name}} = ({{TID.Type}})(get_global_id({{ loop.index }})); 
  {% else %}
  const {{TID.Type}} {{TID.Name}} = {{TID.Start}} + ({{TID.Type}})(get_global_id({{ loop.index }}))*{{TID.Stride}}; 
  {% endif %}
  {% endfor %}
  {# /*------------------------------------------------------------- BEG. CHECK EXIT COND ------------------------------------------------------------- */ #}
  {% include "inc_exit_cond.cl" %}
  {# /*------------------------------------------------------------- END. CHECK EXIT COND ------------------------------------------------------------- */ #}
  {% if Kernel.IsBoolean %}
  bool kgenExitCond = false;
  {% endif %}
  {% endif %}
  ///////////////////////////////////////////////////////////////// prolog
  {# /*------------------------------------------------------------- KERNEL SOURCE ------------------------------------------------------------- */ #}
  {{Kernel.Source}}
  {# /*------------------------------------------------------------- KERNEL SOURCE ------------------------------------------------------------- */ #}
  {% if Kernel.HasEpilog %}
  KGEN_EPILOG:
  {% if Kernel.IsBoolean %}
  {
    const bool exitHappened = (kgen_tFlagsMask & KGEN_FLAG_SET_EXIT_NEGATIVE) != 0 ? !kgenExitCond : kgenExitCond;
    if((kgen_tFlagsMask & KGEN_FLAG_DONT_SET_EXIT) == 0 && exitHappened)
      kgen_threadFlags[tid] = ((kgen_tFlagsMask & KGEN_FLAG_BREAK) != 0) ? KGEN_FLAG_BREAK : KGEN_FLAG_RETURN;
  };
  {% endif %}
  {# /*------------------------------------------------------------- BEG. REDUCTION PASS ------------------------------------------------------------- */ #}
  {% if length(Kernel.SubjToRed) > 0 or length(Kernel.ArrsToRed) > 0 %}                      
  {% include "inc_reduction_inkernel.h" %}
  {% endif %}
  {# /*------------------------------------------------------------- END. REDUCTION PASS ------------------------------------------------------------- */ #}
  {% endif %} {# /* END of 'if Kernel.HasEpilog'  */ #}
}
{% if Kernel.FinishRed %}

{% include "inc_reduction_finish.h" %}
{% endif %}

{% endif %} {# /* end if 'Kernel.IsMaker' */ #}

## endfor
{% if UseServiceMemCopy %}
{% if not UseSpecConstWgSize %}
__attribute__((reqd_work_group_size(256, 1, 1)))
{% endif %}
__kernel void copyKernelFloat(
  __global float* out_data,
  __global float* in_data,
  const uint length)
{
  const uint i = get_global_id(0);
  if(i >= length)
    return;
  out_data[i] = in_data[i];
}
{% endif %} {# /* UseServiceMemCopy */ #}