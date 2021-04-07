/////////////////////////////////////////////////////////////////////
/////////////////// include files ///////////////////////////////////
/////////////////////////////////////////////////////////////////////
#include "include/OpenCLMath.h"
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
  blocksNum.x = ({{Kernel.IndirectSizeX}} + {{Kernel.WGSizeX}} - 1)/{{Kernel.WGSizeX}};
  {% if Kernel.threadDim == 2 %}
  blocksNum.y = ({{Kernel.IndirectSizeY}} + {{Kernel.WGSizeY}} - 1)/{{Kernel.WGSizeY}};
  {% endif %}
  {% if Kernel.threadDim == 3 %}
  blocksNum.z = ({{Kernel.IndirectSizeZ}} + {{Kernel.WGSizeZ}} - 1)/{{Kernel.WGSizeZ}};
  {% endif %}
  indirectBuffer[{{Kernel.IndirectOffset}}] = blocksNum;
} 
{% endif %}
{% if Kernel.IsMaker %}
{% include "inc_maker.cl" %}
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
  __global struct {{MainClassName}}_UBO_Data* ubo,
  const uint {{Kernel.threadIdName1}}, 
  const uint {{Kernel.threadIdName2}},
  const uint {{Kernel.threadIdName3}},
  const uint kgen_tFlagsMask)
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
  {% for name in Kernel.threadNames %}
  const uint {{name}} = get_global_id({{ loop.index }}); 
  {% endfor %}
  {# /*------------------------------------------------------------- BEG. CHECK EXIT COND ------------------------------------------------------------- */ #}
  {% if Kernel.threadDim == 3 %}
  if({{Kernel.threadName1}} >= {{Kernel.IndirectSizeX}} || {{Kernel.threadName2}} >= {{Kernel.IndirectSizeY}} || {{Kernel.threadName3}} >= {{Kernel.IndirectSizeZ}})
    return;
  {% else if Kernel.threadDim == 2 %}
  if({{Kernel.threadName1}} >= {{Kernel.IndirectSizeX}} || {{Kernel.threadName2}} >= {{Kernel.IndirectSizeY}})
    return;
  {% else %}
  if({{Kernel.threadName1}} >= {{Kernel.IndirectSizeX}})
    return;
  {% endif %}
  {% if Kernel.shouldCheckExitFlag %}
  if((kgen_threadFlags[{{Kernel.ThreadOffset}}] & kgen_tFlagsMask) != 0) 
    return;
  {% endif %}
  {# /*------------------------------------------------------------- END. CHECK EXIT COND ------------------------------------------------------------- */ #}
  {% for Member in Kernel.Members %}
  const {{Member.Type}} {{Member.Name}} = ubo->{{Member.Name}};
  {% endfor %}
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
