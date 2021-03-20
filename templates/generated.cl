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
// indirect update kernel for {{Kernel.Name}}
__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void {{Kernel.Name}}_UpdateIndirect(__global struct {{MainClassName}}_UBO_Data* restrict ubo, __global uint4* indirectBuffer)
{
  uint4 blocksNum = {0,0,0,0};
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
__attribute__((reqd_work_group_size({{Kernel.WGSizeX}}, {{Kernel.WGSizeY}}, {{Kernel.WGSizeZ}}))) 
__kernel void {{Kernel.Name}}(
## for Arg in Kernel.Args 
  __global {{Arg.Type}} restrict {{Arg.Name}},
## endfor
## for UserArg in Kernel.UserArgs 
  {{UserArg.Type}} {{UserArg.Name}},
## endfor
  __global struct {{MainClassName}}_UBO_Data* restrict ubo,
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
  {% if length(Kernel.SubjToRed) > 0 or length(Kernel.ArrsToRed) > 0 %}                        {# BEG. REDUCTION INIT #}
  {% for redvar in Kernel.SubjToRed %} 
  __local {{redvar.Type}} {{redvar.Name}}Shared[{{Kernel.WGSizeX}}*{{Kernel.WGSizeY}}*{{Kernel.WGSizeZ}}]; 
  {% endfor %}
  {% for redvar in Kernel.ArrsToRed %} 
  __local {{redvar.Type}} {{redvar.Name}}Shared[{{redvar.ArraySize}}][{{Kernel.WGSizeX}}*{{Kernel.WGSizeY}}*{{Kernel.WGSizeZ}}]; 
  {% endfor %}
  {
    {% if Kernel.threadDim == 1 %}
    const uint localId = get_local_id(0); 
    {% else %}
    const uint localId = get_local_id(0) + {{Kernel.WGSizeX}}*get_local_id(1); 
    {% endif %}
    {% for redvar in Kernel.SubjToRed %} 
    {{redvar.Name}}Shared[localId] = {{redvar.Init}}; 
    {% endfor %}
    {% for redvar in Kernel.ArrsToRed %} 
    {% for index in range(redvar.ArraySize) %}
    {{redvar.Name}}Shared[{{loop.index}}][localId] = {{redvar.Init}}; 
    {% endfor %}
    {% endfor %}
  }
  SYNCTHREADS; 
  {% endif %}                                                                                   {# END. REDUCTION INIT #}
  {% for name in Kernel.threadNames %}
  const uint {{name}} = get_global_id({{ loop.index }}); 
  {% endfor %}
  {% if Kernel.threadDim == 3 %}
  if({{Kernel.threadName1}} >= {{Kernel.threadIdName1}} || {{Kernel.threadName2}} >= {{Kernel.threadIdName2}} || {{Kernel.threadName3}} >= {{Kernel.threadIdName3}})
    return;
  {% else if Kernel.threadDim == 2 %}
  if({{Kernel.threadName1}} >= {{Kernel.threadIdName1}} || {{Kernel.threadName2}} >= {{Kernel.threadIdName2}})
    return;
  {% else %}
  if({{Kernel.threadName1}} >= {{Kernel.threadIdName1}})
    return;
  {% endif %}
  {% if Kernel.shouldCheckExitFlag %}
  if((kgen_threadFlags[{{Kernel.ThreadOffset}}] & kgen_tFlagsMask) != 0) 
    return;
  {% endif %}
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
  {% if length(Kernel.SubjToRed) > 0 or length(Kernel.ArrsToRed) > 0 %}                      {# BEG. REDUCTION PASS #}
  {
    {% if Kernel.threadDim == 1 %}
    const uint localId = get_local_id(0); 
    {% else %}
    const uint localId = get_local_id(0) + {{Kernel.WGSizeX}}*get_local_id(1); 
    {% endif %}
    SYNCTHREADS;
    {% for offset in Kernel.RedLoop1 %} 
    if (localId < {{offset}}) 
    {
      {% for redvar in Kernel.SubjToRed %}
      {% if redvar.BinFuncForm %}
      {{redvar.Name}}Shared[localId] = {{redvar.Op}}({{redvar.Name}}Shared[localId], {{redvar.Name}}Shared[localId + {{offset}}]);
      {% else %}
      {{redvar.Name}}Shared[localId] {{redvar.Op}} {{redvar.Name}}Shared[localId + {{offset}}];
      {% endif %}
      {% endfor %}
      {% for redvar in Kernel.ArrsToRed %}
      {% for index in range(redvar.ArraySize) %}
      {% if redvar.BinFuncForm %}
      {{redvar.Name}}Shared[{{loop.index}}][localId] = {{redvar.Op}}({{redvar.Name}}Shared[{{loop.index}}][localId], {{redvar.Name}}Shared[{{loop.index}}][localId + {{offset}}]);
      {% else %}
      {{redvar.Name}}Shared[{{loop.index}}][localId] {{redvar.Op}} {{redvar.Name}}Shared[{{loop.index}}][localId + {{offset}}];
      {% endif %}
      {% endfor %}
      {% endfor %}
    }
    SYNCTHREADS;
    {% endfor %}
    {% for offset in Kernel.RedLoop2 %} 
    if (localId < {{offset}}) 
    {
      {% for redvar in Kernel.SubjToRed %}
      {% if redvar.BinFuncForm %}
      {{redvar.Name}}Shared[localId] = {{redvar.Op}}({{redvar.Name}}Shared[localId], {{redvar.Name}}Shared[localId + {{offset}}]);
      {% else %}
      {{redvar.Name}}Shared[localId] {{redvar.Op}} {{redvar.Name}}Shared[localId + {{offset}}];
      {% endif %}
      {% endfor %}
      {% for redvar in Kernel.ArrsToRed %}
      {% for index in range(redvar.ArraySize) %}
      {% if redvar.BinFuncForm %}
      {{redvar.Name}}Shared[{{loop.index}}][localId] = {{redvar.Op}}({{redvar.Name}}Shared[{{loop.index}}][localId], {{redvar.Name}}Shared[{{loop.index}}][localId + {{offset}}]);
      {% else %}
      {{redvar.Name}}Shared[{{loop.index}}][localId] {{redvar.Op}} {{redvar.Name}}Shared[{{loop.index}}][localId + {{offset}}];
      {% endif %}
      {% endfor %}
      {% endfor %}
    }
    {% if Kernel.threadDim > 1 %}
    SYNCTHREADS;
    {% endif %}
    {% endfor %}
    if(localId == 0)
    {
      {% if Kernel.threadDim == 1 %}
      const uint offset = get_group_id(0);
      {% else %}
      const uint offset = get_group_id(0) + get_num_groups(0)*get_group_id(1);
      {% endif %}
      {% for redvar in Kernel.SubjToRed %}
      {% if redvar.SupportAtomic %}
      {{redvar.AtomicOp}}(&ubo->{{redvar.Name}}, {{redvar.Name}}Shared[0]);
      {% else %}
      {{ redvar.OutTempName }}[offset] = {{redvar.Name}}Shared[0]; // finish reduction in subsequent kernel passes
      {% endif %}
      {% endfor %}
      {% for redvar in Kernel.ArrsToRed %}
      {% for outName in redvar.OutTempNameA %}
      {% if redvar.SupportAtomic %}
      {{redvar.AtomicOp}}(&(ubo->{{redvar.Name}}[{{loop.index}}]), {{redvar.Name}}Shared[{{loop.index}}][0]);
      {% else %}
      {{ outName }}[offset] = {{redvar.Name}}Shared[{{loop.index}}][0]; // finish reduction in subsequent kernel passes
      {% endif %}
      {% endfor %}
      {% endfor %}
    }
  }                                                                                             {# END. REDUCTION PASS #}
  {% endif %}
  {% endif %}

}
{% if Kernel.FinishRed %}

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void {{Kernel.Name}}_Reduction(
## for Arg in Kernel.Args 
  __global {{Arg.Type}} restrict {{Arg.Name}},
## endfor
## for UserArg in Kernel.UserArgs 
  {{UserArg.Type}} {{UserArg.Name}},
## endfor
   __global struct {{MainClassName}}_UBO_Data* restrict ubo,
  const uint {{Kernel.threadIdName1}}, 
  const uint {{Kernel.threadIdName2}},
  const uint {{Kernel.threadIdName3}},
  const uint kgen_tFlagsMask)
{
  const uint globalId = get_global_id(0);
  const uint localId  = get_local_id(0);

  {% for redvar in Kernel.SubjToRed %}
  {% if not redvar.SupportAtomic %}
  __local {{redvar.Type}} {{redvar.Name}}Shared[256]; 
  {{redvar.Name}}Shared[localId] = (globalId < {{Kernel.threadIdName1}}) ?  {{ redvar.OutTempName }}[{{Kernel.threadIdName2}} + globalId]  :  {{redvar.Init}}; // use {{Kernel.threadIdName2}} for 'InputOffset'
  {% endif %}
  {% endfor %}
  {% for redvar in Kernel.ArrsToRed %}
  __local {{redvar.Type}} {{redvar.Name}}Shared[{{redvar.ArraySize}}][256]; 
  {% for outName in redvar.OutTempNameA %}
  {% if not redvar.SupportAtomic %}
  {{redvar.Name}}Shared[{{loop.index}}][localId] = (globalId < {{Kernel.threadIdName1}}) ? {{ outName }}[{{Kernel.threadIdName2}} + globalId] : {{redvar.Init}}; // use {{Kernel.threadIdName2}} for 'InputOffset'
  {% endif %}
  {% endfor %}
  {% endfor %}
  SYNCTHREADS;
  {% for offset in Kernel.RedLoop1 %} 
  if (localId < {{offset}}) 
  {
    {% for redvar in Kernel.SubjToRed %}
    {% if not redvar.SupportAtomic %}
    {% if redvar.BinFuncForm %}
    {{redvar.Name}}Shared[localId] = {{redvar.Op}}({{redvar.Name}}Shared[localId], {{redvar.Name}}Shared[localId + {{offset}}]);
    {% else %}
    {{redvar.Name}}Shared[localId] {{redvar.Op}} {{redvar.Name}}Shared[localId + {{offset}}];
    {% endif %}
    {% endif %}
    {% endfor %}
    {% for redvar in Kernel.ArrsToRed %}
    {% for index in range(redvar.ArraySize) %}
    {% if not redvar.SupportAtomic %}
    {% if redvar.BinFuncForm %}
    {{redvar.Name}}Shared[{{loop.index}}][localId] = {{redvar.Op}}({{redvar.Name}}Shared[{{loop.index}}][localId], {{redvar.Name}}Shared[{{loop.index}}][localId + {{offset}}]);
    {% else %}
    {{redvar.Name}}Shared[{{loop.index}}][localId] {{redvar.Op}} {{redvar.Name}}Shared[{{loop.index}}][localId + {{offset}}];
    {% endif %}
    {% endif %}
    {% endfor %}
    {% endfor %}
  }
  SYNCTHREADS;
  {% endfor %}
  {% for offset in Kernel.RedLoop2 %} 
  if (localId < {{offset}}) 
  {
    {% for redvar in Kernel.SubjToRed %}
    {% if not redvar.SupportAtomic %}
    {% if redvar.BinFuncForm %}
    {{redvar.Name}}Shared[localId] = {{redvar.Op}}({{redvar.Name}}Shared[localId], {{redvar.Name}}Shared[localId + {{offset}}]);
    {% else %}
    {{redvar.Name}}Shared[localId] {{redvar.Op}} {{redvar.Name}}Shared[localId + {{offset}}];
    {% endif %}
    {% endif %}
    {% endfor %}
    {% for redvar in Kernel.ArrsToRed %}
    {% for index in range(redvar.ArraySize) %}
    {% if not redvar.SupportAtomic %}
    {% if redvar.BinFuncForm %}
    {{redvar.Name}}Shared[{{loop.index}}][localId] = {{redvar.Op}}({{redvar.Name}}Shared[{{loop.index}}][localId], {{redvar.Name}}Shared[{{loop.index}}][localId + {{offset}}]);
    {% else %}
    {{redvar.Name}}Shared[{{loop.index}}][localId] {{redvar.Op}} {{redvar.Name}}Shared[{{loop.index}}][localId + {{offset}}];
    {% endif %}
    {% endif %}
    {% endfor %}
    {% endfor %}
  }
  {% endfor %}
  if(localId == 0)
  {
    if((kgen_tFlagsMask & KGEN_REDUCTION_LAST_STEP) != 0)
    {
      {% for redvar in Kernel.SubjToRed %}
      {% if not redvar.SupportAtomic %}
      {% if redvar.NegLastStep %}
      ubo->{{redvar.Name}} -= {{redvar.Name}}Shared[0];
      {% else %}
      {% if redvar.BinFuncForm %}
      ubo->{{redvar.Name}} = {{redvar.Op}}(ubo->{{redvar.Name}}, {{redvar.Name}}Shared[0]);
      {% else %}
      ubo->{{redvar.Name}} {{redvar.Op}} {{redvar.Name}}Shared[0];
      {% endif %}
      {% endif %}
      {% endif %}
      {% endfor %}
      {% for redvar in Kernel.ArrsToRed %}
      {% for index in range(redvar.ArraySize) %}
      {% if not redvar.SupportAtomic %}
      {% if redvar.NegLastStep %}
      ubo->{{redvar.Name}}[{{loop.index}}] -= {{redvar.Name}}Shared[{{loop.index}}][0];
      {% else %}
      {% if redvar.BinFuncForm %}
      ubo->{{redvar.Name}}[{{loop.index}}] = {{redvar.Op}}(ubo->{{redvar.Name}}[{{loop.index}}], {{redvar.Name}}Shared[{{loop.index}}][0]);
      {% else %}
      ubo->{{redvar.Name}}[{{loop.index}}] {{redvar.Op}} {{redvar.Name}}Shared[{{loop.index}}][0];
      {% endif %}
      {% endif %}
      {% endif %}
      {% endfor %}
      {% endfor %}
    }
    else
    {
      {% for redvar in Kernel.SubjToRed %}
      {% if not redvar.SupportAtomic %}
      {{ redvar.OutTempName }}[{{Kernel.threadIdName3}} + get_group_id(0)] = {{redvar.Name}}Shared[0]; // use {{Kernel.threadIdName3}} for 'OutputOffset'
      {% endif %}
      {% endfor %}
      {% for redvar in Kernel.ArrsToRed %}
      {% for outName in redvar.OutTempNameA %}
      {% if not redvar.SupportAtomic %}
      {{ outName }}[{{Kernel.threadIdName3}} + get_group_id(0)] = {{redvar.Name}}Shared[{{loop.index}}][0]; // use {{Kernel.threadIdName3}} for 'OutputOffset'
      {% endif %}
      {% endfor %}
      {% endfor %}
    }
  }
}
{% endif %}

## endfor

__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void copyKernelFloat(
  __global float* restrict out_data,
  __global float* restrict in_data,
  const uint length)
{
  const uint i = get_global_id(0);
  if(i >= length)
    return;
  out_data[i] = in_data[i];
}
