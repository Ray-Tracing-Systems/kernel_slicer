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
  {% if Kernel.InitKPass %}
  if(get_global_id(0)!=0)
    return;
  {% else %}
  ///////////////////////////////////////////////////////////////// prolog
  {% if Kernel.HasEpilog %}
  {% for redvar in Kernel.SubjToRed %} 
  __local {{redvar.Type}} {{redvar.Name}}Shared[{{Kernel.WGSizeX}}]; 
  {{redvar.Name}}Shared[get_local_id(0)] = {{redvar.Init}}; 
  {% endfor %}
  SYNCTHREADS; 
  {% endif %}
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
  ///////////////////////////////////////////////////////////////// prolog
  {% endif %}
{{Kernel.Source}}
  {% if Kernel.HasEpilog %}
  KGEN_EPILOG:
  {% if Kernel.IsBoolean %}
  {
    const bool exitHappened = (kgen_tFlagsMask & KGEN_FLAG_SET_EXIT_NEGATIVE) != 0 ? !kgenExitCond : kgenExitCond;
    if((kgen_tFlagsMask & KGEN_FLAG_DONT_SET_EXIT) == 0 && exitHappened)
      kgen_threadFlags[tid] = ((kgen_tFlagsMask & KGEN_FLAG_BREAK) != 0) ? KGEN_FLAG_BREAK : KGEN_FLAG_RETURN;
  };
  {% endif %}
  {% if Kernel.HasReduct %}
  {
    const uint32_t localId = get_local_id(0);
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
    }
    {% endfor %}
    if(localId == 0)
    {
      {% for redvar in Kernel.SubjToRed %}
      {% if redvar.SupportAtomic %}
      {{redvar.AtomicOp}}(&ubo->{{redvar.Name}}, {{redvar.Name}}Shared[0]);
      {% else %}
      {{ redvar.OutTempName }}[get_global_id(0)/{{Kernel.WGSizeX}}] = {{redvar.Name}}Shared[0]; // fill finish reduction in subsequent kernel passes
      {% endif %}
      {% endfor %}
    }
  }
  {% endif %}
  {% endif %}
}
{% if Kernel.FinishRed %}

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
  __local {{redvar.Type}} {{redvar.Name}}Shared[{{Kernel.WGSizeX}}]; 
  {{redvar.Name}}Shared[localId] = {{redvar.Init}}; 
  {% endif %}
  {% endfor %}
  SYNCTHREADS;
  if(globalId < {{Kernel.threadIdName1}})
  {
    {% for redvar in Kernel.SubjToRed %}
    {% if not redvar.SupportAtomic %}
    {{redvar.Name}}Shared[localId] = {{ redvar.OutTempName }}[{{Kernel.threadIdName2}} + globalId]; // use {{Kernel.threadIdName2}} for 'InputOffset' 
    {% endif %}
    {% endfor %}
  }
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
    }
    else
    {
      {% for redvar in Kernel.SubjToRed %}
      {% if not redvar.SupportAtomic %}
      {{ redvar.OutTempName }}[{{Kernel.threadIdName3}} + globalId/{{Kernel.WGSizeX}}] = {{redvar.Name}}Shared[0]; // use {{Kernel.threadIdName3}} for 'OutputOffset'
      {% endif %}
      {% endfor %}
    }
  }
}
{% endif %}

## endfor


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
