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
#define KGEN_FLAG_RETURN 1
#define KGEN_FLAG_BREAK  2
#define KGEN_FLAG_DONT_SET_EXIT 4
#define KGEN_FLAG_SET_EXIT_NEGATIVE 8

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
  ///////////////////////////////////////////////////////////////// 
  {% for name in Kernel.threadNames %}const uint {{name}} = get_global_id({{ loop.index }}); 
  {% endfor %}{% if Kernel.threadDim == 3 %}if({{Kernel.threadName1}} >= {{Kernel.threadIdName1}} || {{Kernel.threadName2}} >= {{Kernel.threadIdName2}} || {{Kernel.threadName3}} >= {{Kernel.threadIdName3}})
    return;{% else if Kernel.threadDim == 2 %}if({{Kernel.threadName1}} >= {{Kernel.threadIdName1}} || {{Kernel.threadName2}} >= {{Kernel.threadIdName2}})
    return;{% else %}if({{Kernel.threadName1}} >= {{Kernel.threadIdName1}})
    return;{% endif %}
  {% if Kernel.shouldCheckExitFlag %}if((kgen_threadFlags[{{Kernel.ThreadOffset}}] & kgen_tFlagsMask) != 0) 
    return;{% endif %}
  {% for Member in Kernel.Members %}const {{Member.Type}} {{Member.Name}} = ubo->{{Member.Name}};
  {% endfor %}{% if Kernel.IsBoolean %}bool kgenExitCond = false;{% endif %}
  ///////////////////////////////////////////////////////////////// 
{{Kernel.Source}}
  {% if Kernel.HasEpilog %}
  ///////////////////////////////////////////////////////////////// 
  KGEN_END:
  {% if Kernel.IsBoolean %}{
    const bool exitHappened = (kgen_tFlagsMask & KGEN_FLAG_SET_EXIT_NEGATIVE) != 0 ? !kgenExitCond : kgenExitCond;
    if((kgen_tFlagsMask & KGEN_FLAG_DONT_SET_EXIT) == 0 && exitHappened)
      kgen_threadFlags[tid] = ((kgen_tFlagsMask & KGEN_FLAG_BREAK) != 0) ? KGEN_FLAG_BREAK : KGEN_FLAG_RETURN;
  };{% endif %}
  ///////////////////////////////////////////////////////////////// 
  {% endif %}
}

## endfor

/* TODO for redection

__local float4 sdata[256];

uint32_t localId =  get_local_id(0);

static for (uint c = 256 / 2; c>32; c /= 2) where 32 is warp size
{
  if (localId < c)
    sdata[localId] += sdata[localId + c];
  SYNCTHREADS_LOCAL;
}

sdata[localId] += sdata[localId + 32]; 
sdata[localId] += sdata[localId + 16]; 
sdata[localId] += sdata[localId +  8]; 
sdata[localId] += sdata[localId +  4]; 
sdata[localId] += sdata[localId +  2]; 
sdata[localId] += sdata[localId +  1]; 

*/


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
