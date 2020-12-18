/////////////////////////////////////////////////////////////////////
/////////////////// include files ///////////////////////////////////
/////////////////////////////////////////////////////////////////////

## for Incl in Includes  
#include "{{Incl}}"
## endfor

/////////////////////////////////////////////////////////////////////
/////////////////// local functions /////////////////////////////////
/////////////////////////////////////////////////////////////////////

## for LocalFunc in LocalFunctions  
{{LocalFunc}}

## endfor
#define KGEN_FLAG_RETURN 1
#define KGEN_FLAG_BREAK  2

/////////////////////////////////////////////////////////////////////
/////////////////// kernels /////////////////////////////////////////
/////////////////////////////////////////////////////////////////////

## for Kernel in Kernels  
__kernel void {{Kernel.Name}}(
## for Arg in Kernel.Args 
  __global {{Arg.Type}} restrict {{Arg.Name}},
## endfor
  const uint kgen_iNumElementsX, 
  const uint kgen_iNumElementsY,
  const uint kgen_iNumElementsZ,
  const uint kgen_tFlagsMask)
{
  /////////////////////////////////////////////////////////////////
  {% for name in Kernel.threadNames %}const uint {{name}} = get_global_id({{ loop.index }}); 
  {% endfor %}{% if Kernel.threadDim == 3 %}if({{Kernel.threadName1}} >= kgen_iNumElementsX || {{Kernel.threadName2}} >= kgen_iNumElementsY || {{Kernel.threadName3}} >= kgen_iNumElementsZ)
    return;{% else if Kernel.threadDim == 2 %}if({{Kernel.threadName1}} >= kgen_iNumElementsX || {{Kernel.threadName2}} >= kgen_iNumElementsY)
    return;{% else %}if({{Kernel.threadName1}} >= kgen_iNumElementsX)
    return;{% endif %}
  {% for Vec in Kernel.Vecs %}const uint {{Vec.Name}}_size = kgen_data[{{Vec.SizeOffset}}]; 
  {% endfor %}/////////////////////////////////////////////////////////////////
{{Kernel.Source}}
}
## endfor
