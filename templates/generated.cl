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

/////////////////////////////////////////////////////////////////////
/////////////////// kernels /////////////////////////////////////////
/////////////////////////////////////////////////////////////////////

## for Kernel in Kernels  
__kernel void {{Kernel.Name}}(
## for Arg in Kernel.Args 
  __global {{Arg.Type}} restrict {{Arg.Name}},
## endfor
## for ArgSize in Kernel.ArgSizes 
  const uint {{ArgSize}},
## endfor 
  const uint a_dummyArg)
{
  /////////////////////////////////////////////////////////////////
  {% for name in Kernel.threadNames %}const uint {{name}} = get_global_id({{ loop.index }}); 
  {% endfor %}{% if Kernel.threadDim == 3 %}if({{Kernel.threadName1}} >= {{Kernel.threadSize1}} || {{Kernel.threadName2}} >= {{Kernel.threadSize2}} || {{Kernel.threadName3}} >= {{Kernel.threadSize3}})
    return;{% else if Kernel.threadDim == 2 %}if({{Kernel.threadName1}} >= {{Kernel.threadSize1}} || {{Kernel.threadName2}} >= {{Kernel.threadSize2}})
    return;{% else %}if({{Kernel.threadName1}} >= {{Kernel.threadSize1}})
    return;{% endif %}
  {% for Vec in Kernel.Vecs %}const uint {{Vec.Name}}_size = kgen_data[{{Vec.SizeOffset}}]; 
  {% endfor %}/////////////////////////////////////////////////////////////////
{{Kernel.Source}}
}
## endfor
