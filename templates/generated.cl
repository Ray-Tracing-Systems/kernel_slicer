/////////////////////////////////////////////////////////////////////
/////////////////// include files ///////////////////////////////////
/////////////////////////////////////////////////////////////////////

## for Incl in Includes  
#include "{{Incl}}"
## endfor

/////////////////////////////////////////////////////////////////////
/////////////////// declarations in class ///////////////////////////
/////////////////////////////////////////////////////////////////////
## for Decl in ClassDecls  
{{Decl}}
## endfor

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
  {% for Vec in Kernel.Vecs %}const uint {{Vec.Name}}_size = kgen_data[{{Vec.SizeOffset}}]; 
  {% endfor %}{% for Member in Kernel.Members %}const {{Member.Type}} {{Member.Name}} = *( (__global {{Member.Type}}*)(kgen_data + {{Member.Offset}}) );
  {% endfor %}/////////////////////////////////////////////////////////////////
{{Kernel.Source}}
}

## endfor
