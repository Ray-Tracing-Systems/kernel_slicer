#version 460
#extension GL_EXT_ray_query : require
#extension GL_GOOGLE_include_directive : require

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
/////////////////// kernel  /////////////////////////////////////////
/////////////////////////////////////////////////////////////////////

layout(local_size_x = {{Kernel.WGSizeX}}, local_size_y = {{Kernel.WGSizeY}}, local_size_z = {{Kernel.WGSizeZ}}) in;

## for Arg in Kernel.Args
{% if not Arg.IsUBO %} 
layout(binding = {{loop.index}}, set = 0) buffer data{{loop.index}} { {{Arg.Type}} {{Arg.Name}}[]; }; //
{% endif %}
## endfor

layout( push_constant ) uniform kernelIntArgs
{
  {% for UserArg in Kernel.UserArgs %} 
  {{UserArg.Type}} {{UserArg.Name}},
  {% endfor %}
  uint iNumElementsX; // const uint kgen_iNumElementsX, 
  uint iNumElementsY; // const uint kgen_iNumElementsY,
  uint iNumElementsZ; // const uint kgen_iNumElementsZ,
  uint tFlagsMask;    // const uint kgen_tFlagsMask)
} kgenArgs;

void {{Kernel.Name}}()
{
  ///////////////////////////////////////////////////////////////// prolog
  {% if Kernel.InitKPass %}
  if(get_global_id(0)!=0)
    return;
  {% else %}
  {% for name in Kernel.threadNames %}
  const uint {{name}} = gl_GlobalInvocationID[{{ loop.index }}]; 
  {% endfor %}
  {# /*------------------------------------------------------------- BEG. CHECK EXIT COND ------------------------------------------------------------- */ #}
  {% include "inc_exit_cond.glsl" %}
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

  {% endif %} {# /* END of 'if Kernel.HasEpilog'  */ #}
}



