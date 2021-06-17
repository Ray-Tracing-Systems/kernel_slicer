#version 460
#extension GL_GOOGLE_include_directive : require

#include "common_generated.h"

/////////////////////////////////////////////////////////////////////
/////////////////// kernel  /////////////////////////////////////////
/////////////////////////////////////////////////////////////////////

layout(local_size_x = {{Kernel.WGSizeX}}, local_size_y = {{Kernel.WGSizeY}}, local_size_z = {{Kernel.WGSizeZ}}) in;

## for Arg in Kernel.Args
{% if not Arg.IsUBO %} 
layout(binding = {{loop.index}}, set = 0) buffer data{{loop.index}} { {{Arg.Type}} {{Arg.Name}}[]; }; //
{% endif %}
## endfor
layout(binding = {{length(Kernel.Args)}}, set = 0) buffer dataUBO { {{MainClassName}}_UBO_Data ubo; };

layout( push_constant ) uniform kernelIntArgs
{
  {% for UserArg in Kernel.UserArgs %} 
  {{UserArg.Type}} {{UserArg.Name}};
  {% endfor %}
  uint iNumElementsX; 
  uint iNumElementsY; 
  uint iNumElementsZ; 
  uint tFlagsMask;    
} kgenArgs;

void main()
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
  //KGEN_EPILOG:
  {% if Kernel.IsBoolean %}
  {
    const bool exitHappened = (kgenArgs.tFlagsMask & KGEN_FLAG_SET_EXIT_NEGATIVE) != 0 ? !kgenExitCond : kgenExitCond;
    if((kgenArgs.tFlagsMask & KGEN_FLAG_DONT_SET_EXIT) == 0 && exitHappened)
      kgen_threadFlags[tid] = ((kgenArgs.tFlagsMask & KGEN_FLAG_BREAK) != 0) ? KGEN_FLAG_BREAK : KGEN_FLAG_RETURN;
  };
  {% endif %}

  {% endif %} {# /* END of 'if Kernel.HasEpilog'  */ #}
}
