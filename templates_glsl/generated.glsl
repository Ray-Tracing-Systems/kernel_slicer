#version 460
#extension GL_GOOGLE_include_directive : require

#include "common_generated.h"

## for Arg in Kernel.Args
{% if not Arg.IsUBO %} 
{% if Arg.IsImage %}
layout(binding = {{loop.index}}, set = 0{% if Arg.NeedFmt%}, {{Arg.ImFormat}}{% endif %}) uniform {{Arg.Type}} {{Arg.Name}}; //
{% else %}
layout(binding = {{loop.index}}, set = 0) buffer data{{loop.index}} { {{Arg.Type}} {{Arg.Name}}[]; }; //
{% endif %} {# /* Arg.IsImage */ #}
{% endif %} {# /* not Arg.IsUBO */ #}
## endfor
layout(binding = {{length(Kernel.Args)}}, set = 0) buffer dataUBO { {{MainClassName}}_UBO_Data ubo; };

## for MembFunc in MemberFunctions  
{{MembFunc}}

## endfor
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

layout(local_size_x = {{Kernel.WGSizeX}}, local_size_y = {{Kernel.WGSizeY}}, local_size_z = {{Kernel.WGSizeZ}}) in;

layout( push_constant ) uniform kernelIntArgs
{
  {% for UserArg in Kernel.UserArgs %} 
  {{UserArg.Type}} {{UserArg.Name}};
  {% endfor %}
  uint {{Kernel.threadIdName1}}; 
  uint {{Kernel.threadIdName2}}; 
  uint {{Kernel.threadIdName3}}; 
  uint tFlagsMask;    
} kgenArgs;

void main()
{
  {% if not Kernel.InitKPass %}
  ///////////////////////////////////////////////////////////////// prolog
  {% for TID in Kernel.ThreadIds %}
  const {{TID.Type}} {{TID.Name}} = {{TID.Type}}(gl_GlobalInvocationID[{{ loop.index }}]); 
  {% endfor %}
  {# /*------------------------------------------------------------- BEG. CHECK EXIT COND ------------------------------------------------------------- */ #}
  {% include "inc_exit_cond.glsl" %}
  {# /*------------------------------------------------------------- END. CHECK EXIT COND ------------------------------------------------------------- */ #}
  {% if length(Kernel.Members) > 0 %}

  {% for Member in Kernel.Members %}
  const {{Member.Type}} {{Member.Name}} = ubo.{{Member.Name}};
  {% endfor %} 
  {% endif %} {# /* length(Kernel.Members) > 0 */ #}
  {% if Kernel.IsBoolean %}
  bool kgenExitCond = false;
  {% endif %}
  ///////////////////////////////////////////////////////////////// prolog
  {% endif %}
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
