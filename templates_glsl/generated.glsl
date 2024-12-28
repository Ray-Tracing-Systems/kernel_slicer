#version 460
#extension GL_GOOGLE_include_directive : require
{% if Kernel.UseRayGen %}
#extension GL_EXT_ray_tracing : require 
{% if Kernel.UseMotionBlur %}
#extension GL_NV_ray_tracing_motion_blur : require
{% endif %}
{% else %}
#extension GL_EXT_ray_query : require
{% endif %}
{% if Kernel.NeedTexArray %}
#extension GL_EXT_nonuniform_qualifier : require
{% endif %}
{% if HasAllRefs %}
#extension GL_EXT_buffer_reference : require
{% endif %}
{% if Kernel.UseSubGroups %}
#extension GL_KHR_shader_subgroup_arithmetic: enable
{% endif %}

{% include "common_generated.glsl" %}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
{% if not Kernel.UseRayGen %}
layout(local_size_x = {{Kernel.WGSizeX}}, local_size_y = {{Kernel.WGSizeY}}, local_size_z = {{Kernel.WGSizeZ}}) in;
{% endif %}

layout( push_constant ) uniform kernelArgs
{
  {% for UserArg in Kernel.UserArgs %} 
  {{UserArg.Type}} {{UserArg.Name}};
  {% endfor %}
  {{Kernel.threadSZType1}} {{Kernel.threadSZName1}}; 
  {{Kernel.threadSZType2}} {{Kernel.threadSZName2}}; 
  {{Kernel.threadSZType3}} {{Kernel.threadSZName3}}; 
  uint tFlagsMask;    
} kgenArgs;

{% if not Kernel.InitKPass %}
{% for redvar in Kernel.SubjToRed %} 
shared {{redvar.Type}} {{redvar.Name}}Shared[{{Kernel.WGSizeX}}*{{Kernel.WGSizeY}}*{{Kernel.WGSizeZ}}]; 
{% endfor %}
{% for redvar in Kernel.ArrsToRed %} 
shared {{redvar.Type}} {{redvar.Name}}Shared[{{redvar.ArraySize}}][{{Kernel.WGSizeX}}*{{Kernel.WGSizeY}}*{{Kernel.WGSizeZ}}]; 
{% endfor %}
{% endif %}

{% if Kernel.EnableBlockExpansion %}
{% for TID in Kernel.ThreadSizeBE %}
const {{TID.Type}} {{TID.Name}} = {{TID.Value}}; 
{% endfor %}
{% for Var in Kernel.SharedBE %}
shared {{Var}}
{% endfor %}
{% endif %}

void main()
{
  {% if not Kernel.EnableBlockExpansion %}
  bool runThisThread = true;
  {% endif %}
  {% if not Kernel.InitKPass %}
  {% if Kernel.EnableBlockExpansion %}
  {% for TID in Kernel.ThreadIds %}
  {% if TID.Simple %}
  const {{TID.Type}} {{TID.Name}} = {{TID.Type}}(gl_WorkGroupID[{{ loop.index }}]); 
  {% else %}
  const {{TID.Type}} {{TID.Name}} = {{TID.Start}} + {{TID.Type}}(gl_WorkGroupID[{{ loop.index }}])*{{TID.Stride}}; 
  {% endif %}
  {% endfor %}
  {% else %}
  {% for TID in Kernel.ThreadIds %}
  {% if TID.Simple %}
  const {{TID.Type}} {{TID.Name}} = {{TID.Type}}({% if Kernel.UseRayGen %}gl_LaunchIDEXT{% else %}gl_GlobalInvocationID{% endif %}[{{ loop.index }}]); 
  {% else %}
  const {{TID.Type}} {{TID.Name}} = {{TID.Start}} + {{TID.Type}}({% if Kernel.UseRayGen %}gl_LaunchIDEXT{% else %}gl_GlobalInvocationID{% endif %}[{{ loop.index }}])*{{TID.Stride}}; 
  {% endif %}
  {% endfor %}
  {% endif %} {# /* Kernel.EnableBlockExpansion */ #}
  {# /*------------------------------------------------------------- BEG. INIT ------------------------------------------------------------- */ #}
  {% include "inc_exit_cond.glsl" %}
  {% if length(Kernel.SubjToRed) > 0 or length(Kernel.ArrsToRed) > 0 %}                        
  {% include "inc_reduction_init.glsl" %}
  {% endif %} 
  {# /*------------------------------------------------------------- END. INIT ------------------------------------------------------------- */ #}
  {% if Kernel.IsBoolean %}
  bool kgenExitCond = false;
  {% endif %}
  {% endif %}
  {% if Kernel.EnableBlockExpansion %}
  {% for Block in Kernel.SourceBE %}
  {% if Block.IsParallel %}
  barrier();
  {
  {{Block.Text}}
  }
  barrier();
  {% else %}
  if(gl_LocalInvocationID[0] == 0) {
  {{Block.Text}}
  }
  {% endif %}
  {% endfor %}
  {% else %}
  if(runThisThread)
  {
  {# /*------------------------------------------------------------- KERNEL SOURCE ------------------------------------------------------------- */ #}
  {{Kernel.Source}}
  {# /*------------------------------------------------------------- KERNEL SOURCE ------------------------------------------------------------- */ #}
  }
  {% endif %} {# /* EnableBlockExpansion */ #}
  {% if Kernel.HasEpilog %}
  //KGEN_EPILOG:
  {% if Kernel.IsBoolean %}
  {
    const bool exitHappened = (kgenArgs.tFlagsMask & KGEN_FLAG_SET_EXIT_NEGATIVE) != 0 ? !kgenExitCond : kgenExitCond;
    if((kgenArgs.tFlagsMask & KGEN_FLAG_DONT_SET_EXIT) == 0 && exitHappened)
      kgen_threadFlags[tid] = ((kgenArgs.tFlagsMask & KGEN_FLAG_BREAK) != 0) ? KGEN_FLAG_BREAK : KGEN_FLAG_RETURN;
  };
  {% endif %}
  {# /*------------------------------------------------------------- BEG. REDUCTION PASS ------------------------------------------------------------- */ #}
  {% if length(Kernel.SubjToRed) > 0 or length(Kernel.ArrsToRed) > 0 %}                      
  {% include "inc_reduction_inkernel.glsl" %}
  
  {% endif %}
  {# /*------------------------------------------------------------- END. REDUCTION PASS ------------------------------------------------------------- */ #}
  {% endif %} {# /* END of 'if Kernel.HasEpilog'  */ #}
}
