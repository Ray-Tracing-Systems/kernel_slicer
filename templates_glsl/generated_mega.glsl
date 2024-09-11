#version 460
#extension GL_GOOGLE_include_directive : require
{% if length(Kernel.RTXNames) > 0 %}
{% if Kernel.UseRayGen %}
#extension GL_EXT_ray_tracing : require 
{% if Kernel.UseMotionBlur %}
#extension GL_NV_ray_tracing_motion_blur : require
{% endif %}
{% else %}
#extension GL_EXT_ray_query : require
{% endif %}
{% endif %}
{% if Kernel.NeedTexArray %}
#extension GL_EXT_nonuniform_qualifier : require
{% endif %}
{% if HasAllRefs %}
#extension GL_EXT_buffer_reference : require
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

///////////////////////////////////////////////////////////////// subkernels here
{% for sub in Kernel.Subkernels %}
{{sub.Decl}}
{
  {{sub.Source}}
}

{% endfor %}
///////////////////////////////////////////////////////////////// subkernels here

void main()
{
  ///////////////////////////////////////////////////////////////// prolog
  {% for TID in Kernel.ThreadIds %}
  const {{TID.Type}} {{TID.Name}} = {{TID.Type}}({% if Kernel.UseRayGen %}gl_LaunchIDEXT{% else %}gl_GlobalInvocationID{% endif %}[{{ loop.index }}]); 
  {% endfor %}
  ///////////////////////////////////////////////////////////////// prolog

  {# /*------------------------------------------------------------- KERNEL SOURCE ------------------------------------------------------------- */ #}
  {{Kernel.Source}}
  {# /*------------------------------------------------------------- KERNEL SOURCE ------------------------------------------------------------- */ #}
}
