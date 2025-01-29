//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

{% include "common_generated.slang" %}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
{% if not Kernel.UseRayGen %}
layout(local_size_x = {{Kernel.WGSizeX}}, local_size_y = {{Kernel.WGSizeY}}, local_size_z = {{Kernel.WGSizeZ}}) in;
{% endif %}
struct KernelArgs
{
  {% for UserArg in Kernel.UserArgs %} 
  {{UserArg.Type}} {{UserArg.Name}};
  {% endfor %}
  {{Kernel.threadSZType1}} {{Kernel.threadSZName1}}; 
  {{Kernel.threadSZType2}} {{Kernel.threadSZName2}}; 
  {{Kernel.threadSZType3}} {{Kernel.threadSZName3}}; 
  uint tFlagsMask;    
};

///////////////////////////////////////////////////////////////// subkernels here
{% for sub in Kernel.Subkernels %}
{{sub.Decl}}
{
  {{sub.Source}}
}

{% endfor %}
///////////////////////////////////////////////////////////////// subkernels here

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct KernelArgs
{
  {% for UserArg in Kernel.UserArgs %} 
  {{UserArg.Type}} {{UserArg.Name}};
  {% endfor %}
  {{Kernel.threadSZType1}} {{Kernel.threadSZName1}}; 
  {{Kernel.threadSZType2}} {{Kernel.threadSZName2}}; 
  {{Kernel.threadSZType3}} {{Kernel.threadSZName3}}; 
  uint tFlagsMask;    
};

[shader("compute")]
[numthreads({{Kernel.WGSizeX}}, {{Kernel.WGSizeY}}, {{Kernel.WGSizeZ}})]
void main(uint3 a_globalTID: SV_DispatchThreadID, uniform KernelArgs kgenArgs)
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
