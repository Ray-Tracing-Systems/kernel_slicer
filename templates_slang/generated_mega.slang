//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

{% include "common_generated.slang" %}

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
  {% for TID in Kernel.ThreadIds %}
  const {{TID.Type}} {{TID.Name}} = {{TID.Type}}(a_globalTID[{{ loop.index }}]); 
  {% endfor %}
  {% if UsePersistentThreads %}
  g_persistentTotalSize = kgenArgs.{{Kernel.threadSZName1}};
  {%endif%}
  {# /*------------------------------------------------------------- KERNEL SOURCE ------------------------------------------------------------- */ #}
  {{Kernel.Source}}
  {# /*------------------------------------------------------------- KERNEL SOURCE ------------------------------------------------------------- */ #}
}
