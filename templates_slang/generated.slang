//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

{% include "common_generated.slang" %}

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
void main(uint3 a_globalTID: SV_DispatchThreadID, uint3 a_loacalTID : SV_GroupThreadID, uniform KernelArgs kgenArgs)
{
  {% if not Kernel.EnableBlockExpansion %}
  bool runThisThread = true;
  {% endif %}
  {% if not Kernel.InitKPass %}
  {% if Kernel.EnableBlockExpansion %}
  {% for TID in Kernel.ThreadIds %}
  {% if TID.Simple %}
  const {{TID.Type}} {{TID.Name}} = {{TID.Type}}(a_globalTID[{{ loop.index }}]); 
  {% else %}
  const {{TID.Type}} {{TID.Name}} = {{TID.Start}} + {{TID.Type}}(a_globalTID[{{ loop.index }}])*{{TID.Stride}}; 
  {% endif %}
  {% endfor %}
  {% else %}
  {% for TID in Kernel.ThreadIds %}
  {% if TID.Simple %}
  const {{TID.Type}} {{TID.Name}} = {{TID.Type}}(a_globalTID[{{ loop.index }}]); 
  {% else %}
  const {{TID.Type}} {{TID.Name}} = {{TID.Start}} + {{TID.Type}}(a_globalTID[{{ loop.index }}])*{{TID.Stride}}; 
  {% endif %}
  {% endfor %}
  {% endif %} {# /* Kernel.EnableBlockExpansion */ #}
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
  if(a_loacalTID[0] == 0) {
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
  // GENERATED EPILOG:
  // ...
  {% endif %} {# /* END of 'if Kernel.HasEpilog'  */ #}
}
