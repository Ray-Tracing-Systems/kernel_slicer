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
  // GENERATED EPILOG:
  // ...
  {% endif %} {# /* END of 'if Kernel.HasEpilog'  */ #}
}
