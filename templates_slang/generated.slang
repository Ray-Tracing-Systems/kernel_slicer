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

{% if not Kernel.InitKPass %}
{% for redvar in Kernel.SubjToRed %}
groupshared {{redvar.Type}} {{redvar.Name}}Shared[{{Kernel.WGSizeX}}*{{Kernel.WGSizeY}}*{{Kernel.WGSizeZ}}]; 
{% endfor %}
{% for redvar in Kernel.ArrsToRed %}
groupshared {{redvar.Type}} {{redvar.Name}}Shared[{{redvar.ArraySize}}][{{Kernel.WGSizeX}}*{{Kernel.WGSizeY}}*{{Kernel.WGSizeZ}}]; 
{% endfor %}
{% endif %}
{% if Kernel.EnableBlockExpansion %}
{% for TID in Kernel.ThreadSizeBE %}
const {{TID.Type}} {{TID.Name}} = {{TID.Value}}; 
{% endfor %}
{% for Var in Kernel.SharedBE %}
groupshared {{Var}}
{% endfor %}
{% endif %}

{% for Fun in Kernel.TemplatedFun %}
{% if Fun.NameT == "ReduceAdd" %}
groupshared {{Fun.Type0}} shared_data[256];
void {{Fun.Name}}(RWStructuredBuffer<{{Fun.Type0}}> a_buffer, uint a_offset, {{Fun.Type0}} value, uint threadIdx)
{
  if (isnan(value) || isinf(value))
    value = 0.0f;

  shared_data[threadIdx] = value;
  GroupMemoryBarrierWithGroupSync();

  if (threadIdx < 128) shared_data[threadIdx] += shared_data[threadIdx + 128]; GroupMemoryBarrierWithGroupSync();
  if (threadIdx < 64)  shared_data[threadIdx] += shared_data[threadIdx + 64];  GroupMemoryBarrierWithGroupSync();
  if (threadIdx < 32)  shared_data[threadIdx] += shared_data[threadIdx + 32];  GroupMemoryBarrierWithGroupSync();
  if (threadIdx < 16)  shared_data[threadIdx] += shared_data[threadIdx + 16];  GroupMemoryBarrierWithGroupSync();
  if (threadIdx < 8)   shared_data[threadIdx] += shared_data[threadIdx + 8];   GroupMemoryBarrierWithGroupSync();
  if (threadIdx < 4)   shared_data[threadIdx] += shared_data[threadIdx + 4];   GroupMemoryBarrierWithGroupSync();
  if (threadIdx < 2)   shared_data[threadIdx] += shared_data[threadIdx + 2];   GroupMemoryBarrierWithGroupSync();
  if (threadIdx < 1)   shared_data[threadIdx] += shared_data[threadIdx + 1];   GroupMemoryBarrierWithGroupSync();
  
  if (threadIdx == 0)
    InterlockedAdd(a_buffer[a_offset], shared_data[0]);
}
{% endif %}
{% endfor %}

[shader("compute")]
[numthreads({{Kernel.WGSizeX}}, {{Kernel.WGSizeY}}, {{Kernel.WGSizeZ}})]
void main(uint3 a_globalTID: SV_DispatchThreadID, {% if Kernel.HasEpilog %} uint3 a_groupID : SV_GroupID, {% endif%} uint3 a_localTID : SV_GroupThreadID, uniform KernelArgs kgenArgs)
{
  {% if not Kernel.EnableBlockExpansion and not Kernel.IsSingleThreaded %}
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
  {% include "inc_exit_cond.slang" %}
  {% if length(Kernel.SubjToRed) > 0 or length(Kernel.ArrsToRed) > 0 %}                        
  {% include "inc_red_init.slang" %}
  {% endif %} 
  {% endif %}
  {% if Kernel.EnableBlockExpansion %}
  {% for Block in Kernel.SourceBE %}
  {% if Block.IsParallel %}
  GroupMemoryBarrierWithGroupSync();
  // BLOCK-EXPANDED KERNEL BODY:
  //
  {
  {{Block.Text}}
  }
  GroupMemoryBarrierWithGroupSync();
  {% else %}
  if(a_loacalTID[0] == 0) {
  {{Block.Text}}
  }
  {% endif %}
  {% endfor %}
  {% else %}
  // KERNEL BODY:
  //
  {% if not Kernel.IsSingleThreaded %}
  if(runThisThread)
  {
  {% endif %}  
  {# /*------------------------------------------------------------- KERNEL SOURCE ------------------------------------------------------------- */ #}
  {{Kernel.Source}}
  {# /*------------------------------------------------------------- KERNEL SOURCE ------------------------------------------------------------- */ #}
  {% if not Kernel.IsSingleThreaded %}
  }
  {% endif %}  
  {% endif %} {# /* EnableBlockExpansion */ #}
  {% if Kernel.HasEpilog %}
  // GENERATED EPILOG:
  //
  {% if length(Kernel.SubjToRed) > 0 or length(Kernel.ArrsToRed) > 0 %}                      
  {% include "inc_red_finish.slang" %}
  {% endif %}
  
  {% endif %} {# /* END of 'if Kernel.HasEpilog'  */ #}
}
