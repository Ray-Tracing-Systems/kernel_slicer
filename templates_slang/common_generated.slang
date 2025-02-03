#include "common{{MainClassSuffixLowerCase}}_slang.h"
{% for KSpec in Kernel.SpecConstants %}
layout (constant_id = {{KSpec.Id}}) const int {{KSpec.Name}} = {{KSpec.Id}}; // 
{% endfor %}

{% for Arg in Kernel.Args %}
{% if not Arg.IsUBO %} 
{% if Arg.IsImage %}
RWTexture2D<Arg.Type> {{Arg.Name}}; // TODO: this is Ot a combined image sampler (!!!)
{% else if Arg.IsAccelStruct %}
layout(binding = {{loop.index}}, set = 0) uniform accelerationStructureEXT {{Arg.Name}};
{% else %}
RWStructuredBuffer<{{Arg.Type}}> {{Arg.Name}}; // 
{% endif %} {# /* Arg.IsImage */ #}
{% endif %} {# /* not Arg.IsUBO */ #}
{% endfor %}
RWStructuredBuffer<{{MainClassName}}{{MainClassSuffix}}_UBO_Data> ubo;

{% for Array in Kernel.ThreadLocalArrays %}
{{Array.Type}} {{Array.Name}}[{{Array.Size}}];
{% endfor %}
{% for MembFunc in Kernel.MemberFunctions %}
{{MembFunc.Decl}};

{% endfor %}
{% for MembFunc in Kernel.MemberFunctions %}
{% if not (MembFunc.IsRayQuery and (Kernel.UseRayGen or (length(Kernel.IntersectionHierarhcy.Implementations) >= 1)) ) %}

{{MembFunc.Text}}
{% endif%}
{% endfor %} 
