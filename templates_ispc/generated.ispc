/////////////////////////////////////////////////////////////////////
/////////////////// include files ///////////////////////////////////
/////////////////////////////////////////////////////////////////////
#include "LiteMathISPC.h"

struct {{MainClassName}}{{MainClassSuffix}}_UBO_Data
{
  {% for Field in UBO.UBOStructFields %}
  {% if Field.IsDummy %} 
  uint {{Field.Name}}; 
  {% else %}
  {{Field.Type}} {{Field.Name}}{% if Field.IsArray %}[{{Field.ArraySize}}]{% endif %}; 
  {% endif %}
  {% endfor %}
  uint dummy_last;
};

## for Incl in Includes  
#include "{{Incl}}"
## endfor

/////////////////////////////////////////////////////////////////////
/////////////////// declarations in class ///////////////////////////
/////////////////////////////////////////////////////////////////////
## for Decl in ClassDecls  
{{Decl.Text}}
## endfor

/////////////////////////////////////////////////////////////////////
/////////////////// local functions /////////////////////////////////
/////////////////////////////////////////////////////////////////////

## for LocalFunc in LocalFunctions  
{{LocalFunc}}

## endfor

/////////////////////////////////////////////////////////////////////
/////////////////// kernels /////////////////////////////////////////
/////////////////////////////////////////////////////////////////////

## for Kernel in Kernels
{% if Kernel.SingleThreadISPC %}
export void {{Kernel.Name}}_ISPC(
  {% for Arg in Kernel.Args %}
  {% if not Arg.IsUBO %} 
  {% if Arg.IsPointer %}
  uniform {{Arg.Type}} {{Arg.Name}}[], 
  {% else %}
  uniform {{Arg.Type}} {{Arg.Name}},
  {% endif %}
  {% endif %}
  {% endfor %}
  uniform struct {{MainClassName}}{{MainClassSuffix}}_UBO_Data ubo[1])
{
  {# /* declare varyings which are subjected to reduction, they are initialized with neutral element of reduction */ #}
  {% for Var in Kernel.SubjToRed %}
  {{Var.Type}} {{Var.Name}} = {{Var.Init}};
  {% endfor %}
  {{Kernel.InitSource}}
  {% if Kernel.ExplicitIdISPC %}
  {% if Kernel.threadDim == 1 %}
  for (uniform {{Kernel.ThreadId0.Type}} {{Kernel.ThreadId0.Name}} = {{Kernel.ThreadId0.Start}}; {{Kernel.ThreadId0.Name}} < {{Kernel.ThreadId0.Size}}; {{Kernel.ThreadId0.Name}} += programCount) {
  {% else %}
  for (uniform {{Kernel.ThreadId1.Type}} {{Kernel.ThreadId1.Name}} = {{Kernel.ThreadId1.Start}}; {{Kernel.ThreadId1.Name}} < {{Kernel.ThreadId1.Size}}; {{Kernel.ThreadId1.Name}} ++) {
  for (uniform {{Kernel.ThreadId0.Type}} {{Kernel.ThreadId0.Name}} = {{Kernel.ThreadId0.Start}}; {{Kernel.ThreadId0.Name}} < {{Kernel.ThreadId0.Size}}; {{Kernel.ThreadId0.Name}} += programCount) {
  {% endif %}
  {% else %}
  foreach ({% for TID in Kernel.ThreadIds %}{{TID.Name}} = {{TID.Start}} ... {{TID.Size}}{% if loop.index1 != Kernel.threadDim %}, {% endif %}{% endfor %}) { 
  {% endif %}
  {{Kernel.Source}}
  {% if Kernel.ExplicitIdISPC and Kernel.threadDim == 2 %}
  }}
  {% else %}
  }
  {% endif %}
  {% for Var in Kernel.SubjToRed %}
  {% if Var.Op2 == "add" %}
  ubo[0].{{Var.Name}} += {{Var.Op}}({{Var.Name}});
  {% else %}
  ubo[0].{{Var.Name}} = {{Var.Op2}}(ubo[0].{{Var.Name}}, {{Var.Op}}({{Var.Name}}));
  {% endif %}
  {% endfor %}
  {{Kernel.FinishSource}}
}
{% else %} {% if Kernel.OpenMPAndISPC %}
{% if Kernel.HasInitISPC %}
export void {{Kernel.Name}}_Init_ISPC( // executed once before parallel loop
  {% for Arg in Kernel.Args %}
  {% if not Arg.IsUBO %} 
  {% if Arg.IsPointer %}
  uniform {{Arg.Type}} {{Arg.Name}}[], 
  {% else %}
  uniform {{Arg.Type}} {{Arg.Name}},
  {% endif %}
  {% endif %}
  {% endfor %}
  uniform struct {{MainClassName}}{{MainClassSuffix}}_UBO_Data ubo[1])
{
  {{Kernel.InitSource}}
}

{% endif %}
export void {{Kernel.Name}}_ISPC(
  {% for Arg in Kernel.Args %}
  {% if not Arg.IsUBO %} 
  {% if Arg.IsPointer %}
  uniform {{Arg.Type}} {{Arg.Name}}[], 
  {% else %}
  uniform {{Arg.Type}} {{Arg.Name}},
  {% endif %}
  {% endif %}
  {% endfor %}
  uniform struct {{MainClassName}}{{MainClassSuffix}}_UBO_Data ubo[1],
  uniform int _start, uniform int _end, uniform int _ycoord)
{
  {# /* declare varyings which are subjected to reduction, they are initialized with neutral element of reduction */ #}
  {% for Var in Kernel.SubjToRed %}
  {{Var.Type}} {{Var.Name}} = {{Var.Init}};
  {% endfor %}
  {% if Kernel.threadDim == 1 %}
  foreach({{Kernel.ThreadId0.Name}} = _start ... _end) {
  {% else %}
  const uniform int {{Kernel.ThreadId0.Name}} = _ycoord;
  foreach({{Kernel.ThreadId1.Name}} = _start ... _end) {
  {% endif %}
  {{Kernel.Source}}
  }
  {% for Var in Kernel.SubjToRed %}
  atomic_{{Var.Op2}}_global(&ubo[0].{{Var.Name}}, {{Var.Op}}({{Var.Name}}));
  {% endfor %}
}
{% if Kernel.HasFinishISPC %}

export void {{Kernel.Name}}_Finish_ISPC( // executed once after parallel loop
  {% for Arg in Kernel.Args %}
  {% if not Arg.IsUBO %} 
  {% if Arg.IsPointer %}
  uniform {{Arg.Type}} {{Arg.Name}}[], 
  {% else %}
  uniform {{Arg.Type}} {{Arg.Name}},
  {% endif %}
  {% endif %}
  {% endfor %}
  uniform struct {{MainClassName}}{{MainClassSuffix}}_UBO_Data ubo[1])
{
  {{Kernel.FinishSource}}
}
{% endif %}
{% else %}
task void {{Kernel.Name}}_Task(
  {% for Arg in Kernel.Args %}
  {% if not Arg.IsUBO %} 
  {% if Arg.IsPointer %}
  uniform {{Arg.Type}} {{Arg.Name}}[], 
  {% else %}
  uniform {{Arg.Type}} {{Arg.Name}},
  {% endif %}
  {% endif %}
  {% endfor %}
  uniform struct {{MainClassName}}{{MainClassSuffix}}_UBO_Data ubo[1],
  uniform int _start, uniform int _end, uniform int _span)
{
  {# /* declare varyings which are subjected to reduction, they are initialized with neutral element of reduction */ #}
  {% for Var in Kernel.SubjToRed %}
  {{Var.Type}} {{Var.Name}} = {{Var.Init}};
  {% endfor %}
  {% if Kernel.threadDim == 1 %}
  const uniform int xstart = _start + taskIndex * _span;
  const uniform int xend   = _start + min((uniform int)((taskIndex+1) * _span), _end);
  foreach({{Kernel.ThreadId0.Name}} = xstart ... xend) {
  {% else %}
  const uniform int ystart = _start + taskIndex * _span;
  const uniform int yend   = _start + min((uniform int)((taskIndex+1) * _span), _end);
  foreach({{Kernel.ThreadId0.Name}} = ystart ... yend, {{Kernel.ThreadId1.Name}} = {{Kernel.ThreadId1.Start}} ... {{Kernel.ThreadId1.Size}}) {
  {% endif %}
  {{Kernel.Source}}
  }
  {% for Var in Kernel.SubjToRed %}
  atomic_{{Var.Op2}}_global(&ubo[0].{{Var.Name}}, {{Var.Op}}({{Var.Name}}));
  {% endfor %}
}

export void {{Kernel.Name}}_ISPC(
  {% for Arg in Kernel.Args %}
  {% if not Arg.IsUBO %} 
  {% if Arg.IsPointer %}
  uniform {{Arg.Type}} {{Arg.Name}}[], 
  {% else %}
  uniform {{Arg.Type}} {{Arg.Name}},
  {% endif %}
  {% endif %}
  {% endfor %}
  uniform struct {{MainClassName}}{{MainClassSuffix}}_UBO_Data ubo[1])
{ 
  {{Kernel.InitSource}}
  {% for TID in Kernel.ThreadIds %}
  const uniform int l_start{{loop.index}} = {{TID.Start}};
  const uniform int l_size{{loop.index}}  = {{TID.Size}};
  {% endfor %}
  {% if Kernel.threadDim == 1 %}
  const uniform int span   = 1024;
  {% else %}
  const uniform int span   = 4;
  {% endif %}
  const uniform int length = l_size0-l_start0;
  launch[(length + span - 1)/span] {{Kernel.Name}}_Task({% for Arg in Kernel.Args %}{{Arg.Name}},{% endfor %}ubo,l_start0,l_size0,span);
  sync;
  {{Kernel.FinishSource}}
}
{% endif %}
{% endif %}

## endfor
