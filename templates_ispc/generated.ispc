/////////////////////////////////////////////////////////////////////
/////////////////// include files ///////////////////////////////////
/////////////////////////////////////////////////////////////////////
#include "include/{{UBOIncl}}"
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
  uniform struct {{MainClassName}}_UBO_Data ubo[1])
{
  {# /* declare varyings which are subjected to reduction */ #}
  {% for Var in Kernel.SubjToRed %}
  {{Var.Type}} {{Var.Name}};
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
  ubo[0].{{Var.Name}} = {{Var.Op}}({{Var.Name}});
  {% endfor %}
}
{% else %} {% if Kernel.OpenMPAndISPC %}
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
  uniform struct {{MainClassName}}_UBO_Data ubo[1],
  uniform int _start, uniform int _end, uniform int _ycoord)
{
  {# /* declare varyings which are subjected to reduction */ #}
  {% for Var in Kernel.SubjToRed %}
  {{Var.Type}} {{Var.Name}};
  {% endfor %}
  {{Kernel.InitSource}}
  {% if Kernel.threadDim == 1 %}
  foreach({{Kernel.ThreadId0.Name}} = _start ... _end) {
  {% else %}
  const uniform int {{Kernel.ThreadId1.Name}} = _ycoord;
  foreach({{Kernel.ThreadId0.Name}} = _start ... _end) {
  {% endif %}
  {{Kernel.Source}}
  }
  {% for Var in Kernel.SubjToRed %}
  atomic_{{Var.Op2}}_global(&ubo[0].{{Var.Name}}, {{Var.Op}}({{Var.Name}}));
  {% endfor %}
}
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
  uniform struct {{MainClassName}}_UBO_Data ubo[1],
  uniform int _start, uniform int _end, uniform int _span)
{
  {# /* declare varyings which are subjected to reduction */ #}
  {% for Var in Kernel.SubjToRed %}
  {{Var.Type}} {{Var.Name}};
  {% endfor %}
  {{Kernel.InitSource}}
  {% if Kernel.threadDim == 1 %}
  const uniform int xstart = _start + taskIndex * _span;
  const uniform int xend   = _start + min((uniform int)((taskIndex+1) * _span), _end);
  foreach({{Kernel.ThreadId0.Name}} = xstart ... xend) {
  {% else %}
  const uniform int ystart = _start + taskIndex * _span;
  const uniform int yend   = _start + min((uniform int)((taskIndex+1) * _span), _end);
  foreach({{Kernel.ThreadId1.Name}} = ystart ... yend, {{Kernel.ThreadId0.Name}} = {{Kernel.ThreadId0.Start}} ... {{Kernel.ThreadId0.Size}}) {
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
  uniform struct {{MainClassName}}_UBO_Data ubo[1])
{ 
  {% for TID in Kernel.ThreadIds %}
  const uniform int l_start{{loop.index}} = {{TID.Start}};
  const uniform int l_size{{loop.index}}  = {{TID.Size}};
  {% endfor %}
  {% if Kernel.threadDim == 1 %}
  const uniform int width = l_size0-l_start0;
  const uniform int span  = 1024;
  launch[width/span] {{Kernel.Name}}_Task({% for Arg in Kernel.Args %}{{Arg.Name}},{% endfor %}ubo,l_start0,l_size0,span);
  {% else %}
  const uniform int span   = 4;
  const uniform int height = l_size1-l_start1;
  launch[height/span] {{Kernel.Name}}_Task({% for Arg in Kernel.Args %}{{Arg.Name}},{% endfor %}ubo,l_start1,l_size1, span);
  {% endif %}
}
{% endif %}
{% endif %}

## endfor
