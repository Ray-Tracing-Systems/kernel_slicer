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
  foreach ({% for TID in Kernel.ThreadIds %}{{TID.Name}} = {{TID.Start}} ... {{TID.Size}}{% if loop.index1 != Kernel.threadDim %}, {% endif %}{% endfor %}) { 
  {# /*------------------------------------------------------------- KERNEL SOURCE ------------------------------------------------------------- */ #}
  {{Kernel.Source}}
  }
  {# /*------------------------------------------------------------- KERNEL SOURCE ------------------------------------------------------------- */ #}
  {% for Var in Kernel.SubjToRed %}
  ubo[0].{{Var.Name}} = {{Var.Op}}({{Var.Name}});
  {% endfor %}
}

## endfor
