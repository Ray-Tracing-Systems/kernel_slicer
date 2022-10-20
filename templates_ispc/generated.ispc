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
  {% for UserArg in Kernel.UserArgs %}
  uniform const {{UserArg.Type}} {{UserArg.Name}},
  {% endfor %}
  uniform const uint {{Kernel.threadSZName1}}, 
  uniform const uint {{Kernel.threadSZName2}},
  uniform const uint {{Kernel.threadSZName3}},
  uniform struct {{MainClassName}}_UBO_Data ubo[1])
{
  {# /* declare varyings which are subjected to reduction */ #}
  {% for Var in Kernel.SubjToRed %}
  {{Var.Type}} {{Var.Name}};
  {% endfor %}
  {{Kernel.InitSource}}
  {% for TID in Kernel.ThreadIds %}
  {% if TID.Simple %}
  foreach ({{TID.Name}} = {{TID.Start}} ... {{TID.Size}}) {
  {% else %}
  const {{TID.Type}} {{TID.Name}} = {{TID.Start}} + ({{TID.Type}})(get_global_id({{ loop.index }}))*{{TID.Stride}};
  foreach ({{TID.Name}}_L = {{TID.Start}} ... {{TID.Size}}) { 
  const {{TID.Type}} {{TID.Name}} = {{TID.Name}}_L*{{TID.Stride}};  
  {% endif %}
  {% endfor %}
  {# /*------------------------------------------------------------- KERNEL SOURCE ------------------------------------------------------------- */ #}
  {{Kernel.Source}}
  {% for TID in Kernel.ThreadIds %}
  }
  {% endfor %}
  {# /*------------------------------------------------------------- KERNEL SOURCE ------------------------------------------------------------- */ #}
  {% for Var in Kernel.SubjToRed %}
  ubo[0].{{Var.Name}} = {{Var.Op}}({{Var.Name}});
  {% endfor %}
}

## endfor
