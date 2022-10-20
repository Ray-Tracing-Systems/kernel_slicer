#include <vector>
#include <memory>
#include <string>
#include <unordered_map>

{{Includes}}

## for Decl in ClassDecls  
{% if Decl.InClass and Decl.IsType %}
using {{Decl.Type}} = {{MainClassName}}::{{Decl.Type}}; // for passing this data type to UBO
{% endif %}
## endfor

#include "include/{{UBOIncl}}"

class {{MainClassName}}_ISPC : public {{MainClassName}}
{
public:

  {% for ctorDecl in Constructors %}
  {% if ctorDecl.NumParams == 0 %}
  {{ctorDecl.ClassName}}_ISPC() {}
  {% else %}
  {{ctorDecl.ClassName}}_ISPC({{ctorDecl.Params}}) : {{ctorDecl.ClassName}}({{ctorDecl.PrevCall}}) {}
  {% endif %}
  {% endfor %}

  virtual ~{{MainClassName}}_ISPC();

  {% if HasCommitDeviceFunc %}
  void CommitDeviceData() override // you have to define this virtual function in the original imput class
  {
    
  }  
  {% endif %}
  {% if HasGetTimeFunc %}
  void GetExecutionTime(const char* a_funcName, float a_out[4]) override {} // TODO: implement it 
  {% endif %}

protected:

  {% for Kernel in Kernels %}
  void {{Kernel.Name}}({% for Arg in Kernel.Args %}
  {% if not Arg.IsUBO %} 
  {{Arg.Type}} {{Arg.Name}},
  {% endif %}
  {% endfor %}
  {% for UserArg in Kernel.UserArgs %}
  const {{UserArg.Type}} {{UserArg.Name}},
  {% endfor %}) override { {{Kernel.Name}}_ISPC(...); }
  {% endfor %}

  {{MainClassName}}_UBO_Data m_uboData;
};
