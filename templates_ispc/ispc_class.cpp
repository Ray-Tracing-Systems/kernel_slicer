#include <vector>
#include <memory>
#include <string>
#include <unordered_map>

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
  ~{{MainClassName}}_ISPC(){}

  {% if HasCommitDeviceFunc %}
  void CommitDeviceData() override 
  {
    UpdatePlainMembers();
  }  
  {% endif %}
  {% if HasGetTimeFunc %}
  void GetExecutionTime(const char* a_funcName, float a_out[4]) override {} // TODO: implement it 
  {% endif %}

  {% for MainFunc in MainFunctions %}
  {{MainFunc.ReturnType}} {{MainFunc.DeclOrig}} override;
  {% endfor %}

protected:

  virtual void UpdatePlainMembers();
  virtual void ReadPlainMembers();

  {% for Kernel in Kernels %}
  void {{Kernel.Name}}({% for Arg in Kernel.Args %}{% if not Arg.IsUBO %} {% if Arg.IsPointer %}{{Arg.Type}}* {{Arg.Name}}{% if loop.index1 != Kernel.LastArgNF %}, {% endif %}{% else %}{{Arg.Type}} {{Arg.Name}}{% if loop.index1 != Kernel.LastArgNF %}, {% endif %} {% endif %}{% endif %} {% endfor %}
  {% for UserArg in Kernel.UserArgs %}const {{UserArg.Type}} {{UserArg.Name}},{% endfor %}) override { ispc::{{Kernel.Name}}_ISPC({% for Arg in Kernel.Args %}{{Arg.Name}}{% if loop.index1 != Kernel.LastArgNF %},{% endif %}{% endfor %},&m_uboData); }
  {% endfor %}

  ispc::{{MainClassName}}_UBO_Data m_uboData;
};

void {{MainClassName}}_ISPC::UpdatePlainMembers()
{
  const size_t maxAllowedSize = std::numeric_limits<uint32_t>::max();
## for Var in ClassVars
  {% if Var.IsArray %}
  memcpy(m_uboData.{{Var.Name}},{{Var.Name}},sizeof({{Var.Name}}));
  {% else %}
  m_uboData.{{Var.Name}} = {{Var.Name}};
  {% endif %}
## endfor
## for Var in ClassVectorVars 
  m_uboData.{{Var.Name}}_size     = uint32_t( {{Var.Name}}.size() );    assert( {{Var.Name}}.size() < maxAllowedSize );
  m_uboData.{{Var.Name}}_capacity = uint32_t( {{Var.Name}}.capacity() ); assert( {{Var.Name}}.capacity() < maxAllowedSize );
## endfor
}

void {{MainClassName}}_ISPC::ReadPlainMembers()
{
  {% for Var in ClassVars %}
  {% if not Var.IsConst %}
  {% if Var.IsArray %}
  memcpy({{Var.Name}}, m_uboData.{{Var.Name}}, sizeof({{Var.Name}}));
  {% else %}
  {{Var.Name}} = m_uboData.{{Var.Name}};
  {% endif %}
  {% endif %} {#/* end of if not var.IsConst */#}
  {% endfor %}
  {% for Var in ClassVectorVars %}
  {{Var.Name}}.resize(m_uboData.{{Var.Name}}_size);
  {% endfor %}
}

{% for MainFunc in MainFunctions %}
{{MainFunc.ReturnType}} {{MainClassName}}_ISPC::{{MainFunc.DeclOrig}}
{
  UpdatePlainMembers();
  {{MainClassName}}::{{MainFunc.Name}}({% for DS in MainFunc.DescriptorSets %}{% for Arg in DS.Args %}{{Arg.NameOriginal}}{% if loop.index1 != DS.ArgNumber %},{% endif %}{% endfor %}{% endfor %});
  ReadPlainMembers();
}
{% endfor %}