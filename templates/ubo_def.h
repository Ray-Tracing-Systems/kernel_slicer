#ifndef {{MainClassName}}_UBO_H
#define {{MainClassName}}_UBO_H

#include "OpenCLMath.h"

struct {{MainClassName}}_UBO_Data
{
## for Field in UBOStructFields  
  {{Field.Type}} {{Field.Name}}{% if Field.IsArray %}[{{Field.ArraySize}}]{% endif %};
## endfor
  unsigned int dummy_last;
{% for hierarchy in Hierarchies %}
{% if hierarchy.IndirectDispatch %}
  
  unsigned int objNum_{{hierarchy.Name}}Tst[{{hierarchy.ImplAlignedSize}}];
  unsigned int objNum_{{hierarchy.Name}}Src[{{hierarchy.ImplAlignedSize}}];  
  unsigned int objNum_{{hierarchy.Name}}Acc[{{hierarchy.ImplAlignedSize}}];
{% endif %}  
{% endfor %}
};

#endif
