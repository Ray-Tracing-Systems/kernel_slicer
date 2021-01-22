#ifndef {{MainClassName}}_UBO_H
#define {{MainClassName}}_UBO_H

struct {{MainClassName}}_UBO_Data
{
## for Field in UBOStructFields  
  {{Field.Type}} {{Field.Name}}{% if Field.IsArray %}[{{Field.ArraySize}}]{% endif %};
## endfor
  unsigned int dummy_last;
};

#endif
