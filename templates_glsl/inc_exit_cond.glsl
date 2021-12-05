  {% if Kernel.IsIndirect %}
  {% if Kernel.threadDim == 3 %}
  if({{Kernel.threadName1}} >= {{Kernel.IndirectSizeX}} + {{Kernel.CondLE1}} || {{Kernel.threadName2}} >= {{Kernel.IndirectSizeY}} + {{Kernel.CondLE2}} || {{Kernel.threadName3}} >= {{Kernel.IndirectSizeY}} + {{Kernel.CondLE3}})
    return;
  {% else if Kernel.threadDim == 2 %}
  if({{Kernel.threadName1}} >= {{Kernel.IndirectSizeX}} + {{Kernel.CondLE1}} || {{Kernel.threadName2}} >= {{Kernel.IndirectSizeY}} + {{Kernel.CondLE2}})
    return;
  {% else %}
  if({{Kernel.threadName1}} >= {{Kernel.IndirectSizeX}} + {{Kernel.CondLE1}})
    return;
  {% endif %}
  {% if Kernel.shouldCheckExitFlag %}
  if((kgen_threadFlags[{{Kernel.ThreadOffset}}] & kgenArgs.tFlagsMask) != 0) 
    return;
  {% endif %}
  {% else %}
  {% if Kernel.threadDim == 3 %}
  if({{Kernel.threadName1}} >= kgenArgs.{{Kernel.threadSZName1}} + {{Kernel.CondLE1}} || {{Kernel.threadName2}} >= kgenArgs.{{Kernel.threadSZName2}} + {{Kernel.CondLE2}} || {{Kernel.threadName3}} >= kgenArgs.{{Kernel.threadSZName3}} + {{Kernel.CondLE3}})
    return;
  {% else if Kernel.threadDim == 2 %}
  if({{Kernel.threadName1}} >= kgenArgs.{{Kernel.threadSZName1}} + {{Kernel.CondLE1}} || {{Kernel.threadName2}} >= kgenArgs.{{Kernel.threadSZName2}} + {{Kernel.CondLE2}})
    return;
  {% else %}
  if({{Kernel.threadName1}} >= kgenArgs.{{Kernel.threadSZName1}} + {{Kernel.CondLE1}})
    return;
  {% endif %}
  {% if Kernel.shouldCheckExitFlag %}
  if((kgen_threadFlags[{{Kernel.ThreadOffset}}] & kgenArgs.tFlagsMask) != 0) 
    return;
  {% endif %}
  {% endif %} {# /* if Kernel.IsIndirect */ #}
  