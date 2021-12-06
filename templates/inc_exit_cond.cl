  {% if Kernel.threadDim == 3 %}
  if({{Kernel.threadName1}} >= {{Kernel.IndirectSizeX}} + {{Kernel.CondLE1}} || {{Kernel.threadName2}} >= {{Kernel.IndirectSizeY}} + {{Kernel.CondLE2}} || {{Kernel.threadName3}} >= {{Kernel.IndirectSizeZ}} + {{Kernel.CondLE3}})
    return;
  {% else if Kernel.threadDim == 2 %}
  if({{Kernel.threadName1}} >= {{Kernel.IndirectSizeX}} + {{Kernel.CondLE1}} || {{Kernel.threadName2}} >= {{Kernel.IndirectSizeY}} + {{Kernel.CondLE2}})
    return;
  {% else %}
  if({{Kernel.threadName1}} >= {{Kernel.IndirectSizeX}} + {{Kernel.CondLE1}})
    return;
  {% endif %}
  {% if Kernel.shouldCheckExitFlag %}
  if((kgen_threadFlags[{{Kernel.ThreadOffset}}] & kgen_tFlagsMask) != 0) 
    return;
  {% endif %}