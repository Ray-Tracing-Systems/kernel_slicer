  {% if Kernel.threadDim == 3 %}
  if({{Kernel.threadName1}} >= {{Kernel.IndirectSizeX}} || {{Kernel.threadName2}} >= {{Kernel.IndirectSizeY}} || {{Kernel.threadName3}} >= {{Kernel.IndirectSizeZ}})
    return;
  {% else if Kernel.threadDim == 2 %}
  if({{Kernel.threadName1}} >= {{Kernel.IndirectSizeX}} || {{Kernel.threadName2}} >= {{Kernel.IndirectSizeY}})
    return;
  {% else %}
  if({{Kernel.threadName1}} >= {{Kernel.IndirectSizeX}})
    return;
  {% endif %}
  {% if Kernel.shouldCheckExitFlag %}
  if((kgen_threadFlags[{{Kernel.ThreadOffset}}] & kgen_tFlagsMask) != 0) 
    return;
  {% endif %}