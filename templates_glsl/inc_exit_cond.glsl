  {% if Kernel.threadDim == 3 %}
  if({{Kernel.threadName1}} >= kgenArgs.{{Kernel.threadIdName1}} || {{Kernel.threadName2}} >= kgenArgs.{{Kernel.threadIdName2}} || {{Kernel.threadName3}} >= kgenArgs.{{Kernel.threadIdName3}})
    return;
  {% else if Kernel.threadDim == 2 %}
  if({{Kernel.threadName1}} >= kgenArgs.{{Kernel.threadIdName1}} || {{Kernel.threadName2}} >= kgenArgs.{{Kernel.threadIdName2}})
    return;
  {% else %}
  if({{Kernel.threadName1}} >= kgenArgs.{{Kernel.threadIdName1}})
    return;
  {% endif %}
  {% if Kernel.shouldCheckExitFlag %}
  if((kgen_threadFlags[{{Kernel.ThreadOffset}}] & kgenArgs.tFlagsMask) != 0) 
    return;
  {% endif %}
  