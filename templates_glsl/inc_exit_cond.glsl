  {% if Kernel.threadDim == 3 %}
  if({{Kernel.threadName1}} >= kgenArgs.iNumElementsX || {{Kernel.threadName2}} >= kgenArgs.iNumElementsY || {{Kernel.threadName3}} >= kgenArgs.iNumElementsZ)
    return;
  {% else if Kernel.threadDim == 2 %}
  if({{Kernel.threadName1}} >= kgenArgs.iNumElementsX || {{Kernel.threadName2}} >= kgenArgs.iNumElementsY)
    return;
  {% else %}
  if({{Kernel.threadName1}} >= kgenArgs.iNumElementsX)
    return;
  {% endif %}
  {% if Kernel.shouldCheckExitFlag %}
  if((kgen_threadFlags[{{Kernel.ThreadOffset}}] & kgenArgs.tFlagsMask) != 0) 
    return;
  {% endif %}
  