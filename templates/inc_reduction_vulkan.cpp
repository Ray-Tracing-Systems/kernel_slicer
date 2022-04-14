  ///// complete kernel with reduction passes
  {
    VkBufferMemoryBarrier barUBO = BarrierForSingleBuffer(m_classDataBuffer);
    VkBufferMemoryBarrier redBars   [{{Kernel.RedVarsFPNum}}]; 
    VkBuffer              redBuffers[{{Kernel.RedVarsFPNum}}+1] = { {% for RedVarName in Kernel.RedVarsFPArr %}m_vdata.{{RedVarName.Name}}Buffer, {% endfor %} VK_NULL_HANDLE};
    size_t                szOfElems [{{Kernel.RedVarsFPNum}}+1] = { {% for RedVarName in Kernel.RedVarsFPArr %}sizeof({{RedVarName.Type}}), {% endfor %} 0};
    BarriersForSeveralBuffers(redBuffers, redBars, {{Kernel.RedVarsFPNum}});
    
    vkCmdPipelineBarrier(m_currCmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, {{Kernel.RedVarsFPNum}}, redBars, 0, nullptr);
    vkCmdBindPipeline(m_currCmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, {{Kernel.Name}}ReductionPipeline);
    
    {% if Kernel.threadDim == 1 %}
    uint32_t oldSize    = {{Kernel.tidX}};
    uint32_t wholeSize  = (oldSize + blockSizeX - 1) / blockSizeX; // assume first pass of reduction is done inside kernel itself
    {% else %}
    uint32_t oldSize    = {{Kernel.tidX}}*{{Kernel.tidY}};
    uint32_t wholeSize  = (oldSize + blockSizeX*blockSizeY - 1) / (blockSizeX*blockSizeY); // assume first pass of reduction is done inside kernel itself
    {% endif %}
    uint32_t wgSize     = REDUCTION_BLOCK_SIZE;
    uint32_t currOffset = 0;
    while (wholeSize > 1)
    {
      uint32_t nextSize = (wholeSize + wgSize - 1) / wgSize;
      pcData.m_sizeX  = oldSize;                // put current size here
      pcData.m_sizeY  = currOffset;             // put input offset here
      pcData.m_sizeZ  = currOffset + wholeSize; // put output offet here
      pcData.m_tFlags = m_currThreadFlags;      // now flags:
      if(wholeSize <= wgSize)                   // stop if last pass
        pcData.m_tFlags |= KGEN_REDUCTION_LAST_STEP;

      {% if UseSeparateUBO %}
      {
        vkCmdUpdateBuffer(m_currCmdBuffer, m_uboArgsBuffer, 0, sizeof(KernelArgsPC), &pcData);
        VkBufferMemoryBarrier barUBO2 = BarrierForArgsUBO(sizeof(KernelArgsPC));
        vkCmdPipelineBarrier(m_currCmdBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &barUBO2, 0, nullptr);
      }
      {% else %}
      vkCmdPushConstants(m_currCmdBuffer, {{Kernel.Name}}Layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(KernelArgsPC), &pcData);
      {% endif %} 
      vkCmdDispatch(m_currCmdBuffer, nextSize, 1, 1);
      
      if(wholeSize <= wgSize)
        vkCmdPipelineBarrier(m_currCmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &barUBO, 0, nullptr);
      else
      {
        uint32_t arrSize = sizeof(redBars)/sizeof(redBars[0]);
        for(int barId=0;barId<arrSize;barId++)
        {
          redBars[barId].offset = pcData.m_sizeZ*szOfElems[barId]; // put output offset here (for barrier)
          redBars[barId].size   = nextSize;                        // put output data size (for barrier)
        }
        vkCmdPipelineBarrier(m_currCmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, {{Kernel.RedVarsFPNum}}, redBars, 0, nullptr);
      } 

      currOffset += wholeSize;
      oldSize    =  wholeSize;
      wholeSize  =  nextSize;
    }
  }