  // graphics pipeline (LiteRS): {% if GPDynamicRendering %}graphics pipelines for draws, Vulkan 1.3 dynamic rendering{% else %}render passes for LiteRS::Pipeline members and graphics pipelines for draws, Vulkan 1.0{% endif %}
  //
  struct GPPassObjects
  {
    VkRenderPass  renderPass  = VK_NULL_HANDLE;
    VkFramebuffer framebuffer = VK_NULL_HANDLE;
    uint32_t      width       = 0;
    uint32_t      height      = 0;
  };
  {% for Pipe in GPPipelines %}
  GPPassObjects m_gp_{{Pipe.Name}};
  virtual void Clear_{{Pipe.Name}}Cmd(VkCommandBuffer a_commandBuffer);
  {% endfor %}
  {% for Draw in GPDraws %}
  struct {{Draw.Name}}_PC // push constants of vertex shader '{{Draw.VSOriginal}}'
  {
    {% for Arg in Draw.UserArgs %}
    {{Arg.Type}} {{Arg.Name}};
    {% endfor %}
    uint32_t gpDummy;
  };
  VkPipelineLayout {{Draw.Name}}Layout   = VK_NULL_HANDLE;
  VkPipeline       {{Draw.Name}}Pipeline = VK_NULL_HANDLE;
  virtual void {{Draw.Name}}Cmd(VkCommandBuffer a_commandBuffer, VkDescriptorSet a_dsVS, VkDescriptorSet a_dsPS, {% if Draw.Indexed %}VkBuffer a_indexBuffer, {% endif %}uint32_t a_count{% if Draw.Instanced %}, uint32_t a_instNum{% endif %}{% for Arg in Draw.UserArgs %}, {{Arg.Type}} {{Arg.Name}}{% endfor %});
  {% endfor %}
  VkMemoryBarrier m_gpBarrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT };
  virtual void InitGraphics();    ///< creates render passes, framebuffers and graphics pipelines; called by CommitDeviceData()
  virtual void DestroyGraphics();
  void InitGraphicsTargets();
  void DestroyGraphicsTargets();
  {% for Target in GPRenderTargets %}
  VkDeviceMemory m_gpMemory_{{Target.Name}} = VK_NULL_HANDLE;
  VkExtent2D m_gpSize_{{Target.Name}} = {};
  {% endfor %}
