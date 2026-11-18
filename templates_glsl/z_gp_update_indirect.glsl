#version 460
#extension GL_GOOGLE_include_directive : require
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Graphics pipeline (LiteRS): indirect draws. Counts of these draws read data of the class, so they are computed on GPU right
// before the draw. Every call of a draw command has its own command in 'indirectBuffer' (8 uints), see GPIndirectDraw.
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

{% include "inc_ubo.glsl" %}

layout(binding = 0, set = 0) buffer dataUBO { {{MainClassName}}{{MainClassSuffix}}_UBO_Data ubo; };
layout(binding = 1, set = 0) buffer dataInd { uint indirectBuffer[]; };

layout( push_constant ) uniform UpdateArgs
{
  uint drawId;
  uint hostCount;    // counts computed on host (arguments and local variables of control function)
  uint hostInstNum;
} a_args;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

void main()
{
  uint count   = a_args.hostCount;
  uint instNum = a_args.hostInstNum;
  switch(a_args.drawId)
  {
    {% for Draw in GPIndirectDraws %}
    case {{Draw.Id}}:
      {% if Draw.CountExpr != "" %}
      count   = {{Draw.CountExpr}};
      {% endif %}
      {% if Draw.InstExpr != "" %}
      instNum = {{Draw.InstExpr}};
      {% endif %}
      break;
    {% endfor %}
    default:
      break;
  }

  // VkDrawIndexedIndirectCommand: indexCount, instanceCount, firstIndex, vertexOffset, firstInstance;
  // VkDrawIndirectCommand:        vertexCount, instanceCount, firstVertex, firstInstance
  const uint offset = a_args.drawId*8;
  indirectBuffer[offset + 0] = count;
  indirectBuffer[offset + 1] = instNum;
  indirectBuffer[offset + 2] = 0;
  indirectBuffer[offset + 3] = 0;
  indirectBuffer[offset + 4] = 0;
}
